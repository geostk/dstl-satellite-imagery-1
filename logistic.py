from collections import defaultdict
import csv
import sys
import cv2
from shapely.geometry import MultiPolygon, Polygon
import shapely.wkt
import shapely.affinity
import numpy as np
import tifffile as tiff
import matplotlib.pyplot
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import average_precision_score

csv.field_size_limit(sys.maxsize);


def get_scalers(image_size, x_max, y_min):
    h, w = image_size  # they are flipped so that get_polygon_mask works correctly
    w_ = w * (w / (w + 1))
    h_ = h * (h / (h + 1))
    return w_ / x_max, h_ / y_min


def get_polygon_mask(polygons, image_size):
    img_mask = np.zeros(image_size, np.uint8)
    if not polygons:
        return img_mask
    int_coords = lambda x: np.array(x).round().astype(np.int32)
    exteriors = [int_coords(poly.exterior.coords) for poly in polygons]
    interiors = [int_coords(pi.coords) for poly in polygons
                 for pi in poly.interiors]
    cv2.fillPoly(img_mask, exteriors, 1)
    cv2.fillPoly(img_mask, interiors, 0)
    return img_mask


def convert_mask_to_polygons(mask, epsilon=10., min_area=10.):
    # first, find contours with cv2: it's much faster than shapely
    image, contours, hierarchy = cv2.findContours(
        ((mask == 1) * 255).astype(np.uint8),
        cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
    # create approximate contours to have reasonable submission size
    approx_contours = [cv2.approxPolyDP(cnt, epsilon, True)
                       for cnt in contours]
    if not contours:
        return MultiPolygon()
    # now messy stuff to associate parent and child contours
    cnt_children = defaultdict(list)
    child_contours = set()
    assert hierarchy.shape[0] == 1
    # http://docs.opencv.org/3.1.0/d9/d8b/tutorial_py_contours_hierarchy.html
    for idx, (_, _, _, parent_idx) in enumerate(hierarchy[0]):
        if parent_idx != -1:
            child_contours.add(idx)
            cnt_children[parent_idx].append(approx_contours[idx])
    # create actual polygons filtering by area (removes artifacts)
    all_polygons = []
    for idx, cnt in enumerate(approx_contours):
        if idx not in child_contours and cv2.contourArea(cnt) >= min_area:
            assert cnt.shape[1] == 1
            poly = Polygon(
                shell=cnt[:, 0, :],
                holes=[c[:, 0, :] for c in cnt_children.get(idx, [])
                       if cv2.contourArea(c) >= min_area])
            all_polygons.append(poly)
    # approximating polygons might have created invalid ones, fix them
    all_polygons = MultiPolygon(all_polygons)
    if not all_polygons.is_valid:
        all_polygons = all_polygons.buffer(0)
        # Sometimes buffer() converts a simple Multipolygon to just a Polygon,
        # need to keep it a Multi throughout
        if all_polygons.type == 'Polygon':
            all_polygons = MultiPolygon([all_polygons])
    return all_polygons


def train(train_ids, class_id, polys, gs, patch_size):
    print('TRAINING...')

    # First calculate standard scaler parameters
    scaler = StandardScaler()
    for img_id in train_ids:
        im_rgb = tiff.imread('input/three_band/{}.tif'.format(img_id)).transpose([1, 2, 0])
        patches = extract_patches_2d(im_rgb, patch_size)
        patches = np.reshape(patches, (len(patches), -1))
        #xs = im_rgb.reshape(-1, 3).astype(np.float32)
        xs = patches.astype(np.float32)
        scaler.partial_fit(xs)

    # Next build the logistic model
    model = SGDClassifier(loss='log')
    for img_id in train_ids:
        print('Training on ' + str(img_id) + ' for class ' + str(class_id))

        # Load grid size for current image polygon coordinates
        x_max, y_min = gs[gs['ImageId'] == img_id].iloc[0,1:].astype(float)

        # Read current image with tiff
        im_rgb = tiff.imread('input/three_band/{}.tif'.format(img_id)).transpose([1, 2, 0])
        im_size = im_rgb.shape[:2]
        patches = extract_patches_2d(im_rgb, patch_size)
        print(len(patches))
        patches = np.reshape(patches, (len(patches), -1))


        # Read in polygons for current image
        cur_polygons = polys[(polys['ImageId'] == img_id) & (polys['ClassType'] == class_id)].iloc[0]['MultipolygonWKT']
        train_polygons = shapely.wkt.loads(cur_polygons)

        
        x_scaler, y_scaler = get_scalers(im_size, x_max, y_min)
        train_polygons_scaled = shapely.affinity.scale(train_polygons, xfact=x_scaler, yfact=y_scaler, origin=(0, 0, 0))
        train_mask = get_polygon_mask(train_polygons_scaled, im_size)

        # Load xs from image and ys from polygon mask
        #xs = im_rgb.reshape(-1, 3).astype(np.float32)
        xs = patches.astype(np.float32)
        ys = train_mask[0:-1, 0:-1].reshape(-1) # Drop last row and column to account for grid size
        #ys = train_mask.reshape(-1)

        # Scale x values with trained scaler
        #print(xs.mean(axis=0))
        xs = scaler.transform(xs)
        print(im_rgb.shape)
        print(xs.shape)
        print(ys.shape)
        #print(xs.mean(axis=0))

        print('training partial fit...')
        model.partial_fit(xs, ys, classes = (0, 1))

    return scaler, model


def predict(test_ids, class_id, gs, scaler, model, patch_size):

    d = []

    for img_id in test_ids:
        print('Predicting on ' + str(img_id) + ' for class ' + str(class_id))
        
        # Load grid size for current image polygon coordinates
        x_max, y_min = gs[gs['ImageId'] == img_id].iloc[0,1:].astype(float)

        # Read current image with tiff
        im_rgb = tiff.imread('input/three_band/{}.tif'.format(img_id)).transpose([1, 2, 0])
        im_size = im_rgb.shape[:2]
        patches = extract_patches_2d(im_rgb, patch_size)
        patches = np.reshape(patches, (len(patches), -1))

        # Scale x values with trained scaler
        #xs = im_rgb.reshape(-1, 3).astype(np.float32)
        xs = patches.astype(np.float32)
        xs = scaler.transform(xs)
        
        # Predict pixel-level probabilities and apply 0.3 threshold for binary pixel mask
        pred_ys = model.predict_proba(xs)[:, 1]
        #pred_mask = pred_ys.reshape(im_size) # 3348 3403
        
        temp_mask = pred_ys.reshape(np.subtract(im_size, 1)) # To deal with patch indexing
        pred_mask = np.zeros((temp_mask.shape[0]+1, temp_mask.shape[1]+1)) # Pad with additional row and column of zeros
        pred_mask[:-1,:-1] = temp_mask # Pad with additional row and column of zeros

        threshold = 0.3
        pred_binary_mask = pred_mask >= threshold

        # Convert pixel-level mask to polygon mask
        pred_polygons = convert_mask_to_polygons(pred_binary_mask)

        # Scale polygon values from image coordinates to grid size coordinates
        x_scaler, y_scaler = get_scalers(im_size, x_max, y_min)
        scaled_pred_polygons = shapely.affinity.scale(pred_polygons, xfact=1 / x_scaler, yfact=1 / y_scaler, origin=(0, 0, 0))

        # Convert polygons to WKT format
        dumped_prediction = shapely.wkt.dumps(scaled_pred_polygons)
        final_polygons = shapely.wkt.loads(dumped_prediction)
        print({'ImageId': img_id, 'ClassType': class_id, 'MultipolygonWKT': final_polygons})
        d.append({'ImageId': img_id, 'ClassType': class_id, 'MultipolygonWKT': final_polygons})

    return(pd.DataFrame(d))


class_ids = [1]
patch_size = (2, 2)
#class_ids = (1,2,3,4,5,6,7,8,9,10)

#train_polygons = None
polys = pd.read_csv('input/train_wkt_v4.csv')

# Load grid sizes
gs = pd.read_csv('input/grid_sizes.csv', names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)

# Extract train and test IDs from data
train_ids = polys['ImageId'].unique()
test_ids = gs[~gs.isin(train_ids)].dropna()['ImageId']

# Train model by class & Generate Predictions
predictions = pd.DataFrame()
for class_id in class_ids:
    scaler, model = train(train_ids, class_id, polys, gs, patch_size)
    new_predictions = predict(test_ids, class_id, gs, scaler, model, patch_size)
    predictions = pd.concat([predictions, new_predictions])

predictions.to_csv('output/temp_logistic_buildings_all_train_scaled_patches_5_5.csv', index = False)

