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
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import average_precision_score

csv.field_size_limit(sys.maxsize);


def get_scalers(image_size, x_max, y_min):
    h, w = image_size  # they are flipped so that mask_for_polygons works correctly
    w_ = w * (w / (w + 1))
    h_ = h * (h / (h + 1))
    return w_ / x_max, h_ / y_min


def mask_for_polygons(polygons, image_size):
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


def scale_percentile(matrix):
    w, h, d = matrix.shape
    matrix = np.reshape(matrix, [w * h, d]).astype(np.float64)
    # Get 2nd and 98th percentile
    mins = np.percentile(matrix, 1, axis=0)
    maxs = np.percentile(matrix, 99, axis=0) - mins
    matrix = (matrix - mins[None, :]) / maxs[None, :]
    matrix = np.reshape(matrix, [w, h, d])
    matrix = matrix.clip(0, 1)
    return matrix


def mask_to_polygons(mask, epsilon=10., min_area=10.):
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


def train(train_ids):
    all_xs = np.array([0,0,0])
    all_ys = None

    for img_id in train_ids[0:2,]:
        print('Operating on ' + str(img_id))

        cur_polygons = df[(df['ImageId'] == img_id) & (df['ClassType'] == POLY_TYPE)].iloc[0]['MultipolygonWKT']
        train_polygons = shapely.wkt.loads(cur_polygons)

        # Load grid size
        x_max, y_min = gs[gs['ImageId'] == img_id].iloc[0,1:].astype(float)

        # Read image with tiff
        im_rgb = tiff.imread('input/three_band/{}.tif'.format(img_id)).transpose([1, 2, 0])
        im_size = im_rgb.shape[:2]

        x_scaler, y_scaler = get_scalers(im_size, x_max, y_min)
        train_polygons_scaled = shapely.affinity.scale(train_polygons, xfact=x_scaler, yfact=y_scaler, origin=(0, 0, 0))
        train_mask = mask_for_polygons(train_polygons_scaled, im_size)

        xs = im_rgb.reshape(-1, 3).astype(np.float32)
        ys = train_mask.reshape(-1)

        all_xs = np.vstack((all_xs, xs))
        all_ys = np.append(all_ys, ys)

    pipeline = make_pipeline(StandardScaler(), SGDClassifier(loss='log'))

    print('training...')
    # do not care about overfitting here
    pipeline.fit(xs, ys)

    return(pipeline)


def predict(test_ids, pipeline):

    d = []

    for img_id in test_ids:
        print('Operating on ' + str(img_id))
        # Load grid size
        x_max, y_min = gs[gs['ImageId'] == img_id].iloc[0,1:].astype(float)

        # Read image with tiff
        im_rgb = tiff.imread('input/three_band/{}.tif'.format(img_id)).transpose([1, 2, 0])
        im_size = im_rgb.shape[:2]

        x_scaler, y_scaler = get_scalers(im_size, x_max, y_min)
        xs = im_rgb.reshape(-1, 3).astype(np.float32)
        
        pred_ys = pipeline.predict_proba(xs)[:, 1]
        pred_mask = pred_ys.reshape(im_size) # 3348 3403
        threshold = 0.3
        pred_binary_mask = pred_mask >= threshold
        pred_polygons = mask_to_polygons(pred_binary_mask)
        pred_poly_mask = mask_for_polygons(pred_polygons, im_size)

        scaled_pred_polygons = shapely.affinity.scale(
        pred_polygons, xfact=1 / x_scaler, yfact=1 / y_scaler, origin=(0, 0, 0))

        dumped_prediction = shapely.wkt.dumps(scaled_pred_polygons)
        print('Prediction size: {:,} bytes'.format(len(dumped_prediction)))
        final_polygons = shapely.wkt.loads(dumped_prediction)
        d.append({'ImageId': img_id, 'ClassType': POLY_TYPE, 'MultipolygonWKT': final_polygons})

    return(pd.DataFrame(d))


#img_id = '6120_2_2'
POLY_TYPE = 1  # buildings

# Load train poly with shapely
train_polygons = None
df = pd.read_csv('input/train_wkt_v4.csv')
train_ids = df['ImageId'].unique()

# Load grid sizes
gs = pd.read_csv('input/grid_sizes.csv', names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)
test_ids = gs[~gs.isin(train_ids)].dropna()['ImageId']
test_ids = '6170_0_1'

pipeline = train(train_ids)
predictions = predict(test_ids, pipeline)

#predictions.to_csv('output/temp_logistic_buildings.csv', index = False)

#submission = pd.read_csv('input/sample_submission.csv')

#new_submission = submission
#new_submission.update(predictions['MultipolygonWKT'], overwrite=True)
#new_submission.to_csv('output/logistic_buildings.csv', index = False)




















# # Predictions on New Images
# pred_ys = pipeline.predict_proba(xs)[:, 1]
# print('average precision', average_precision_score(ys, pred_ys))
# pred_mask = pred_ys.reshape(train_mask.shape)

# threshold = 0.3
# pred_binary_mask = pred_mask >= threshold

# # check jaccard on the pixel level
# tp, fp, fn = (( pred_binary_mask &  train_mask).sum(),
#               ( pred_binary_mask & ~train_mask).sum(),
#               (~pred_binary_mask &  train_mask).sum())
# print('Pixel jaccard', tp / (tp + fp + fn))


# pred_polygons = mask_to_polygons(pred_binary_mask)
# pred_poly_mask = mask_for_polygons(pred_polygons, im_size)

# scaled_pred_polygons = shapely.affinity.scale(
#     pred_polygons, xfact=1 / x_scaler, yfact=1 / y_scaler, origin=(0, 0, 0))

# dumped_prediction = shapely.wkt.dumps(scaled_pred_polygons)
# print('Prediction size: {:,} bytes'.format(len(dumped_prediction)))
# final_polygons = shapely.wkt.loads(dumped_prediction)

# print('Final jaccard',
#       final_polygons.intersection(train_polygons).area /
#       final_polygons.union(train_polygons).area)