# DSTL Satellite Imagery Feature Detection

## Setup Details

Python installation used is built using the Kaggle Docker installation described [here](http://blog.kaggle.com/2016/02/05/how-to-get-started-with-data-science-in-containers/)

## Process New
1. Put all input files in input/ directory
2. Create directories:
    * subm/ for submissions
    * msk/ for pixel masks
    * weights/ to save neural network weights while running
    * data/ to save validation and training data while running
3. Run all code from top-level directory:
    * `kpython run_unet_keras.py` to run Kaggle kernel as-is (M-band images with 8 bands)
    * `kpython run_unet_keras_by_class.py` to run class-wise predictions on 20-band images (512x512)
        * Set class in main function at bottom of file. Not yet optimized for running all classes in one run
        * Set 'slug' at top of file to some descriptive name. This will be used to save intermediary files without conflicts
    * `kpython run_unet_keras_named` 
        * Predicts all classes at once on 20-band images (512x512)
        * Set 'slug' at top of file to some descriptive name. This will be used to save intermediary files without conflicts
4. Once predictions are generated, you may run into submission errors. To correct these, you can use the parameters from the Kaggle error message to modify the repair_topology_exception.py function. Then:
    * `kpython repair_topology_exception.py`

## Process Old

1. Put all input files in an input/ directory
2. Within the input/ directory create a masks/ directory
3. Run the generate_masks.py file from the top level directory
    * Top level is required due to Docker permissioning
    * `kpython generate_masks.py` 
4. To Run Logistic Regression Classifier:
    * `kpython logistic.py`
    * Run R code in generate_solution_file.R
