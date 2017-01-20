# DSTL Satellite Imagery Feature Detection

## Setup Details

Python installation used is built using the Kaggle Docker installation described [here](http://blog.kaggle.com/2016/02/05/how-to-get-started-with-data-science-in-containers/)

## Process

1. Put all input files in an input/ directory
2. Within the input/ directory create a masks/ directory
3. Run the generate_masks.py file from the top level directory
    * Top level is required due to Docker permissioning
    * `kpython generate_masks.py` 
