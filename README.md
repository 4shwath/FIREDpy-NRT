# Firedpy-NRT

This branch contains the code and outcome of training a deep learning model to perform semantic segmentation on an integrated dataset of SAR and optical satellite imagery.
This document highlights the uses of each script, how to adapt them for custom use, and the evaluation metrics of the trained model.
<br/><br/>

First, install any missing dependencies with  
`conda env create -f environment.yml`  
`conda activate myenv`  
OR  
`pip install -r requirements.txt`

The scripts can be split into three groups based on function:
1. Dataset creation
2. Model training
3. Model inference
4. Development scripts

## 1. Dataset creation

#### config.py  
Used to set Sentinel and Landsat credentials to access the Copernicus Data Space Ecosystem for downloading raster data in the form of SAFE files. You can also set download parameters for sentinel and landsat, and paths to input and output directories.

#### download_scenes.py  
Downloads SAFE files containing raster data.

To run it from the command line, use  
`python3 download_scenes.py --event_id 9844 --satellite sentinel`  
OR  
`python3 download_scenes.py -id 9844 -s sentinel`  

Replace *9844* with your desired fire event id and *sentinel* with your desired satellite (sentinel or landsat).  

#### download_all_scenes.sh
This bash script can optionally be used to download raster data for an entire list of fire events. To customize it, simply replace *fire_ids* with your own list of ids, and run it from the command line with `./download_all_scenes.sh`  

> Note: The script will retry downloads up to 3 times in the case of failure, but this does not guarantee a successful download.

#### dataset_generation.ipynb
Creates a dataset of geotiff files from the downloaded SAFE files for a specific fire event and satellite. The dataset will consist of dNBR and mask files, and optionally RGB files. The output directories, fire event id and satellite can be set at the start of the notebook. Then, simply click 'run all' or manually run all code cells to obtain the output directories for dNBR and masks. In the last code cell, `get_RGB = False` by default. Change it to True before clicking 'run all' to obtain an output directory for the RGB files. 

>Note: The code currently expects the output directories to already exist and will not create them if not found.

#### organize_data.py
This script uses a portion of code from *organize_data_dev.ipynb*. It requires 4 directories, consisting of geotiff files, namely, SAR luminance change, SAR coherence change, dNBR and SAR binary product. All 4 directory paths can be set near the bottom of the file, while the output file path can be set in *csv_file_path*.  The output of this script is a csv file containing 'matches' of images from SAR luminance change, SAR coherence change, dNBR and SAR binary product. A 'match' is when corresponding files refer to the same fire event and have exact matching dates. In the absence of exact matching dates, the file with the closest dates is found. 

> Note: There is no option for a date threshold in this script. Please refer to *organize_data_dev.ipynb* for code which contains a date threshold. Note also, that currently only Sentinel-2 dNBR images are considered for matches, while Landsat images are not.


## 2. Model training

#### training.py
This script uses a portion of code from training_integrated_data_dev.ipynb. It loads data from *csv_file*, preprocesses it and creates input arrays by stacking SAR luminance change, SAR coherence change and dNBR. It creates segmentation masks from SAR binary product. Then it trains a deep learning model using DeepLabV3 with a ResNet50 backbone. The code uses distributed GPU training and several training optimizations for extracting maximum efficiency out of the hardware it was trained on. This was largely due to hardware limitations, particularly CUDA out of memory errors, when attempting model training. Note that the code resumes model training from the latest available checkpoint in *checkpoint_dir* if any are available. It also creates an output csv file for the training and test metrics, for which the path can be set by changing *metrics_output_path*.

## 3. Model inference

#### model_inference.py
Loads a model checkpoint and a new set of data from a csv file, repeats the process from *training.py* to create input arrays and segmentation masks, runs the model over the new data to obtain predictions and saves the predictions as h5 files. The checkpoint path, csv file path and output directory can all be set towards the bottom of the code.

## Extra
*utils.py* contains some helper functions used in *dataset_generation.ipynb*.  
*organize_data_dev.ipynb* is a notebook used to develop the code in *organize_data.py*.  
*training_integrated_data_dev.ipynb* is a notebook used to develop the code in *training_integrated_data.py*.  

## Metrics for *best_model_checkpoint*
**Train Loss**: 0.8183, **Test Loss**: 0.6836, **IoU**: 0.3162, **Accuracy**: 0.6196
