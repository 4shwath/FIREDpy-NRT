# FIREDpy-NRT
![image](https://github.com/earthlab/FIREDpy-NRT/assets/67020853/ad384d49-2111-4d84-9821-83a492321017)
    
![FIREDpy - NRT - Page 2](https://github.com/earthlab/FIREDpy-NRT/assets/67020853/e56abc54-99d4-490a-ade0-994b63c8fbfb)
Project funded by NASA Applied Sciences Program and developed by ESOC and Earth Lab, CIRES - CU Boulder


## Fist Stage (Summer 2023)
### Done by [Behzad Vahedi](mailto:behzad@colorado.edu)

Our project aims to extende the EarthLab’s "Fire Event Delineation for Python (FIREDpy)" library by updating it for Near-Real-Time (NRT) fire event perimeter mapping through the fusion of optical and radar remote sensing data. Our main motivations are twofold: to achieve closer-to-real-time temporal resolution and to improve the spatial resolution of the FIREDpy outputs. 

In the first stage of the project, we focused on answering two questions:
- Is applying BRDF correction necessary to generated NRT fire perimeters?
- How does fire affect the signal in Sentinel-2 and Landsat-8 images and how do these satellites compare?

### BERDF Correction
The flowchart below demonstrates the steps I took to answer the first question, with the caveat that I only tested BRDF on Sentinel-2 images (using the Sen2nbar package) since I couldn't find any python implementation of this correction for Landsat images.

<img src="./images/Flowchart.jpg" width=800>

__Now, let's go through the different steps in the flowchart__

The steps in the first row of this flowchart are done using the [FiredPy](https://github.com/earthlab/firedpy) package. This resulted in a shapefile of 20 fire events in the Americas which you can find in the [fire_events](./fire_events/) folder.

Once the fire events are extracted, you need to __get pre- and post-fire image pairs__ for each event. This is done using the [optical_scenes.py](./scripts/optical_scenes.py) script. This script uses the footprint of the fire as well as its start and end date to search for optical images. It can download the images and/or save their footprints. You can run this script using the command below:

`python optical_scenes.py --event_id "202" --satellite "sentinel"`

or alternatively, 

`python optical_scenes.py -id "202" -s "sentinel"`

In these commands, the value of the `event_id` (or `id`) argument must be the ID of the fire event for which satellite images are to be searched. This comes from the id field of the [fire events shapefile](./fire_events/Fire_events.zip). The value of the `satellite` (or `s`) argument must be the name of the satellite, either "sentinel" or "landsat".

You can set or change the search criteria in the `config.py` file. These parameters are:
- `max_cloud_cover`: Maximum cloud cover percentage. default: 40%.
- `delta_days_landsat`: The number of days to use as a buffer before and after the fire event when searching for Landsat images. default: 70 days.
- `delta_days_sentinel`: The number of days to use as a buffer before and after the fire event when searching for Sentinel-2 images. default: 40 days.
- `download_scenes`: Whether to download images or not. default: False. __CAUTION: depending on the number of the images found, this could take a long time and might reqiure a huge amount of storage__
- `save_footprints`: Whether to save image footprints or not. default: False.
- `update_json`: deprecated.
- `data_dir`: The directory where the images and/or their footprints will be saved. See below for more information.

#### Directory Organization
`data_dir` is the directory where the you would want the images to be saved. Once you create this directory, extract (unzip) the `Fire_events.zip` there. This will (should) create a new subdirectory called Fire_events. Per each fire id you run the `optical_scenes.py` script for, a new subdirectory will be created within "Fire_events". Also, depending on what the `-s` parameter is, the corresponding subdirectory will be created within the fire_id folder where the footprints and images will be saved.

<img src="./images/directory_tree.png" width="300">


#### Important Note:
Before running the `optical_scenes.py` script, you should create separate accounts for Landsat and Sentinel API and set your username and password in the corresponidng variables in the `config.py` file.

### Optical Signal Evaluation