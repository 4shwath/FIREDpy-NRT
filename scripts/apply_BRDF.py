import os
import glob
import argparse
from config import Config
from sen2nbar.nbar import nbar_SAFE

config = Config()

def apply_brdf_correction (fire_id, base_directory):
    """
    Apply BRDF correction on Sentinel-2 images located in the specified directory using the nbar_SAFE function 
    from the sen2nbar package.
    
    Parameters:
    - fire_id (str): The fire event ID corresponding to the images to be processed.
    - base_directory (str): The base directory where the fire event folders are located.

    Returns:
    - None

    Raises:
    - FileNotFoundError: If the specified base directory does not exist.

    Usage:
    This function assumes a directory structure where each fire event is stored in a separate folder named by its fire ID.
    Within each fire event folder, there should be a sub-folder named 'Sentinel' containing the Sentinel-2 images in
    the SAFE format.

    The function applies the BRDF correction to each of the SAFE files in the 'Sentinel' sub-folder for the specified fire event.
    It creates a new folder called "NBAR" whithin each SAFE folder where the BRDF corrected images for each implemented band is
    stored.

    Example:
    
    apply_brdf_correction('202', '/path/to/fire/events')
    """
    
    # Check if the file exists
    if not os.path.exists(base_directory):
        raise FileNotFoundError(f"The directory {base_directory} was not found.")
    
    image_dir = os.path.join(base_directory, fire_id, 'Sentinel')
    file_address = glob.glob(os.path.join(image_dir, '*.SAFE'))

    for file in file_address:
        print(f"applying BRDF (NBAR) correction for event ID {fire_id} and image {file.split(config.dir_sep)[-1]}")
        #     apply BRDF adjustment
        nbar_SAFE(file, quiet=False)


if __name__ == "__main__":
    # Example to run the script
    # python test_BRDF.py -id "173" -d "E:/Projects/UCB/FiredPy/data/Fire_events/"

    parser = argparse.ArgumentParser(description="Apply BRDF correction on Sentinel-2 images using sen2nbar packages.")
    parser.add_argument("-id", "--event_id", type=str, help="ID of the fire event for which satellite images are to be searched. This comes from the id field of the fire events shapefile.")
    parser.add_argument("-d","--directory", type=str, help="The base directory where Sentinel images are saved.")
    args = parser.parse_args()
    apply_brdf_correction(args.event_id, args.directory)



