import os

class Config():

    def __init__(self):
        # [credential_landsat]
        # landsat API
        self.username_landsat = "YOUR_USERNAME"
        self.password_landsat = "YOUR_PASSWORD"


        # [credential_sentinel]
        # Copernicus Open Access Hub (SciHub) username and password
        self.username_sentinel = "YOUR_USERNAME"
        self.password_sentinel = "YOUR_PASSWORD"

        # [params]
        self.max_cloud_cover = 40
        self.delta_days_landsat = 70
        self.delta_days_sentinel = 40
        # Sentinel-2: S2MSI2A,S2MSI1C, S2MS2Ap
        self.producttype_sentinel = "S2MSI2A"

        # whether to download the scenes or not
        self.download_scenes = False
        self.save_footprints = True
        self.update_json = False
        # [directory]
        self.data_dir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'data')) # ("../data/") 

        if os.name == 'nt':  # Windows
            self.dir_sep = '\\'
        else:  # Linux, macOS, and other platforms
            self.dir_sep = '/'