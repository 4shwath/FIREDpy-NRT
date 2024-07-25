import os
import requests
import getpass

def get_access_token(username: str, password: str) -> str:
    data = {
        "client_id": "cdse-public",
        "username": username,
        "password": password,
        "grant_type": "password",
        }
    try:
        r = requests.post("https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
        data=data,
        )
        r.raise_for_status()
    except Exception as e:
        raise Exception(
            f"Keycloak token creation failed. Reponse from the server was: {r.json()}"
            )
    return r.json()["access_token"]

class Config():

    def __init__(self):

        # [credential_sentinel]

        # Copernicus Data Space Ecosystem username and password
        self.username_sentinel = "ashwathram99@gmail.com"
        self.password_sentinel = "Redrocket99#"
        self.access_token = get_access_token(self.username_sentinel, self.password_sentinel)

        # [credential_landsat]

        # Landsat username and password
        self.username_landsat = "ashwathram99@gmail.com"
        self.password_landsat = "Redrocket99#"

        # [params]
        self.max_cloud_cover = 10
        self.delta_days_landsat = 70
        self.delta_days_sentinel = 40
        # Sentinel-2: S2MSI2A,S2MSI1C, S2MS2Ap
        self.producttype_sentinel = "S2MSI2A"

        # whether to download the scenes or not
        self.download_scenes = True
        self.save_footprints = False
        self.update_json = False
        # [directory]
        #self.data_dir = '/home/aramakrishnan/Documents/Firedpy/'
        self.data_dir = '/Bhaltos/ASHWATH/optical_data_NoCloudMask_4Ryan/SAFE_files/'
        # self.data_dir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'data')) # ("../data/") 

        if os.name == 'nt':  # Windows
            self.dir_sep = '\\'
        else:  # Linux, macOS, and other platforms
            self.dir_sep = '/'