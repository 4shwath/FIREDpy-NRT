import os
import requests

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
        self.username_sentinel = '<username>'
        self.password_sentinel = '<password>'
        self.access_token = get_access_token(self.username_sentinel, self.password_sentinel)

        # [credential_landsat]
        # Landsat username and password
        self.username_landsat = '<username>'
        self.password_landsat = '<password>'

        # [params]
        self.max_cloud_cover = 10
        self.delta_days_landsat = 70    # delta_days refers to the number of days to include before and after a fire period
        self.delta_days_sentinel = 40
        self.producttype_sentinel = 'S2MSI2A'

        # whether to download the scenes or not
        self.download_scenes = True
        # [directory]
        self.data_dir = '<set_path_to_output_directory>'     # SAFE files will be downloaded into this output directory
        self.shp_file = f'{self.data_dir}/<set_filename_for_shapefile>'    # Fire event data will be read from this input file

        if os.name == 'nt':  # Windows
            self.dir_sep = '\\'
        else:  # Linux, macOS, and other platforms
            self.dir_sep = '/'
