import os
import urllib
import requests
from bs4 import BeautifulSoup

# download the NetCDF file from Idaho http site daily or the time period matching the MODIS period
# download site: https://www.northwestknowledge.net/metdata/data/

download_source = "https://www.northwestknowledge.net/metdata/data/"
gridmet_download_dir = "/home/chetana/terrian_data/"


def download_gridmet():
    if not os.path.exists(gridmet_download_dir):
        os.makedirs(gridmet_download_dir)

    soup = BeautifulSoup(requests.get(download_source).text, "html.parser")
    tag_links = soup.find_all('a')
    for t in tag_links:
        if '.nc' in t.text and not 'eddi' in t.text and not os.path.isfile(gridmet_download_dir + t.get("href")):
            print(f'downloading {t.get("href")}')
            urllib.request.urlretrieve(download_source + t.get('href'), gridmet_download_dir + t.get("href"))


download_gridmet()

