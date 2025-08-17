from datetime import datetime, timedelta
import os
import subprocess
from snowcast_utils import data_dir, work_dir, output_dir

def generate_links(start_year, end_year):
    '''
    Generate a list of download links for AMSR daily snow data files.

    Args:
        start_year (int): The starting year.
        end_year (int): The ending year (inclusive).

    Returns:
        list: A list of download links for AMSR daily snow data files.
    '''
    base_url = "https://n5eil01u.ecs.nsidc.org/AMSA/AU_DySno.001/"
    date_format = "%Y.%m.%d"
    delta = timedelta(days=1)

    start_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year + 1, 1, 1)

    links = []
    current_date = start_date

    while current_date < end_date:
        date_str_path = current_date.strftime("%Y.%m.%d")  # Format for URL path (YYYY.MM.DD)
        date_str_file = current_date.strftime("%Y%m%d")    # Format for filename (YYYYMMDD)

        link = base_url + date_str_path + "/AMSR_U2_L3_DailySnow_B02_" + date_str_file + ".he5"
        # AMSR_U2_L3_DailySnow_B02_20120702
        links.append(link)
        current_date += delta

    return links

if __name__ == "__main__":
    start_year = 2012
    end_year = 2024

    links = generate_links(start_year, end_year)
    save_location = f"{data_dir}/amsr_testing/raw/"
    link_file = f"{data_dir}/amsr_testing/download_links.txt"

    with open(link_file, "w") as txt_file:
      for l in links:
        txt_file.write(" ".join(l) + "\n")

    print(f"{link_file} is downloaded")
    if not os.path.exists(save_location):
       os.makedirs(save_location)

    for link in links:
        filename = link.split("/")[-1]
        save_path = os.path.join(save_location, filename)
        # if os.path.exists(save_path):
        #     continue;
        
        curl_cmd = f"curl -s --output {save_path} -b ~/.urs_cookies -c ~/.urs_cookies -L -n -O {link}"
        # print(curl_cmd)
        subprocess.run(curl_cmd, shell=True, check=True)
        print(f"Downloaded: {save_path}")
        # break;

    # find /home/chetana/data/amsr_testing/raw -type f -size -200c -print -delete
    # remove the files that are less than 200bytes
    
