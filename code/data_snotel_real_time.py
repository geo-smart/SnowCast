
import requests
import pandas as pd
from bs4 import BeautifulSoup

nohrsc_url_format_string = "https://www.nohrsc.noaa.gov/nearest/index.html?city={lat}%2C{lon}&county=&l=5&u=e&y={year}&m={month}&d={day} "
test_noaa_query_url = nohrsc_url_format_string.format(lat=40.05352381745094, lon=-106.04027196859343, year=2022, month=5, day=4)

print(f"url: {test_noaa_query_url}")

response = requests.get(test_noaa_query_url, headers={'User-Agent': 'Mozilla'})
parsed_html = BeautifulSoup(response.text, features='lxml')
container_div = parsed_html.find("div", attrs={'class': 'container'})
if container_div:
    container_div = container_div.get_text()
else:
    print("could not find div the class 'container'")

live_stats = parsed_html.find_all('table', attrs={'class': 'gray_data_table'})
tables = pd.read_html(str(live_stats))

table_sequence = [
        'Raw Snowfall Observations',
        'Snow Depth',
        'Snow Water Equivalent Observations',
        'Raw Precipitation Observations'
    ]
for idx, t in enumerate(tables):
    print(table_sequence[idx])
    print(t)
    print('--------------------')
