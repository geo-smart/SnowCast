from datetime import datetime
from metloom.pointdata import SnotelPointData


# Write first python in Geoweaver
import os
import urllib.request, urllib.error, urllib.parse
import sys
print(sys.path)

try:
    from BeautifulSoup import BeautifulSoup
except ImportError:
    from bs4 import BeautifulSoup

nohrsc_url_format_string = "https://www.nohrsc.noaa.gov/nearest/index.html?city={lat}%2C{lon}&county=&l=5&u=e&y={year}&m={month}&d={day}"

test_noaa_query_url = nohrsc_url_format_string.format(lat=40.05352381745094, lon=-106.04027196859343, year=2022, month=5, day=4)

print(test_noaa_query_url)

response = urllib.request.urlopen(test_noaa_query_url)
webContent = response.read().decode('UTF-8')

print(webContent)
parsed_html = BeautifulSoup(webContent,features='lxml')
# Check if the div element was found before extracting its text content
container_div = parsed_html.find('div', attrs={'class': 'container'})
if container_div is not None:
    container_text = container_div.get_text()
    print(container_text)
else:
    print("Could not find div with class 'container'")


print(parsed_html.body.find('div', attrs={'class':'container'}).text)



#snotel_point = SnotelPointData("713:CO:SNTL", "MyStation")
#df = snotel_point.get_daily_data(
#    datetime(2020, 1, 2), datetime(2020, 1, 20),
#    [snotel_point.ALLOWED_VARIABLES.SWE]
#)
#print(df)
