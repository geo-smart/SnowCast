import math
import json
import requests
import pandas as pd


def read_json_file(file_path):
  with open(file_path, 'r', encoding='utf-8-sig') as json_file:
    data = json.load(json_file)
    return data


def haversine(lat1, lon1, lat2, lon2):
  lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
  d_lat = lat2 - lat1
  d_long = lon2 - lon1
  a = math.sin(d_lat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(d_long / 2) ** 2
  c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
  distance = 6371 * c  # Earth's radius in kilometers
  return distance


def find_nearest_location(locations, target_lat, target_lon):
  n_location = None
  min_distance = float('inf')
  for location in locations:
    lat = location['location']['lat']
    lon = location['location']['lng']
    distance = haversine(lat, lon, target_lat, target_lon)
    if distance < min_distance:
      min_distance = distance
      n_location = location
      return n_location


def csv_to_json(csv_text):
  lines = csv_text.splitlines()
  header = lines[0]
  field_names = header.split(',')
  json_list = []
  for line in lines[1:]:
    values = line.split(',')
    row_dict = {}
    for i, field_name in enumerate(field_names):
      row_dict[field_name] = values[i]
      json_list.append(row_dict)
      json_string = json.dumps(json_list)
      return json_string


def remove_commented_lines(text):
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        if not line.startswith('#'): 
            cleaned_lines.append(line)
    cleaned_text = "\n".join(cleaned_lines)
    return cleaned_text

csv_file = '/home/chetana/gridmet_test_run/training_data_ready_snotel.csv'
start_date = "2002-01-01"
end_date = "2023-12-12"

station_mapping = pd.read_csv('/home/chetana/gridmet_test_run/station_cell_mapping.csv')
df = pd.DataFrame(columns=['Date', 'Snow Water Equivalent (in) Start of Day Values',
                           'Change In Snow Water Equivalent (in)',
                           'Snow Depth (in) Start of Day Values',
                           'Change In Snow Depth (in)',
                           'Air Temperature Observed (degF) Start of Day Values',
                           'station_name',
                           'station_triplet',
                           'station_elevation',
                           'station_lat',
                           'station_long',
                           'mapping_station_id',
                           'mapping_cell_id']
                  )

for index, row in station_mapping.iterrows():
    station_locations = read_json_file('/home/chetana/gridmet_test_run/snotelStations.json')
    nearest_location = find_nearest_location(station_locations, 41.993149, -120.1787155)

    location_name = nearest_location['name']
    location_triplet = nearest_location['triplet']
    location_elevation = nearest_location['elevation']
    location_station_lat = nearest_location['location']['lat']
    location_station_long = nearest_location['location']['lng']

    url = f"https://wcc.sc.egov.usda.gov/reportGenerator/view_csv/" \
          f"customSingleStationReport/daily/{location_triplet}%7Cid%3D%22%22%7Cname/{start_date},{end_date}%2C0/" \
          "WTEQ%3A%3Avalue%2CWTEQ%3A%3Adelta%2CSNWD%3A%3Avalue%2CSNWD%3A%3Adelta%2CTOBS%3A%3Avalue"

    r = requests.get(url)
    text = remove_commented_lines(r.text)
    json_data = json.loads(csv_to_json(text))

    for item in json_data:
        item['lat'] = row['lat']
        item['lon'] = row['lon']

    with open(csv_file, 'a') as f:
        for entry in json_data:
            pd.DataFrame(entry, index=[0]).to_csv(f, header=True, index=False)

