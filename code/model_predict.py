import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
from snowcast_utils import homedir, work_dir, model_dir, plot_dir, output_dir, month_to_season, test_start_date, test_end_date, process_dates_in_range
import os
import random
import string
import shutil
from model_creation_et import selected_columns
from datetime import datetime, timedelta
# from interpret_model_results import explain_predictions

import shap
import pandas as pd
import matplotlib.pyplot as plt
import traceback

COLUMN_NAME_MAPPER = {'Latitude': 'lat', 
                         'Longitude': 'lon',
                         'vpd': 'mean_vapor_pressure_deficit',
                         'vs': 'wind_speed', 
                         'pr': 'precipitation_amount', 
                         'etr': 'potential_evapotranspiration',
                         'tmmn': 'air_temperature_tmmn',
                         'tmmx': 'air_temperature_tmmx',
                         'rmin': 'relative_humidity_rmin',
                         'rmax': 'relative_humidity_rmax',
#                          'Elevation': 'elevation',
#                          'Slope': 'Slope',
#                          'Aspect': 'Aspect',
#                          'Curvature': 'Curvature',
#                          'Northness': 'Northness',
#                          'Eastness': 'Eastness',
                         'cumulative_AMSR_SWE': 'cumulative_SWE',
                         'cumulative_AMSR_Flag': 'cumulative_Flag',
                         'cumulative_tmmn':'cumulative_air_temperature_tmmn',
                         'cumulative_etr': 'cumulative_potential_evapotranspiration',
                         'cumulative_vpd': 'cumulative_mean_vapor_pressure_deficit',
                         'cumulative_rmax': 'cumulative_relative_humidity_rmax', 
                         'cumulative_rmin': 'cumulative_relative_humidity_rmin',
                         'cumulative_pr': 'cumulative_precipitation_amount',
                         'cumulative_tmmx': 'cumulative_air_temperature_tmmx',
                         'cumulative_vs': 'cumulative_wind_speed',
                         'AMSR_SWE': 'SWE',
                         'AMSR_Flag': 'Flag',
#                          'relative_humidity_rmin': '',
#                          'cumulative_rmin',
#                          'mean_vapor_pressure_deficit', 
#                          'cumulative_vpd', 
#                          'wind_speed',
#                          'cumulative_vs', 
#                          'relative_humidity_rmax', 'cumulative_rmax',

# 'precipitation_amount', 'cumulative_pr', 'air_temperature_tmmx',

# 'cumulative_tmmx', 'potential_evapotranspiration', 'cumulative_etr',

# 'air_temperature_tmmn', 'cumulative_tmmn', 'x', 'y', 'elevation',

# 'slope', 'aspect', 'curvature', 'northness', 'eastness', 'AMSR_SWE',

# 'cumulative_AMSR_SWE', 'AMSR_Flag', 'cumulative_AMSR_Flag',
}

COLUMN_LOOK_BACK = [
    'mean_vapor_pressure_deficit',
    'wind_speed', 
    'precipitation_amount', 
    'potential_evapotranspiration',
    'air_temperature_tmmn',
    'air_temperature_tmmx',
    'relative_humidity_rmin',
    'relative_humidity_rmax',
    'SWE',
    'fsca',
]

COLUMN_UNCHANGED = [
    'Aspect', 
    'Elevation', 
    'Curvature', 
    'Northness', 
    'Flag', 
    'x', 
    'Eastness', 
    'water_year', 
    'Slope', 
    'lc_prop3', 
    'y',
    'snodas_mask',
]

def generate_random_string(length):
    # Define the characters that can be used in the random string
    characters = string.ascii_letters + string.digits  # You can customize this to include other characters if needed

    # Generate a random string of the specified length
    random_string = ''.join(random.choice(characters) for _ in range(length))

    return random_string
  

def load_model(model_path):
    """
    Load a machine learning model from a file.

    Args:
        model_path (str): Path to the saved model file.

    Returns:
        model: The loaded machine learning model.
    """
    return joblib.load(model_path)

def load_data(file_path):
    """
    Load data from a CSV file.

    Args:
        file_path (str): Path to the CSV file containing the data.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the loaded data.
    """
    return pd.read_csv(file_path)


# 'SWE', 'relative_humidity_rmin', 'potential_evapotranspiration',
#     'air_temperature_tmmx', 'relative_humidity_rmax',
#     'mean_vapor_pressure_deficit', 'air_temperature_tmmn', 'wind_speed',
#     'Elevation', 'Aspect', 'Curvature', 'Northness', 'Eastness', 'fsca',
#     'Slope', 'SWE_1', 'air_temperature_tmmn_1',
#     'potential_evapotranspiration_1', 'mean_vapor_pressure_deficit_1',
#     'relative_humidity_rmax_1', 'relative_humidity_rmin_1',
#     'air_temperature_tmmx_1', 'wind_speed_1', 'fsca_1', 'SWE_2',
#     'air_temperature_tmmn_2', 'potential_evapotranspiration_2',
#     'mean_vapor_pressure_deficit_2', 'relative_humidity_rmax_2',
#     'relative_humidity_rmin_2', 'air_temperature_tmmx_2', 'wind_speed_2',
#     'fsca_2', 'SWE_3', 'air_temperature_tmmn_3',
#     'potential_evapotranspiration_3', 'mean_vapor_pressure_deficit_3',
#     'relative_humidity_rmax_3', 'relative_humidity_rmin_3',
#     'air_temperature_tmmx_3', 'wind_speed_3', 'fsca_3', 'SWE_4',
#     'air_temperature_tmmn_4', 'potential_evapotranspiration_4',
#     'mean_vapor_pressure_deficit_4', 'relative_humidity_rmax_4',
#     'relative_humidity_rmin_4', 'air_temperature_tmmx_4', 'wind_speed_4',
#     'fsca_4', 'SWE_5', 'air_temperature_tmmn_5',
#     'potential_evapotranspiration_5', 'mean_vapor_pressure_deficit_5',
#     'relative_humidity_rmax_5', 'relative_humidity_rmin_5',
#     'air_temperature_tmmx_5', 'wind_speed_5', 'fsca_5', 'SWE_6',
#     'air_temperature_tmmn_6', 'potential_evapotranspiration_6',
#     'mean_vapor_pressure_deficit_6', 'relative_humidity_rmax_6',
#     'relative_humidity_rmin_6', 'air_temperature_tmmx_6', 'wind_speed_6',
#     'fsca_6', 'SWE_7', 'air_temperature_tmmn_7',
#     'potential_evapotranspiration_7', 'mean_vapor_pressure_deficit_7',
#     'relative_humidity_rmax_7', 'relative_humidity_rmin_7',
#     'air_temperature_tmmx_7', 'wind_speed_7', 'fsca_7', 'water_year'

def preprocess_data_with_history():
    pass

def preprocess_chunk(chunk, day_offset):
    """
    Load, clean, and rename columns for a specific day.

    Args:
        file_path (str): Path to the CSV file.
        day_offset (int): Day offset (0 for current day, 1 for one day ago, etc.).

    Returns:
        pd.DataFrame: Processed DataFrame for the specific day.
    """
    if "date.1" in chunk.columns:
        chunk = chunk.drop(["date.1"], axis=1)
    chunk.replace('--', pd.NA, inplace=True)
    chunk.rename(columns=COLUMN_NAME_MAPPER, inplace=True)
    chunk['date'] = pd.to_datetime(chunk['date'])

    # print("Before drop: ", chunk.columns)
    if day_offset != 0:
        chunk.drop(COLUMN_UNCHANGED+["date"], axis=1, inplace=True)
        for col in COLUMN_LOOK_BACK:
            chunk.rename(
                columns={col: f"{col}_{day_offset}"}, inplace=True
            )

    # print("After drop: ", chunk.columns)
    return chunk

def preprocess_data(target_date, is_model_input: bool = True):
    """
    Preprocess the input data for model prediction.

    Args:
        target_date (str): Target date in the format 'YYYY-MM-DD'.
        is_model_input (bool): Flag to specify if the data is for model input.

    Returns:
        pd.DataFrame: Preprocessed data ready for prediction.
    """
    
    # Initialize a list to store all data including past 7 days
    all_data = []

    # Process the current day
    # current_day_path = f'{work_dir}/testing_all_ready_{target_date}.csv'
    # current_day_data = process_day_data(current_day_path, 0)
    # all_data.append(current_day_data)

    # Process the past 7 days
    target_date_dt = pd.to_datetime(target_date)
    for i in range(0, 7):
        past_date = (target_date_dt - pd.Timedelta(days=i)).strftime('%Y-%m-%d')
        past_data_path = f'{work_dir}/testing_all_ready_{past_date}.csv_snodas_mask.csv'
        past_day_data = process_day_data(past_data_path, i)
        all_data.append(past_day_data)

    # Merge all data on 'date', 'lat', and 'lon'
    merged_data = all_data[0]
    for additional_data in all_data[1:]:
        merged_data = merged_data.merge(additional_data, on=['date', 'lat', 'lon'], how='outer')

    if is_model_input:
        if "swe_value" in selected_columns:
            selected_columns.remove("swe_value")
        desired_order = selected_columns + ['lat', 'lon']

        merged_data = merged_data[desired_order]
        merged_data = merged_data.reindex(columns=desired_order)

        # print("Reorganized columns: ", merged_data.columns)

    # print(merged_data.head())

    return merged_data

def predict_swe(model, data):
    """
    Predict Snow Water Equivalent (SWE) values using a pre-trained model.

    This function takes in a machine learning model and a DataFrame containing 
    meteorological and geospatial data, preprocesses the data by handling missing 
    values and dropping unnecessary columns, and applies the model to predict SWE values. 
    The predicted SWE values are then added to the original DataFrame as a new column 
    called 'predicted_swe'.

    Args:
        model (object): A pre-trained machine learning model with a `predict` method.
        data (pd.DataFrame): A pandas DataFrame containing input data for prediction.
            It is expected to have columns including 'lat', 'lon', and other relevant 
            features for the model.

    Returns:
        pd.DataFrame: The original DataFrame with an additional column 'predicted_swe' 
        containing the predicted SWE values.
    """
    data = data.fillna(-1)
    input_data = data
    input_data = data.drop(["lat", "lon"], axis=1)

    print("Assign -1 to fsca column..")
    # original_input_data = input_data.copy()
    # input_data.loc[input_data['fsca'] > 100, 'fsca'] = -1 
    for column in input_data.columns:
        if 'fsca' in column.lower() or 'swe' in column.lower():  # Adjust to case-insensitive match
            input_data.loc[input_data[column] > 100, column] = -1

    #input_data = data.drop(['date', 'SWE', 'Flag', 'mean_vapor_pressure_deficit', 'potential_evapotranspiration', 'air_temperature_tmmx', 'relative_humidity_rmax', 'relative_humidity_rmin',], axis=1)
    #scaler = StandardScaler()

    # Fit the scaler on the training data and transform both training and testing data
    #input_data_scaled = scaler.fit_transform(input_data)
    print("Start to predict", input_data.shape)
    predictions = model.predict(input_data)
    input_data['predicted_swe'] = predictions
    input_data['lat'] = data['lat']
    input_data['lon'] = data['lon']

    # print("Explain the prediction: ")
    # explain_predictions(model, input_data, input_data.columns, f"{output_dir}/explain_ai.csv", f"{plot_dir}")
    return input_data

def merge_data(original_data, predicted_data):
    """
    Merge predicted SWE data with the original data.

    Args:
        original_data (pd.DataFrame): Original input data.
        predicted_data (pd.DataFrame): Dataframe with predicted SWE values.

    Returns:
        pd.DataFrame: Merged dataframe.
    """
    #new_data_extracted = predicted_data[["date", "lat", "lon", "predicted_swe"]]
    if "date" not in predicted_data:
    	predicted_data["date"] = test_start_date
    # new_data_extracted = predicted_data[["date", "lat", "lon", "predicted_swe"]]
    # print("original_data.columns: ", original_data.columns)
    # print("predicted_data.columns: ", predicted_data.columns)
    # print("new prediction statistics: ", predicted_data["predicted_swe"].describe())
    #merged_df = original_data.merge(new_data_extracted, on=["date", 'lat', 'lon'], how='left')
    merged_df = original_data.merge(predicted_data, on=['lat', 'lon'], how='left')
    # print("first merged df: ", merged_df.columns)

    # merged_df.loc[merged_df['fsca'] == -1, 'predicted_swe'] = 0
    # merged_df.loc[merged_df['fsca'] == 239, 'predicted_swe'] = 0
    # merged_df.loc[merged_df['fsca'] == 225, 'predicted_swe'] = 0
    #merged_df.loc[merged_df['cumulative_fsca'] == 0, 'predicted_swe'] = 0
    # merged_df.loc[merged_df['fsca'] <= 0, 'predicted_swe'] = 0

    # if predicted value is minus, assign 0
    merged_df.loc[merged_df['predicted_swe'] < 0, 'predicted_swe'] = 0
    
    merged_df.loc[merged_df['air_temperature_tmmx'].isnull(), 
                  'predicted_swe'] = 0

    merged_df.loc[merged_df['lc_prop3'] == 3, 'predicted_swe'] = 0
    merged_df.loc[merged_df['lc_prop3'] == 255, 'predicted_swe'] = 0
    merged_df.loc[merged_df['lc_prop3'] == 27, 'predicted_swe'] = 0

    return merged_df


def predict_in_batches(
    target_date: str, 
    output_path: str = None, 
    batch_size: int = 100000
):
    """
    Predict snow water equivalent (SWE) in batches by processing 7 days' data chunk by chunk.

    Args:
        target_date (str): Target date in the format 'YYYY-MM-DD'.
        output_path (str): Path to save the prediction results.
        batch_size (int): Size of each chunk to process.

    Returns:
        None
    """
    # height = 666
    # width = 694
    model_path = f'{model_dir}/wormhole_ETHole_latest.joblib'
    print(f"Using model: {model_path}")

    if output_path is None:
        output_path = f'{output_dir}/test_data_predicted_latest_{target_date}.csv_snodas_mask.csv'

    if os.path.exists(output_path):
        os.remove(output_path)
        print(f"File '{output_path}' has been removed.")

    # Load the model
    model = load_model(model_path)

    # Initialize file readers for each of the 7 days
    target_date_dt = pd.to_datetime(target_date)
    day_file_iters = []

    for day_offset in range(8):
        day_date = (target_date_dt - pd.Timedelta(days=day_offset)).strftime('%Y-%m-%d')
        day_file_path = f'{work_dir}/testing_all_ready_{day_date}.csv_snodas_mask.csv'

        try:
            print("Loading batches from ", day_file_path)
            day_file_iters.append(pd.read_csv(day_file_path, chunksize=batch_size))
        except FileNotFoundError:
            print(f"File not found: {day_file_path}. Skipping this day.")
            day_file_iters.append(None)

    # Process chunks
    chunk_idx = 0
    while True:
        chunk_list = []
        for day_idx, file_iter in enumerate(day_file_iters):
            if file_iter is None:
                continue
            try:
                chunk = next(file_iter)
                print(f"Read chunk {chunk_idx + 1} from day {day_idx + 1}")
                preprocessed_chunk = preprocess_chunk(chunk, day_offset=day_idx)
                chunk_list.append(preprocessed_chunk)
            except StopIteration:
                print(f"No more chunks for day {day_idx + 1}")
                continue

        # If no more chunks for all days, break
        if not chunk_list:
            print("All chunks are processed")
            break

        # Merge all chunks on 'date', 'lat', and 'lon'
        merged_input = chunk_list[0]
        for additional_chunk in chunk_list[1:]:
            merged_input = merged_input.merge(additional_chunk, on=['lat', 'lon'], how='outer')

        if len(merged_input) != len(chunk_list[0]):
            raise ValueError(
                f"Row number mismatch: merged_input has {len(merged_input)} rows, "
                f"but chunk_list[0] has {len(chunk_list[0])} rows. Ensure data alignment."
            )

        # print("merged_input.columns = ", merged_input.columns)

        # Reorganize columns for model input
        if "swe_value" in selected_columns:
            selected_columns.remove("swe_value")
        desired_order = selected_columns + ['lat', 'lon']
        used_input = merged_input[desired_order].reindex(columns=desired_order)
        unused_input = merged_input[["lc_prop3", "lat", "lon", "date"]]

        # Predict on the merged input
        predictions = predict_swe(model, used_input)
        # print(f"Predicted {len(predictions)} rows for chunk {chunk_idx + 1}")

        # Merge predictions with input
        predictions_merged = merge_data(unused_input, predictions)

        # Save predictions to output file incrementally
        if chunk_idx == 0:
            predictions_merged.to_csv(output_path, index=False, mode='w')
        else:
            predictions_merged.to_csv(output_path, index=False, mode='a', header=False)

        chunk_idx += 1

    print(f"Prediction completed. Results saved to {output_path}")

def predict_for_date(current_date, force: bool = False):
    """
    Example callback function to predict SWE for a specific date.

    Args:
        current_date (datetime): The date to process.
        force (bool): Whether to force processing even if conditions aren't met.
    """
    current_date_str = current_date.strftime("%Y-%m-%d")
    print(f">>>>>\nPredicting SWE for day {current_date_str}")
    # Replace this with actual prediction logic
    predict_in_batches(target_date=current_date_str,)

if __name__ == "__main__":
	process_dates_in_range(
        start_date=test_start_date,
        end_date=test_end_date,
        days_look_back=0,
        # start_date="2025-01-14",
        # end_date="2025-01-14",
        callback=predict_for_date,
        force = True
    )

