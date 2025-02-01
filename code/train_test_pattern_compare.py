# compare patterns in training and testing
# plot the comparison of training and testing variables

# This process only analyzes data; we don't touch the model here.

import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from snowcast_utils import homedir, work_dir, test_start_date
import os
import pandas as pd
import matplotlib.pyplot as plt
from model_creation_et import selected_columns



def clean_train_df(data):
    """
    Clean and preprocess the training data.

    Args:
        data (pd.DataFrame): The training data to be cleaned.

    Returns:
        pd.DataFrame: Cleaned training data.
    """
    # data['date'] = pd.to_datetime(data['date'])
    # reference_date = pd.to_datetime('1900-01-01')
    # data['date'] = (data['date'] - reference_date).dt.days
    # data.replace('--', pd.NA, inplace=True)
    data.fillna(-999, inplace=True)
    print(data.describe())
    # Remove all the rows that have 'swe_value' as -999
    data = data[(data['swe_value'] != -999)]

    print("Get slope statistics")
    print(data["Slope"].describe())
  
    print("Get SWE statistics")
    print(data["swe_value"].describe())

    # data = data.drop('Unnamed: 0', axis=1)
    

    return data

import traceback

def compare(
    target_date: str = "test_date", 
    train_csv_path: str = None, 
    test_csv_path: str = None, 
    plot_path: str = None
):
    """
    Compare training and testing data and create variable comparison plots.

    Returns:
        None
    """
    print("Starting the comparison process...")
    try:
        # Set default paths if not provided
        if test_csv_path is None:
            test_csv_path = f'{work_dir}/testing_all_ready_for_check.csv'
        if train_csv_path is None:
            train_csv_path = f'{work_dir}/final_merged_data_3yrs_cleaned_v3_time_series_cumulative_v1.csv'
        if plot_path is None:
            plot_path = f'{work_dir}/var_comparison/{target_date}_final_comparison.png'

        print(f"Training CSV Path: {train_csv_path}")
        print(f"Testing CSV Path: {test_csv_path}")
        print(f"Plot Output Path: {plot_path}")

        # Read and clean training data
        print("Reading and cleaning training data...")
        tr_df = pd.read_csv(train_csv_path)
        tr_df = clean_train_df(tr_df)
        tr_df = tr_df[selected_columns+["lat", "lon"]]
        print(f"Training DataFrame loaded with shape: {tr_df.shape}")

        # Read testing data
        print("Reading testing data...")
        te_df = pd.read_csv(test_csv_path)
        selected_columns.remove("swe_value")
        test_features = selected_columns+["lat", "lon"]
        te_df = te_df[test_features]
        print(f"Testing DataFrame loaded with shape: {te_df.shape}")

        # Convert testing data to numeric
        print("Converting testing data to numeric...")
        te_df = te_df.apply(pd.to_numeric, errors='coerce')
        print(f"Testing DataFrame statistics:\n{te_df.describe()}")

        print(f"Training columns: {list(tr_df.columns)}")
        print(f"Testing columns: {list(te_df.columns)}")

        # Ensure output directory exists
        print("Ensuring the output directory exists...")
        var_comparison_plot_path = os.path.dirname(plot_path)
        if not os.path.exists(var_comparison_plot_path):
            os.makedirs(var_comparison_plot_path)
            print(f"Created directory: {var_comparison_plot_path}")

        # Prepare subplot dimensions
        print("Preparing subplot dimensions...")
        num_cols = len(tr_df.columns)
        new_num_cols = int(num_cols**0.5)  # Square grid
        new_num_rows = int(num_cols / new_num_cols) + 1
        print(f"Subplot grid: {new_num_rows} rows x {new_num_cols} columns")

        # Create a figure with multiple subplots
        print("Creating subplots...")
        fig, axs = plt.subplots(new_num_rows, new_num_cols, figsize=(24, 20))
        axs = axs.flatten()

        # Iterate over columns and create plots
        print("Generating plots for each column...")
        for i, col in enumerate(tr_df.columns):
            print(f"Processing column {i + 1}/{num_cols}: {col}")
            try:
                # Filter out 0 and -999 from training and testing data
                train_filtered = tr_df[col][~tr_df[col].isin([0, -999])]
                test_filtered = te_df[col][~te_df[col].isin([0, -999])] if col in te_df.columns else None

                # Plot the filtered data
                axs[i].hist(train_filtered, bins=100, alpha=0.5, color='blue', label='Train')
                if test_filtered is not None:
                    axs[i].hist(test_filtered, bins=100, alpha=0.5, color='red', label='Test')
                else:
                    print(f"Warning: Column '{col}' not found in testing data.")

                axs[i].set_title(f'{col}')
                axs[i].legend()
            except Exception as e:
                print(f"Error processing column '{col}': {e}")
                traceback.print_exc()

        # Adjust layout and save the plot
        print("Adjusting layout and saving the plot...")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        print(f"Comparison plot saved at: {plot_path}")

    except Exception as e:
        print(f"Error during comparison process: {e}")
        traceback.print_exc()

    print("Comparison process completed.")


def calculate_feature_colleration_in_training():
#   training_data_path = f'{work_dir}/final_merged_data_3yrs_cleaned_v3_time_series_cumulative_v1.csv'
    
    tr_df = pd.read_csv(training_data_path)
    tr_df = clean_train_df(tr_df)
  

if __name__ == "__main__":
    training_data_path = f"{homedir}/snotel_ghcnd_stations_4yrs_all_cols_log10.csv"
    target_date = "2025-01-01"
    test_csv_path = f'{work_dir}/test_data_predicted_latest_{target_date}.csv'
    plot_path = f"{homedir}/../plots/all_var_comparison_{target_date}.png"
    compare(
        target_date=target_date, 
        train_csv_path=training_data_path, 
        test_csv_path=test_csv_path, 
        plot_path=plot_path
    )


