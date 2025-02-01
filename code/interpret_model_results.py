# do real interpretation for the model results and find real reasons for bad predictions
# prevent aimless and headless attempts that are just wasting time.
# this is an essential step in the loop

import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from snowcast_utils import work_dir, test_start_date, month_to_season, output_dir, plot_dir, model_dir
import os
from sklearn.inspection import partial_dependence,PartialDependenceDisplay
import shap
import matplotlib.pyplot as plt
from model_creation_et import selected_columns
import traceback

import shap
import pandas as pd
import matplotlib.pyplot as plt

feature_names = None

def load_model(model_path):
    return joblib.load(model_path)

def load_data(file_path):
    return pd.read_csv(file_path)

def explain_predictions(
    model, input_data, feature_names, output_csv, name, plot_path
):
    """
    Explains predictions using SHAP, saves the explanations into a CSV file,
    and generates SHAP plots for each row.

    Parameters:
    - model: Trained tree-based model (e.g., RandomForest, LightGBM, XGBoost)
    - input_data: Input data as a numpy array or pandas DataFrame
    - feature_names: List of feature names
    - output_csv: Path to save the explanation report as a CSV
    - plot_path: Directory to save the SHAP plots
    """
    print("Starting the explanation process...")

    # Ensure input_data is a DataFrame
    if not isinstance(input_data, pd.DataFrame):
        print("Converting input data to a DataFrame...")
        input_data = pd.DataFrame(input_data, columns=feature_names)

    input_data = input_data.apply(pd.to_numeric, errors='coerce')
    # Identify non-numeric columns
    print("input_data = ", input_data)
    non_numeric_columns = [
        col for col in input_data.columns if not pd.api.types.is_numeric_dtype(input_data[col])
    ]

    if non_numeric_columns:
        print("Non-numeric data found in the following columns:")
        for col in non_numeric_columns:
            print(f"  - {col}")

        # Attempt to convert non-numeric columns to numeric
        for col in non_numeric_columns:
            try:
                print(f"Converting column '{col}' to numeric...")
                input_data[col] = pd.to_numeric(input_data[col], errors='coerce')
                if input_data[col].isna().any():
                    print(f"Replacing NaN values in column '{col}' with 0.")
                    input_data[col].fillna(0, inplace=True)
            except Exception as e:
                raise ValueError(f"Failed to convert column '{col}' to numeric: {e}")

    print("Double check .. ")
    non_numeric_columns = [
        col for col in input_data.columns if not pd.api.types.is_numeric_dtype(input_data[col])
    ]
    if non_numeric_columns:
        raise ValueError("WTF")

    print("Initializing SHAP explainer...")
    explainer = shap.TreeExplainer(model)

    # Calculate SHAP values for the data
    print(f"Calculating SHAP values for {len(input_data)} samples...")
    shap_values = explainer.shap_values(input_data)

    # Handle single-output vs multi-output models
    if isinstance(shap_values, list):
        print("Multi-output model detected. Using the first output for SHAP values.")
        shap_values = shap_values[0]

    # Store explanations
    explanations = []

    print(f"Processing {len(input_data)} rows...")
    for i in range(len(input_data)):
        print(f"Explaining row {i + 1}/{len(input_data)}...")

        try:
            # Get SHAP values for the current row
            shap_row = shap_values[i]
            prediction = model.predict(input_data.iloc[[i]])[0]

            # Collect all feature contributions for the row
            feature_contributions = {
                feature_names[j]: shap_row[j] for j in range(len(feature_names))
            }

            # Append explanation to the list
            explanations.append({
                "Row": i,
                "Prediction": prediction,
                **feature_contributions
            })

            # Generate and save SHAP waterfall plot
            print("Checking for non-numeric columns in input_data...")

            # Generate and save SHAP waterfall plot
            data_row = input_data.iloc[i].values.astype(float)

            # Ensure shap_row is 1D
            shap_row = np.ravel(shap_row)  # Flatten it if necessary

            # Ensure data_row is also 1D
            data_row = np.ravel(data_row)

            # Ensure base_values is a scalar (for single-output models)
            base_value = explainer.expected_value

            print(f"Shape of shap_row: {shap_row.shape}")  # Should be (n_features,)
            print(f"Shape of data_row: {data_row.shape}")  # Should be (n_features,)
            print(f"Number of features: {len(feature_names)}")  # Should match the previous two
            # Check if base_value is a scalar (single-output model) or array (multi-output model)
            if isinstance(base_value, np.ndarray):
                # For multi-output models, select the correct base value (e.g., base_value[0] for the first output)
                base_value = base_value[0]  # Adjust index if necessary for your model
                print(f"Using base_value from multi-output model: {base_value}")
            else:
                print(f"Using base_value for single-output model: {base_value}")

            shap_results = shap.Explanation(
                values=shap_row,
                base_values=base_value,
                data=data_row,
                feature_names=feature_names
            )

            print("shap_results = ", shap_results)
            print("SHAP Explanation object created successfully.")
            plot_file = f"{plot_path}_row_{i + 1}.png"
            shap.waterfall_plot(shap_results, max_display=len(feature_names))
            plt.title(
                f"SHAP Waterfall Plot {name}", 
                fontsize=14
            )
            plt.savefig(plot_file, bbox_inches="tight")
            plt.close()
            print(f"Plot saved: {plot_file}")

        except Exception as e:
            print("Detailed traceback:")
            traceback.print_exc()
            print(f"Failed to generate plot for row {i + 1}: {e}")
            continue

    # Save explanations to a CSV file
    explanations_df = pd.DataFrame(explanations)
    try:
        print(f"Saving explanations to {output_csv}...")
        explanations_df.to_csv(output_csv, index=False)
        print(f"Explanations successfully saved to {output_csv}")
    except Exception as e:
        print(f"Failed to save explanations CSV: {e}")

    print(f"Process completed. Explanations saved to {output_csv}. Plots saved in {plot_path}.")

def explain_predictions_for_latlon(
    lat, lon, target_date,
    model_path: str = f'{model_dir}/wormhole_ETHole_latest.joblib'
):
    """
    Explains predictions for a specific location using SHAP, saves the explanation report,
    and generates a SHAP plot.

    Parameters:
    - lat (float): Latitude of the target location.
    - lon (float): Longitude of the target location.
    - target_date (str): Target date in the format 'YYYY-MM-DD'.
    - predicted_csv (str): Path to the CSV file with predictions.
    - model: Trained tree-based model (e.g., RandomForest, LightGBM, XGBoost).
    - feature_names: List of feature names.
    - output_csv (str): Path to save the explanation report as a CSV.
    - output_plots_dir (str): Directory to save the SHAP plot.
    """
    print(f"Starting explanation for location (lat: {lat}, lon: {lon}) on {target_date}...")

    #  predicted_csv, model, feature_names, output_csv,
    
    # Load the predictions CSV
    predicted_csv = f"{output_dir}/test_data_predicted_latest_{target_date}.csv_snodas_mask.csv"
    try:
        print(f"Loading predictions from {predicted_csv}...")
        predictions_df = pd.read_csv(predicted_csv)
    except FileNotFoundError:
        print(f"Error: Predicted CSV file not found at {predicted_csv}.")
        return
    
    # Find the closest row to the specified latitude and longitude
    print(f"Finding closest row to (lat: {lat}, lon: {lon})...")

    # Calculate the squared distance
    predictions_df['distance'] = ((predictions_df['lat'] - lat) ** 2 + (predictions_df['lon'] - lon) ** 2)

    # Find the row with the minimum distance
    closest_row_index = predictions_df['distance'].idxmin()
    closest_row = predictions_df.loc[closest_row_index]

    # Print the index and the row details
    print(f"Closest row index: {closest_row_index}")
    print(f"Closest row details: {closest_row.to_dict()}")

    # Get the right feature names, minus the target column
    feature_names = [col for col in selected_columns if col != "swe_value"]

    # Extract the input features for SHAP
    input_data = closest_row[feature_names].to_frame().T
    # print("input data columns: ", input_data.columns)
    # print("feature names: ", feature_names)

    # Load model
    model = load_model(model_path)

    output_csv = f"{output_dir}/eai_et_model_{lat}_{lon}_{target_date}.csv_snodas_mask.csv"

    # Explain the prediction for this specific row
    explain_predictions(
        model=model,
        input_data=input_data,
        feature_names=feature_names,
        output_csv=output_csv,
        name = f"{lat}_{lon}_{target_date}",
        plot_path=f"{plot_dir}/eai_plot_{lat}_{lon}_{target_date}.png",
    )

    print(f"Explanation completed for location (lat: {lat}, lon: {lon}) on {target_date}.")


if __name__ == "__main__":
    #plot_feature_importance()  # no need, this step is already done in the model post processing step. 
    # interpret_prediction()
    # plot_model()

    explain_predictions_for_latlon(
        41.742627, -102.255249,
        target_date = "2025-01-15",
    )


