"""
This script defines the ETHole class, which is used for training and evaluating an Extra Trees Regressor model for SWE prediction.

Attributes:
    ETHole (class): A class for training and using an Extra Trees Regressor model for SWE prediction.

Functions:
    custom_loss(y_true, y_pred): A custom loss function that penalizes errors for values greater than 10.
    get_model(): Returns the Extra Trees Regressor model with specified hyperparameters.
    create_sample_weights(y, scale_factor): Creates sample weights based on target values and a scaling factor.
    preprocessing(): Preprocesses the training data, including data cleaning and feature extraction.
    train(): Trains the Extra Trees Regressor model.
    post_processing(): Performs post-processing, including feature importance analysis and visualization.
"""

import pandas as pd
import joblib
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
from model_creation_rf import RandomForestHole
from snowcast_utils import work_dir, homedir, model_dir, plot_dir, month_to_season
from sklearn.preprocessing import StandardScaler
from sklearn.utils import compute_sample_weight
import seaborn as sns
import os
from datetime import datetime

working_dir = work_dir

class ETHole(RandomForestHole):

    #training_data_path = f'{working_dir}/final_merged_data_3yrs_cleaned.csv'
    #training_data_path = f'{working_dir}/final_merged_data_3yrs_cleaned_v3.csv'
    #training_data_path = f'{working_dir}/all_merged_training_cum_water_year_winter_month_only.csv' # snotel points
    # training_data_path = f'{working_dir}/final_merged_data_3yrs_cleaned_v3_time_series_cumulative_v1.csv'
    # training_data_path = f"{working_dir}/snotel_ghcnd_stations_4yrs_all_cols_log10.csv"
    # training_data_path = f"{working_dir}/all_points_final_merged_training.csv"
    # training_data_path = f"{working_dir}/all_points_final_merged_training_snodas_mask.csv"
    training_data_path = f"{working_dir}/all_points_final_merged_training_snodas_mask_resnet.csv"
    # training_data_path = f"{working_dir}/../snotel_ghcnd_stations_4yrs_all_cols_log10.csv"


  
    def custom_loss(y_true, y_pred):
        """
        A custom loss function that penalizes errors for values greater than 10.

        Args:
            y_true (numpy.ndarray): True target values.
            y_pred (numpy.ndarray): Predicted target values.

        Returns:
            numpy.ndarray: Custom loss values.
        """
        errors = np.abs(y_true - y_pred)
        
        return np.where(y_true > 10, 2 * errors, errors)

    def get_model(self):
        """
        Returns the Extra Trees Regressor model with specified hyperparameters.

        Returns:
            ExtraTreesRegressor: The Extra Trees Regressor model.
        """
#         return ExtraTreesRegressor(n_estimators=200, 
#                                    max_depth=None,
#                                    random_state=42, 
#                                    min_samples_split=2,
#                                    min_samples_leaf=1,
#                                    n_jobs=5
#                                   )
        return ExtraTreesRegressor(n_jobs=-1, random_state=123)

    def create_sample_weights(self, X, y, scale_factor, columns):
        """
        Creates sample weights based on target values and a scaling factor.

        Args:
            y (numpy.ndarray): Target values.
            scale_factor (float): Scaling factor for sample weights.

        Returns:
            numpy.ndarray: Sample weights.
        """
        #return np.where(X["fsca"] < 100, scale_factor, 1)
        return (y - np.min(y)) / (np.max(y) - np.min(y)) * scale_factor
        # Create a weight vector to assign weights to features - this is not a good idea
#         feature_weights = {'date': 0.1, 'SWE': 1.5, 'wind_speed': 1.5, 'precipitation_amount': 2.0}
#         default_weight = 1.0

#         # Create an array of sample weights based on feature_weights
#         sample_weights = np.array([feature_weights.get(feature, default_weight) for feature in columns])
        #return sample_weights

      
    def preprocessing(self, chosen_columns=None):
        """
        Preprocesses the training data, including data cleaning and feature extraction.
        """
        print("preparing training data from csv: ", self.training_data_path)
        data = pd.read_csv(self.training_data_path)
        print("data.shape = ", data.shape)
        for column in data.columns:
            print(column)
        
        data['date'] = pd.to_datetime(data['date'])
        #reference_date = pd.to_datetime('1900-01-01')
        #data['date'] = (data['date'] - reference_date).dt.days
        # just use julian day
        #data['date'] = data['date'].dt.strftime('%j').astype(int)
        # just use the season to reduce the bias on month or dates
        data['date'] = data['date'].dt.month.apply(month_to_season)
        
        data.replace('--', pd.NA, inplace=True)
        data.fillna(-1, inplace=True)
        data = data.replace(-999, -1)
        
        data = data[(data['swe_value'] != -1)]
        data = data[(data['swe_value'] != -999)]
        print("After dropping -1: ", data.shape)
        
        if chosen_columns == None:
#           data = data.drop('Unnamed: 0', axis=1)
          non_numeric_columns = data.select_dtypes(exclude=['number']).columns
          # Drop non-numeric columns
          data = data.drop(columns=non_numeric_columns)
          print("all non-numeric columns are dropped: ", non_numeric_columns)
          #data = data.drop('level_0', axis=1)
          data = data.drop(['date'], axis=1)
          data = data.drop(['lat'], axis=1)
          data = data.drop(['lon'], axis=1)
        else:
          data = data[chosen_columns]
        
        # these columns all come from SNOTEL report
        cumulative_columns = [col for col in data.columns if 'cumulative' in col.lower()]

        precip_history_columns = [col for col in data.columns if 'precipitation_amount' in col.lower()]

        snotel_columns_to_drop = [
            'station_name', 
            # 'swe_value', 
            'change_in_swe_inch', 
            'snow_depth', 
            'air_temperature_observed_f', 
            # 'precipitation_amount'
        ] + cumulative_columns + precip_history_columns

        # Drop columns only if they exist in the DataFrame
        data = data.drop(columns=[col for col in snotel_columns_to_drop if col in data.columns], axis=1)

        print("SNOTEL columns dropped successfully!")
        print(data.columns)  # Print the remaining columns for verification

        

        X = data.drop('swe_value', axis=1)
        print('required features after removing swe_value:', X.columns)

        print("assign fsca > 100. and SWE > 100 to -1")
        # Iterate over columns and check if 'fsca' or 'SWE' is in the column name
        for column in X.columns:
            if 'fsca' in column.lower() or 'swe' in column.lower():
                # Set values > 100 to -1
                X.loc[X[column] > 100, column] = -1

        y = data['swe_value']
        print("describe the statistics of training input: ", X.describe())
        print("describe the statistics of swe_value: ", y.describe())
        
        print("input features and order: ", X.columns)
        print("training data row number: ", len(X))
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Initialize the StandardScaler
        #scaler = StandardScaler()

        # Fit the scaler on the training data and transform both training and testing data
        #X_train_scaled = scaler.fit_transform(X_train)
        #X_test_scaled = scaler.transform(X_test)
        
        self.weights = self.create_sample_weights(X_train, y_train, scale_factor=10, columns=X.columns)

        self.train_x, self.train_y = X_train, y_train
        self.test_x, self.test_y = X_test, y_test
        #self.train_x, self.train_y = X_train_scaled, y_train
        #self.test_x, self.test_y = X_test_scaled, y_test
        self.feature_names = X_train.columns
        
    def train(self):
        """
        Trains the Extra Trees Regressor model.
        """
        # Fit the classifier
        self.classifier.fit(self.train_x, self.train_y)

        # don't use the sample weights which are more appropriate for classification problem
        # Make predictions
        # predictions = self.classifier.predict(self.train_x)

        # # Calculate absolute errors
        # errors = np.abs(self.train_y - predictions)

        # # Assign weights based on errors (higher errors get higher weights)
        # weights = compute_sample_weight('balanced', errors)
        # self.classifier.fit(self.train_x, self.train_y, sample_weight=weights)

    def evaluate(self):
        print("Starting evaluation...")
        
        # Get predictions and true values
        print("Generating predictions...")
        y_pred = self.test()
        y_test = self.test_y

        # Calculate evaluation metrics
        print("Calculating metrics...")
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = np.mean(np.abs(y_test - y_pred))

        print(f"Metrics calculated:\n - R²: {r2:.4f}\n - RMSE: {rmse:.4f}\n - MAE: {mae:.4f}")

        # Scatter plot for predicted vs actual values
        print("Creating scatter plot...")
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=y_test, y=y_pred, color="blue", alpha=0.6, s=50)
        plt.plot(
            [min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', lw=2,
            label="1:1 line"
        )
        plt.title(f"Predicted vs Actual Values (R² = {r2:.2f})", fontsize=16)
        plt.xlabel("Actual SWE", fontsize=14)
        plt.ylabel("Predicted SWE", fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(fontsize=12)
        plt.tight_layout()

        # Save plot
        plot_path = f"{plot_dir}/predict_actual_scatter.png"
        print(f"Ensuring plot directory exists at: {plot_dir}")
        os.makedirs(plot_dir, exist_ok=True)

        print(f"Saving scatter plot to: {plot_path}")
        plt.savefig(plot_path, dpi=300)
        print("Scatter plot saved successfully.")

        # Display metrics in console
        print("\nEvaluation complete.")
        # print(f"Final Metrics:\n - R²: {r2:.4f}\n - RMSE: {rmse:.4f}\n - MAE: {mae:.4f}")

    def post_processing(self, chosen_columns=None,):
        """
        Performs post-processing, including feature importance analysis and visualization.

        Parameters:
        - chosen_columns (list, optional): Columns selected for the model. Defaults to None.
        """
        self.load_model()

        # Ensure feature names are initialized
        if not hasattr(self, 'feature_names') or self.feature_names is None:
            self.feature_names = ['SWE', 'relative_humidity_rmin', 'potential_evapotranspiration',
                'air_temperature_tmmx', 'relative_humidity_rmax',
                'mean_vapor_pressure_deficit', 'air_temperature_tmmn', 'wind_speed',
                'Elevation', 'Aspect', 'Curvature', 'Northness', 'Eastness', 'fsca',
                'Slope', 'SWE_1', 'air_temperature_tmmn_1',
                'potential_evapotranspiration_1', 'mean_vapor_pressure_deficit_1',
                'relative_humidity_rmax_1', 'relative_humidity_rmin_1',
                'air_temperature_tmmx_1', 'wind_speed_1', 'fsca_1', 'SWE_2',
                'air_temperature_tmmn_2', 'potential_evapotranspiration_2',
                'mean_vapor_pressure_deficit_2', 'relative_humidity_rmax_2',
                'relative_humidity_rmin_2', 'air_temperature_tmmx_2', 'wind_speed_2',
                'fsca_2', 'SWE_3', 'air_temperature_tmmn_3',
                'potential_evapotranspiration_3', 'mean_vapor_pressure_deficit_3',
                'relative_humidity_rmax_3', 'relative_humidity_rmin_3',
                'air_temperature_tmmx_3', 'wind_speed_3', 'fsca_3', 'SWE_4',
                'air_temperature_tmmn_4', 'potential_evapotranspiration_4',
                'mean_vapor_pressure_deficit_4', 'relative_humidity_rmax_4',
                'relative_humidity_rmin_4', 'air_temperature_tmmx_4', 'wind_speed_4',
                'fsca_4', 'SWE_5', 'air_temperature_tmmn_5',
                'potential_evapotranspiration_5', 'mean_vapor_pressure_deficit_5',
                'relative_humidity_rmax_5', 'relative_humidity_rmin_5',
                'air_temperature_tmmx_5', 'wind_speed_5', 'fsca_5', 'SWE_6',
                'air_temperature_tmmn_6', 'potential_evapotranspiration_6',
                'mean_vapor_pressure_deficit_6', 'relative_humidity_rmax_6',
                'relative_humidity_rmin_6', 'air_temperature_tmmx_6', 'wind_speed_6',
                'fsca_6', 'SWE_7', 'air_temperature_tmmn_7',
                'potential_evapotranspiration_7', 'mean_vapor_pressure_deficit_7',
                'relative_humidity_rmax_7', 'relative_humidity_rmin_7',
                'air_temperature_tmmx_7', 'wind_speed_7', 'fsca_7', 'water_year']

        # Extract feature importances and sort them
        feature_importances = self.classifier.feature_importances_
        feature_names = self.feature_names
        sorted_indices = np.argsort(feature_importances)[::-1]
        sorted_importances = feature_importances[sorted_indices]
        sorted_feature_names = np.array(feature_names)[sorted_indices]

        # Set figure size dynamically based on the number of features
        num_features = len(feature_names)
        fig_height = max(8, num_features * 0.3)  # Adjust height based on feature count

        plt.figure(figsize=(12, fig_height))
        plt.barh(range(num_features), sorted_importances, color="skyblue")
        
        # Set feature names as y-tick labels
        plt.yticks(range(num_features), sorted_feature_names, fontsize=10)
        plt.gca().invert_yaxis()  # Reverse the order of features for better readability
        plt.xlabel('Feature Importance', fontsize=14)
        plt.ylabel('Feature', fontsize=14)
        plt.title('Feature Importance Plot (ET model)', fontsize=16)
        plt.tight_layout()  # Ensure no overlap between labels and plot

        # Generate a timestamp to append to the filename
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        
        # Determine the output file name with timestamp
        if chosen_columns is None:
            feature_png = f'{plot_dir}/et-model-feature-importance-{timestamp}.png'
        else:
            feature_png = f'{plot_dir}/et-model-feature-importance-{len(chosen_columns)}-{timestamp}.png'

        # Save the plot
        plt.savefig(feature_png)
        plt.close()  # Close the plot to free memory
        print(f"Feature importance plot saved to {feature_png}")


# Instantiate ETHole class and perform tasks

# all_used_columns = ['station_elevation', 'elevation', 'aspect', 'curvature', 'slope',
# 'eastness', 'northness', 'etr', 'pr', 'rmax', 'rmin', 'tmmn', 'tmmx',
# 'vpd', 'vs', 'lc_code',  'fSCA',  'cumulative_etr',
# 'cumulative_rmax', 'cumulative_rmin', 'cumulative_tmmn',
# 'cumulative_tmmx', 'cumulative_vpd', 'cumulative_vs', 'cumulative_pr', 'swe_value']

#all_used_columns = ['cumulative_pr','station_elevation', 'cumulative_tmmn', 'cumulative_tmmx', 'northness', 'cumulative_vs', 'cumulative_rmax', 'cumulative_etr','aspect','cumulative_rmin', 'elevation', 'cumulative_vpd',  'swe_value']
# selected_columns = ["lat","lon","elevation","slope","curvature","aspect","eastness","northness","cumulative_SWE","cumulative_Flag","cumulative_air_temperature_tmmn","cumulative_potential_evapotranspiration","cumulative_mean_vapor_pressure_deficit","cumulative_relative_humidity_rmax","cumulative_relative_humidity_rmin","cumulative_precipitation_amount","cumulative_air_temperature_tmmx","cumulative_wind_speed", "swe_value"]

# all current variables without time series
# selected_columns = ['SWE', 'Flag', 'air_temperature_tmmn', 'potential_evapotranspiration',
# 'mean_vapor_pressure_deficit', 'relative_humidity_rmax',
# 'relative_humidity_rmin', 'precipitation_amount',
# 'air_temperature_tmmx', 'wind_speed', 'elevation', 'slope', 'curvature',
# 'aspect', 'eastness', 'northness', 'cumulative_SWE',
# 'cumulative_Flag', 'cumulative_air_temperature_tmmn',
# 'cumulative_potential_evapotranspiration',
# 'cumulative_mean_vapor_pressure_deficit',
# 'cumulative_relative_humidity_rmax',
# 'cumulative_relative_humidity_rmin', 'cumulative_precipitation_amount',
# 'cumulative_air_temperature_tmmx', 'cumulative_wind_speed', 'swe_value']


selected_columns = ['SWE', 'swe_value', 'relative_humidity_rmin',
'potential_evapotranspiration', 'air_temperature_tmmx',
'relative_humidity_rmax', 'mean_vapor_pressure_deficit',
'air_temperature_tmmn', 'wind_speed', 'Elevation', 'Aspect',
'Curvature', 'Northness', 'Eastness', 'fsca', 'Slope', 'SWE_1',
'air_temperature_tmmn_1', 'potential_evapotranspiration_1',
'mean_vapor_pressure_deficit_1', 'relative_humidity_rmax_1',
'relative_humidity_rmin_1', 'air_temperature_tmmx_1', 'wind_speed_1',
'fsca_1', 'SWE_2', 'air_temperature_tmmn_2',
'potential_evapotranspiration_2', 'mean_vapor_pressure_deficit_2',
'relative_humidity_rmax_2', 'relative_humidity_rmin_2',
'air_temperature_tmmx_2', 'wind_speed_2', 'fsca_2', 'SWE_3',
'air_temperature_tmmn_3', 'potential_evapotranspiration_3',
'mean_vapor_pressure_deficit_3', 'relative_humidity_rmax_3',
'relative_humidity_rmin_3', 'air_temperature_tmmx_3', 'wind_speed_3',
'fsca_3', 'SWE_4', 'air_temperature_tmmn_4',
'potential_evapotranspiration_4', 'mean_vapor_pressure_deficit_4',
'relative_humidity_rmax_4', 'relative_humidity_rmin_4',
'air_temperature_tmmx_4', 'wind_speed_4', 'fsca_4', 'SWE_5',
'air_temperature_tmmn_5', 'potential_evapotranspiration_5',
'mean_vapor_pressure_deficit_5', 'relative_humidity_rmax_5',
'relative_humidity_rmin_5', 'air_temperature_tmmx_5', 'wind_speed_5',
'fsca_5', 'SWE_6', 'air_temperature_tmmn_6',
'potential_evapotranspiration_6', 'mean_vapor_pressure_deficit_6',
'relative_humidity_rmax_6', 'relative_humidity_rmin_6',
'air_temperature_tmmx_6', 'wind_speed_6', 'fsca_6', 'SWE_7',
'air_temperature_tmmn_7', 'potential_evapotranspiration_7',
'mean_vapor_pressure_deficit_7', 'relative_humidity_rmax_7',
'relative_humidity_rmin_7', 'air_temperature_tmmx_7', 'wind_speed_7',
'fsca_7', 'water_year', 'snodas_mask']

# ['cumulative_relative_humidity_rmin', 'cumulative_air_temperature_tmmx', 'cumulative_air_temperature_tmmn', 'cumulative_relative_humidity_rmax', 'cumulative_potential_evapotranspiration', 'cumulative_wind_speed'] 

if __name__ == "__main__":
  hole = ETHole()
#   hole.preprocessing(chosen_columns = selected_columns)
  hole.preprocessing()

  hole.train()
  hole.test()
  hole.evaluate()
  hole.save()
#   hole.post_processing(chosen_columns = selected_columns)
  hole.post_processing()


