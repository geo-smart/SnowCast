![Workflow Badge](https://img.shields.io/badge/Workflow-snowcast_wormhole-blue.svg)

# Workflow Name: snowcast_wormhole

## Description
The complete workflow for snowcast workflow creation, train, test, validation, deploy, predict.

## Processes
data_sentinel2, model_creation_lstm, model_creation_ghostnet, data_integration, model_train_validate, model_predict, data_terrainFeatures, data_gee_modis_station_only, data_gee_sentinel1_station_only, data_associate_station_grid_cell, data_gee_modis_real_time, data_gee_sentinel1_real_time, base_hole, data_gee_gridmet_station_only, data_gee_gridmet_real_time, model_creation_xgboost, testing_data_integration, snowcast_utils, model_create_kehan, data_snotel_real_time, all_dependencies, data_WUS_UCLA_SR, data_nsidc_4km_swe, model_creation_et, data_snotel_station_only, model_creation_rf, model_creation_pycaret, model_creation_autokeras, model_creation_autopytorch, western_us_dem.py, download_srtm_1arcsec (caution!), gridmet_testing, create_output_tif_template, resample_dem, convert_results_to_images, deploy_images_to_website, training_feature_selection, amsr_testing_realtime, perform_download.sh, merge_custom_traning_range, training_data_range, amsr_features, amsr_swe_data_download, install_dependencies, model_evaluation, interpret_model_results, train_test_pattern_compare, correct_slope, convert_to_time_series, data_merge_hackweek, data_gee_smap_station_only, data_merge_hackweek_testing, training_sanity_check, fSCA_training, fsCA_testing, fsca_py, fSCA_training_extract_data, data_ghcnd_station_only, mod_water_mask, prepare_water_mask_template, download_modis_09, data_mod09_extract_csvs, duplicated_feature_selection, train_self_attention_xgb_slurm, train_xgb_slurm, train_novel_transformerxgb, predict_transformerxgb, bttf_swe_predict, bttf_train, xgb_train, xgb_predict, snodas_test, transformer_snodas_train, snodas_tabnet, snodas_testing_realtime, snodas_tabnet_slurm, add_snodas_mask_column, snodas_fttransformer, snodas_dnn_new, base_nn_hole, clip_basins_for_eval, snodas_resnet

### Process Descriptions
data_sentinel2: null
model_creation_lstm: python
model_creation_ghostnet: python
data_integration: null
model_train_validate: null
model_predict: null
data_terrainFeatures: null
data_gee_modis_station_only: null
data_gee_sentinel1_station_only: null
data_associate_station_grid_cell: null
data_gee_modis_real_time: null
data_gee_sentinel1_real_time: null
base_hole: null
data_gee_gridmet_station_only: null
data_gee_gridmet_real_time: null
model_creation_xgboost: null
testing_data_integration: null
snowcast_utils: null
model_create_kehan: null
data_snotel_real_time: null
all_dependencies: null
data_WUS_UCLA_SR: null
data_nsidc_4km_swe: null
model_creation_et: null
data_snotel_station_only: null
model_creation_rf: null
model_creation_pycaret: null
model_creation_autokeras: null
model_creation_autopytorch: null
western_us_dem.py: null
download_srtm_1arcsec (caution!): null
gridmet_testing: null
create_output_tif_template: null
resample_dem: null
convert_results_to_images: null
deploy_images_to_website: null
training_feature_selection: null
amsr_testing_realtime: null
perform_download.sh: null
merge_custom_traning_range: null
training_data_range: null
amsr_features: null
amsr_swe_data_download: null
install_dependencies: null
model_evaluation: null
interpret_model_results: null
train_test_pattern_compare: null
correct_slope: null
convert_to_time_series: null
data_merge_hackweek: null
data_gee_smap_station_only: null
data_merge_hackweek_testing: null
training_sanity_check: null
fSCA_training: null
fsCA_testing: null
fsca_py: null
fSCA_training_extract_data: null
data_ghcnd_station_only: null
mod_water_mask: null
prepare_water_mask_template: null
download_modis_09: null
data_mod09_extract_csvs: null
duplicated_feature_selection: python
train_self_attention_xgb_slurm: null
train_xgb_slurm: null
train_novel_transformerxgb: null
predict_transformerxgb: null
bttf_swe_predict: null
bttf_train: null
xgb_train: null
xgb_predict: null
snodas_test: null
transformer_snodas_train: null
snodas_tabnet: null
snodas_testing_realtime: null
snodas_tabnet_slurm: null
add_snodas_mask_column: null
snodas_fttransformer: null
snodas_dnn_new: null
base_nn_hole: null
clip_basins_for_eval: python
snodas_resnet: null


## Steps to use the workflow

This section provides detailed instructions on how to use the workflow. Follow these steps to set up and execute the workflow using Geoweaver.

### Step-by-Step Instructions

### Step 1: Download the zip file
### Step 2: Import the Workflow into Geoweaver
Open Geoweaver running on your local machine. [video guidance](https://youtu.be/jUd1dzi18EQ)
1. Click on "Weaver" in the top navigation bar.
2. A workspace to add a workflow opens up. Select the "Import" icon in the top navigation bar.
3. Choose the downloaded zip file4. Click on "Start" to upload the file. If the file is valid, a prompt will ask for your permission to upload. Click "OK".
5. Once the file is uploaded, Geoweaver will create a new workflow.

### Step 3: Execute the Workflow
1. Click on the execute icon in the top navigation bar to start the workflow execution process.[video guidance](https://youtu.be/PJcMNR00QoE)
2. A wizard will open where you need to select the [video guidance](https://youtu.be/KYiEHI0rn_o) and environment [video guidance](https://www.youtube.com/watch?v=H66AVoBBaHs).
3. Click on "Execute" to initiate the workflow. Enter the required password when prompted and click "Confirm" to start executing the workflow.

### Step 4: Monitor Execution and View Results
1. The workflow execution will begin.
2. Note: Green indicates the process is successful, Yellow indicates the process is running, and Red indicates the process has failed.
3. Once the execution is complete, the results will be available immediately.

By following these steps, you will be able to set up and execute the snow cover mapping workflow using Geoweaver.

