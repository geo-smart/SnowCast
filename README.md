![Workflow Badge](https://img.shields.io/badge/Workflow-snowcast_wormhole-blue.svg)

# Workflow Name: snowcast_wormhole

## Description
The complete workflow for snowcast workflow creation, train, test, validation, deploy, predict.

## Processes
data_sentinel2, model_creation_lstm, model_creation_ghostnet, data_integration, model_train_validate, model_predict, data_terrainFeatures, data_gee_modis_station_only, data_gee_sentinel1_station_only, data_associate_station_grid_cell, data_gee_modis_real_time, data_gee_sentinel1_real_time, base_hole, data_gee_gridmet_station_only, data_gee_gridmet_real_time, model_creation_xgboost, testing_data_integration, snowcast_utils, model_create_kehan, data_snotel_real_time, all_dependencies, data_WUS_UCLA_SR, data_nsidc_4km_swe, model_creation_et, data_snotel_station_only, model_creation_rf, model_creation_pycaret, model_creation_autokeras, model_creation_autopytorch, western_us_dem.py, download_srtm_1arcsec (caution!), gridmet_testing, create_output_tif_template, resample_dem, convert_results_to_images, deploy_images_to_website, training_feature_selection, amsr_testing_realtime, perform_download.sh, merge_custom_traning_range, training_data_range, amsr_features, amsr_swe_data_download, install_dependencies, model_evaluation, interpret_model_results, train_test_pattern_compare, correct_slope, convert_to_time_series, data_merge_hackweek, data_gee_smap_station_only, data_merge_hackweek_testing, training_sanity_check, fSCA_training, fsCA_testing, fsca_py, fSCA_training_extract_data, data_ghcnd_station_only, mod_water_mask, prepare_water_mask_template, download_modis_09, data_mod09_extract_csvs, duplicated_feature_selection, train_self_attention_xgb_slurm, train_xgb_slurm, train_novel_transformerxgb, predict_transformerxgb, bttf_swe_predict, bttf_train, xgb_train, xgb_predict, snodas_test, transformer_snodas_train, snodas_tabnet, snodas_testing_realtime, snodas_tabnet_slurm, add_snodas_mask_column, snodas_fttransformer, snodas_dnn_new, base_nn_hole, clip_basins_for_eval, snodas_resnet, clip_basins_swe_sh, clipped_geotif_to_netcdf

### Process Descriptions
|-------------------------------------------------------------|
| Process Name                     | Description              |
|----------------------------------|--------------------------|
| data_sentinel2                   | No description available |
| model_creation_lstm              | python                   |
| model_creation_ghostnet          | python                   |
| data_integration                 | No description available |
| model_train_validate             | No description available |
| model_predict                    | No description available |
| data_terrainFeatures             | No description available |
| data_gee_modis_station_only      | No description available |
| data_gee_sentinel1_station_only  | No description available |
| data_associate_station_grid_cell | No description available |
| data_gee_modis_real_time         | No description available |
| data_gee_sentinel1_real_time     | No description available |
| base_hole                        | No description available |
| data_gee_gridmet_station_only    | No description available |
| data_gee_gridmet_real_time       | No description available |
| model_creation_xgboost           | No description available |
| testing_data_integration         | No description available |
| snowcast_utils                   | No description available |
| model_create_kehan               | No description available |
| data_snotel_real_time            | No description available |
| all_dependencies                 | No description available |
| data_WUS_UCLA_SR                 | No description available |
| data_nsidc_4km_swe               | No description available |
| model_creation_et                | No description available |
| data_snotel_station_only         | No description available |
| model_creation_rf                | No description available |
| model_creation_pycaret           | No description available |
| model_creation_autokeras         | No description available |
| model_creation_autopytorch       | No description available |
| western_us_dem.py                | No description available |
| download_srtm_1arcsec (caution!) | No description available |
| gridmet_testing                  | No description available |
| create_output_tif_template       | No description available |
| resample_dem                     | No description available |
| convert_results_to_images        | No description available |
| deploy_images_to_website         | No description available |
| training_feature_selection       | No description available |
| amsr_testing_realtime            | No description available |
| perform_download.sh              | No description available |
| merge_custom_traning_range       | No description available |
| training_data_range              | No description available |
| amsr_features                    | No description available |
| amsr_swe_data_download           | No description available |
| install_dependencies             | No description available |
| model_evaluation                 | No description available |
| interpret_model_results          | No description available |
| train_test_pattern_compare       | No description available |
| correct_slope                    | No description available |
| convert_to_time_series           | No description available |
| data_merge_hackweek              | No description available |
| data_gee_smap_station_only       | No description available |
| data_merge_hackweek_testing      | No description available |
| training_sanity_check            | No description available |
| fSCA_training                    | No description available |
| fsCA_testing                     | No description available |
| fsca_py                          | No description available |
| fSCA_training_extract_data       | No description available |
| data_ghcnd_station_only          | No description available |
| mod_water_mask                   | No description available |
| prepare_water_mask_template      | No description available |
| download_modis_09                | No description available |
| data_mod09_extract_csvs          | No description available |
| duplicated_feature_selection     | python                   |
| train_self_attention_xgb_slurm   | No description available |
| train_xgb_slurm                  | No description available |
| train_novel_transformerxgb       | No description available |
| predict_transformerxgb           | No description available |
| bttf_swe_predict                 | No description available |
| bttf_train                       | No description available |
| xgb_train                        | No description available |
| xgb_predict                      | No description available |
| snodas_test                      | No description available |
| transformer_snodas_train         | No description available |
| snodas_tabnet                    | No description available |
| snodas_testing_realtime          | No description available |
| snodas_tabnet_slurm              | No description available |
| add_snodas_mask_column           | No description available |
| snodas_fttransformer             | No description available |
| snodas_dnn_new                   | No description available |
| base_nn_hole                     | No description available |
| clip_basins_for_eval             | No description available |
| snodas_resnet                    | No description available |
| clip_basins_swe_sh               | No description available |
| clipped_geotif_to_netcdf         | No description available |
|-------------------------------------------------------------|


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
