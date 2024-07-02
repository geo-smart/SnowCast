
# SnowCast

SnowCast team SkiingPeople

## Overview

This document provides an overview of a comprehensive full-stack workflow designed for forecasting snow water equivalent (SWE). The workflow includes multiple nodes representing different stages and components involved in data processing, model creation, training, prediction, and deployment.

## Getting Started

### Prerequisites

- Python 3.8 or higher

### Installation

- Install Geoweaver:
-- Download and install Geoweaver from geoweaver.dev.

- Download the Workflow:
-- Download the ZIP file containing the workflow from the release page.
-- Extract the ZIP file to your desired location.
-- Loading and Running the Workflow

- Open Geoweaver:
-- Launch Geoweaver on your machine.

- Import the Workflow:
--Use the import button in Geoweaver to load the latest version of the workflow from the extracted directory.

- Run the Workflow:
-- After importing, you can view the workflow nodes and their configurations in Geoweaver.
-- Click the run button in Geoweaver to execute the workflow step-by-step as configured.

### Workflow Description

The workflow begins with data collection and integration from various sources. Nodes such as data_sentinel2, data_terrainFeatures, data_gee_modis_station_only, data_gee_sentinel1_station_only, and others handle the gathering and preprocessing of satellite data, terrain features, and meteorological data. These datasets are crucial for accurate model training and predictions.

Next, the workflow transitions to model creation and training. Nodes like model_creation_lstm, model_creation_ghostnet, model_creation_xgboost, and several others are used to set up different types of machine learning models. The model_train_validate node is then executed to train and validate these models using the integrated data. This step ensures that the models are well-trained and capable of making accurate predictions.

Finally, the workflow focuses on generating predictions and deploying the results. The model_predict node is used to produce predictions based on the trained models. Real-time data testing is performed using nodes such as amsr_testing_realtime and gridmet_testing. Visualization and deployment of results are managed by nodes like convert_results_to_images and deploy_images_to_website, allowing users to easily interpret and share the outcomes.

## Contributing

We welcome contributions to enhance and expand this workflow. To contribute, follow these steps:

- Fork the repository.
- Create a new branch (git checkout -b feature/your-feature).
- Commit your changes (git commit -am 'Add new feature').
- Push to the branch (git push origin feature/your-feature).
- Create a new Pull Request.

By participating in this project, you can help improve data processing, model training, and prediction accuracy. Your contributions will enable others to benefit from advanced data analysis and machine learning techniques.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
Thanks to all contributors and open-source projects that made this workflow possible. Your efforts in data collection, model development, and community support are greatly appreciated.
