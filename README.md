# Truck Axle Weight Prediction

This project predicts truck weights (front, rear, and total) using machine learning models. Using historical weight data and relevant vehicle features, the system predicts truck weights based on new data inputs.
Project Structure

```
├── data
│   ├── testing_acu.csv                   # Testing dataset for prediction
│   └── training_acu.csv                   # Training dataset for model training
├── outputs                                # Directory for output files (predictions and visualizations)
│   ├── predictions_GradientBoostingRegressor.csv
│   ├── predictions_GradientBoostingRegressor.pdf
│   ├── predictions_RandomForestRegressor.csv
│   ├── predictions_RandomForestRegressor.pdf
│   ├── predictions_SVR.csv
│   └── predictions_SVR.pdf
├── requirements.txt                       # Project dependencies
├── src
│   ├── config.yaml                        # Configuration file for model settings and data paths
│   ├── run_predictions.py                 # Script to run predictions and generate outputs
│   └── weight_predictor.py                # Class definitions for data preprocessing, model training, and predictions

```
## Goal

The goal of this project is to predict the front, rear, and total weights of trucks using machine-learning models that leverage key vehicle features.
Tasks

    Data Analysis: Analyze and explore the provided training data to understand patterns in truck weight based on features.
    Model Training: Develop and train machine learning models to predict truck weights on a testing dataset.
    Generate Predictions: Produce a CSV file with predicted weights for each truck in the testing dataset.
    Presentation Preparation: Summarize findings in a presentation to report insights from data analysis and model results.

## Setup Instructions
1. Prerequisites

Ensure you have Python 3.8+ installed. Install dependencies from requirements.txt:

bash command 

```
pip install -r requirements.txt
```
2. Project Configuration

Set up the paths and model parameters in the config.yaml file:

yaml

paths:
  train_data: "../data/training_acu.csv"    # Path to training data
  test_data: "../data/testing_acu.csv"      # Path to testing data

model:
  choices:                                  # Models to be used (extendable in weight_predictor.py)
    - "RandomForestRegressor"
    - "GradientBoostingRegressor"
    - "SVR"

training:
  test_size: 0.2                            # Size of test data split
  random_state: 42                          # Seed for reproducibility

3. Running the Project

Run the prediction script to train models and generate predictions:


bash command

```
python src/run_predictions.py
```

This script:

    Loads data as specified in config.yaml
    Trains specified models on the training data
    Generates predictions for the testing data
    Saves predictions and visualizations in the outputs folder

## Details of Key Files

    src/weight_predictor.py: This file contains the WeightPredictors class, which handles data preprocessing, column alignment between training and testing data, and the model training and prediction process. The models supported include RandomForestRegressor, GradientBoostingRegressor, SVR, and more, as listed in config.yaml.

    src/run_predictions.py: The main script that orchestrates the prediction pipeline. It loads configuration settings, initializes the WeightPredictors class, trains models, and saves predictions as CSV and PDF files in the outputs folder. The script also calls analyze_data to generate correlation matrices and scatter plots to visually compare actual vs. predicted weights.

    data folder: Contains training_acu.csv (historical truck weight data) for model training and testing_acu.csv for predictions.

    config.yaml: Provides model and data configurations, including the list of models to be used, file paths, and training parameters.

    requirements.txt: Lists all dependencies required to run the project, including libraries for data manipulation, machine learning, and plotting.

## Model Evaluation

For each model specified in config.yaml, the script evaluates and records:

    R^2 Score: Indicates model accuracy in capturing weight variance.
    Mean Squared Error (MSE): Measures prediction error.

### Outputs

    Predictions: CSV files for each model with columns: TruckSID, PredictedWeightFront, PredictedWeightBack, and PredictedWeightTotal.
    Visualizations: PDF files with plots of actual vs. predicted weights and correlation matrices for detailed analysis.

## Extending the Project

    Add More Models: Extend the model_choices in config.yaml and model_method in weight_predictor.py to include additional models, such as AdaBoostRegressor or DecisionTreeRegressor.
    Adjust Data Processing: Modify clean_data and align_columns in weight_predictor.py for different preprocessing requirements or new data columns.
    SHAP Explanations: explain_with_shap (in weight_predictor.py) provides SHAP visualizations for model interpretability, showing feature contributions.
    Fuel consumption predictions: use the weight predictions to predict fuel consumption for logistical optimisation.


## Acknowledgments

This project was developed as part of a data science assessment. The goal was to predict truck weights and report the findings for further analysis. 


License

This project is licensed under the MIT License.
