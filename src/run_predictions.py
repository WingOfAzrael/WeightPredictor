#!/usr/bin/env python3

import yaml
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
from weight_predictor import WeightPredictors

def load_config():
    """Load configuration file for model parameters."""

    # Get the directory of the currently running script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct the relative path to config.yaml
    config_path = os.path.join(script_dir, 'config.yaml')
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def analyze_data(original_data, predicted_weights, weight_data, model_name, model_index, truckSIDs_train, truckSIDs_test):
    """Analyze the model's performance and generate plots for correlations and weights."""
    

    front = [ np.round(np.average(i), 0) for i in np.array_split(np.array([i[0] for i in weight_data]), len(predicted_weights))]
    back = [ np.round(np.average(i), 0) for i in np.array_split(np.array([i[1] for i in weight_data]), len(predicted_weights))]
    total = [ np.round(np.average(i), 0) for i in np.array_split(np.array([i[2] for i in weight_data]), len(predicted_weights))]
    
    actuals = [front, back, total]

    # Create output directory if it doesn't exist
    output_dir = os.path.join(os.path.dirname(__file__), "../outputs")
    os.makedirs(output_dir, exist_ok=True)
    # Define the file path within the output directory
    file_path = os.path.join(output_dir, f"predictions_{model_name}.pdf")
    pdf_pages = PdfPages(file_path)

    # Create correlation matrices
    corr_df_train = pd.DataFrame(original_data)
    corr_df_test = pd.DataFrame(predicted_weights)
    weight_labels = ['PredictedWeightFront', 'PredictedWeightBack', 'PredictedWeightTotal']

    full_matrix = pd.concat([corr_df_train, corr_df_test], axis=1).corr()
    
    # Plot correlation matrices
    sns.set(font_scale=0.5)
    plt.figure(figsize=(10, 8))
    sns.heatmap(full_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix of Training and Predicted Weights')
    pdf_pages.savefig()
    plt.clf()

    # Histogram of Actual vs Predicted Weights
    types = ['MAPE front', 'MAPE back', 'MAPE total', '$R^2$ (training)', '$R^2$ (prediction)']
    weight_types = ['front', 'back', 'total']
    for ind in range(len(weight_data.transpose())):
            
            plt.tight_layout()
            fontsize = {'fontsize'  : 2}
            plt.rc('legend', **fontsize)
            plt.style.use('seaborn')


            #Here we make the scatterplots comparing dataset values and weight predictions
            figMOD = plt.figure(figsize=(3, 3))
            plt.plot(1, 1)
            bins = np.linspace(0, max(np.concatenate([actuals[ind], np.array([np.round(i[ind], 0) for i in predicted_weights])])), 200)
            plt.ylabel("Count")
            plt.xlabel("{} outputs".format(model_name))
            plt.title('Weights prediction comparison histogram')
            plt.xlim([min(np.concatenate([actuals[ind], np.array([np.round(i[ind], 0) for i in predicted_weights])])), max(np.concatenate([[np.round(i[ind], 0) for i in weight_data], np.array([np.round(i[ind], 0) for i in predicted_weights])]))])
        
            val_of_bins_x1, edges_of_bins_x1, patches_x1 = plt.hist(actuals[ind], bins, facecolor='blue', edgecolor='black', alpha=0.4, histtype='stepfilled', label= 'Actual {} weights'.format(types[ind]), density = False)
            plt.hist([np.round(i[ind], 0) for i in predicted_weights], edges_of_bins_x1, facecolor='red', edgecolor='black', alpha=0.4, histtype='stepfilled', label= 'Predicted {} weights'.format(types[ind]), density = False)
            plt.legend(bbox_to_anchor=(1.05, 1), loc = 'upper left', borderaxespad=0.)
            #plt.tight_layout()


            pdf_pages.savefig(figMOD, bbox_inches='tight')
            plt.close()

    for ind in range(len(weight_data.transpose())):
        
        plt.tight_layout()
        fontsize = {'fontsize'  : 2}
        plt.rc('legend', **fontsize)
        plt.style.use('seaborn')
        figPlot = plt.figure(figsize=(3, 3))
        plt.plot(1, 1)
        plt.ylabel("Weight (kg)")
        plt.xlabel("TruckSID")
        plt.title('Weights prediction scatterplot')
        plt.xlim([min(np.concatenate([truckSIDs_train, truckSIDs_test]))-100, max(np.concatenate([truckSIDs_train, truckSIDs_test]))+100])
        plt.ylim([min(np.concatenate([[i[ind] for i in weight_data], [np.round(i[ind], 0) for i in predicted_weights]]))-100, max(np.concatenate([[i[ind] for i in weight_data], [np.round(i[ind], 0) for i in predicted_weights]]))+100])
        
        if len(truckSIDs_test) != len([i[ind] for i in weight_data]):
            plt.scatter(x = truckSIDs_train, y = [i[ind] for i in weight_data], label= 'Actual {} weights'.format(weight_types[ind]), color = 'black', s=2)
        elif len(truckSIDs_test) == len([i[ind] for i in weight_data]):
            plt.scatter(x = truckSIDs_test, y = [i[ind] for i in weight_data], label= 'Actual {} weights'.format(weight_types[ind]), color = 'black', s=2)
        
        if len(truckSIDs_test) != len([np.round(i[ind], 0) for i in predicted_weights]):
            print(len(truckSIDs_test))
            print(len([np.round(i[ind], 0) for i in predicted_weights]))
            plt.scatter(x = truckSIDs_test, y = [np.round(i[ind], 0) for i in predicted_weights], label= 'Predicted {} weights'.format(weight_types[ind]), color = 'blue', s=2)
        elif len(truckSIDs_test) == len([np.round(i[ind], 0) for i in predicted_weights]):
            plt.scatter(x = truckSIDs_test, y = [np.round(i[ind], 0) for i in predicted_weights], label= 'Predicted {} weights'.format(weight_types[ind]), color = 'blue', s=2)
        
        plt.legend(bbox_to_anchor=(1.05, 1), loc='best', borderaxespad=0.)
        
        pdf_pages.savefig(figPlot, bbox_inches='tight')
        plt.close()

    pdf_pages.close()
    print(f'Analysis for {model_name} saved to PDF.')

def main():
    #Load configuration
    config = load_config()

    #Initialize the weight predictor class
    weight_predictor = WeightPredictors(config)

    #Select models
    models = list(weight_predictor.model_method())
    model_names = ['Logistic Regression', 'RandomForestRegressor', 'Gradient Boosting Regressor', 
                   'Ada Boost Regressor', 'Decision Tree Regressor', 'Support Vector Regression']


    for index, (model_name, model) in enumerate(models):
        predictions, n_scores, mean_scores = weight_predictor.fit_predict(model)
        
        n_scores = abs(n_scores)
        mean_scores = abs(mean_scores)
        print('Coefficient of determination: {} ({})'.format(np.mean(n_scores), np.std(n_scores)))
        print('Mean squred error           : {} ({})'.format(np.mean(mean_scores), np.std(mean_scores)))


        # Save predictions as CSV
        predictions_df = pd.DataFrame(
            predictions,
            columns=['PredictedWeightFront', 'PredictedWeightBack', 'PredictedWeightTotal']
        )

        # reate output directory if it doesn't exist
        output_dir = os.path.join(os.path.dirname(__file__), "../outputs")
        os.makedirs(output_dir, exist_ok=True)

        #Define the file path within the output directory
        file_path = os.path.join(output_dir, f"predictions_{model_name}.csv")


        predictions_df.to_csv(file_path, index=False)

        #Visualize predictions and scores
        analyze_data(weight_predictor.training_data1, predictions, weight_predictor.training_data2, model_name, index, weight_predictor.original_labels, weight_predictor.target_labels)

if __name__ == "__main__":
    main()
