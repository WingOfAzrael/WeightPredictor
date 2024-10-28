import yaml
import sklearn 
import xgboost 
import os
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, cross_val_predict, RepeatedKFold
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

class WeightPredictors:
    def __init__(self, config):
        #Resolve the paths from the config file
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.train_data_path = os.path.abspath(os.path.join(self.script_dir, config['paths']['train_data']))
        self.test_data_path = os.path.abspath(os.path.join(self.script_dir, config['paths']['test_data']))

        self.model_choices = config['model']['choices']  
        self.test_size = config['training']['test_size']
        self.random_state = config['training']['random_state']

        self.combined_data = pd.read_csv(self.train_data_path, delimiter=";")
        self.combined_prediction_data = pd.read_csv(self.test_data_path, delimiter=";")
        self.clean_data()

        self.original_frame, self.prediction_frame, self.original_targets = self.align_columns(self.combined_data, self.combined_prediction_data)


        #Process the data into arrays
        self.training_data1, self.training_data2, self.testing_data, self.original_labels, self.target_labels = self.process_data()

    def clean_data(self):
        #Retain only matching columns
        common_columns = self.combined_data.columns.intersection(self.combined_prediction_data.columns)
        self.data_features = self.combined_data[common_columns].drop(
            columns=['ActualWeightFront', 'ActualWeightBack', 'ActualWeightTotal'], errors='ignore')
        self.prediction_features = self.combined_prediction_data[common_columns].drop(
            columns=['ActualWeightFront', 'ActualWeightBack', 'ActualWeightTotal'], errors='ignore')

        self.target_weights = self.combined_data[['ActualWeightFront', 'ActualWeightBack', 'ActualWeightTotal']].dropna()

    
    def align_columns(self, combined_data, combined_prediction_data):
        #This function aligns columns between training and testing datasets, ensuring they are consistent. 
        prediction_frame = pd.DataFrame()
        
        names = ['ActualWeightFront', 'ActualWeightBack', 'ActualWeightTotal']

        other_names  = [col for col in combined_data.columns if col not in names]

        original_frame = combined_data[other_names]

        original_targets = combined_data[names]

        prediction_frame = combined_prediction_data
        
        return original_frame, prediction_frame, original_targets

    def process_data(self):
        # This function processes the data by encoding and creating arrays for training, prediction, and target data. 
        # Convert DataFrames to arrays
        final_original_frame = self.original_frame
        final_prediction_frame = self.prediction_frame
        final_target_frame = self.original_targets

        #Extract labels and arrays for original, prediction, and target frames
        self.original_labels = list(final_original_frame.index.values)
        original_array = list(final_original_frame.to_numpy())

        self.prediction_labels = list(final_prediction_frame.index.values)
        prediction_array = list(final_prediction_frame.to_numpy())

        self.target_labels = list(final_target_frame.index.values)
        target_array = list(final_target_frame.to_numpy())

        #Ordinal encoding for categorical features
        training_encoder = OrdinalEncoder()
        target_encoder = OrdinalEncoder()
        prediction_encoder = OrdinalEncoder()

        #Fit and transform training, target, and prediction arrays
        train_ord_enc = training_encoder.fit(np.array(original_array).astype(str))
        targ_enc = target_encoder.fit(np.array(target_array))
        pred_ord_enc = prediction_encoder.fit(np.array(prediction_array).astype(str))

        #Apply encoding to the arrays
        original_combined = train_ord_enc.transform(np.array(original_array).astype(str))
        target_combined = np.array(target_array)  # Target does not need encoding in this case
        prediction_combined = pred_ord_enc.transform(np.array(prediction_array).astype(str))



        #Store the transformed arrays for later use
        self.original_combined = original_combined
        self.target_combined = target_combined
        self.prediction_combined = prediction_combined
        self.training_data1 = original_combined
        self.training_data2 = target_combined
        self.testing_data = prediction_combined


        return self.training_data1, self.training_data2, self.testing_data, self.original_labels, self.prediction_labels

    def give_data(self):
        """ Returns training data, target data, and test data for model training. """
        return self.training_data1, self.training_data2, self.testing_data,  self.original_labels, self.prediction_labels

    def model_method(self):
        #Define available models
        model_dict = {
            'LogisticRegression': LogisticRegression(solver='liblinear', random_state=82, max_iter=150),
            'RandomForestRegressor': RandomForestRegressor(random_state=82, n_estimators=400, max_depth=8),
            'GradientBoostingRegressor': GradientBoostingRegressor(n_estimators=90),
            'AdaBoostRegressor': AdaBoostRegressor(n_estimators=90),
            'DecisionTreeRegressor': DecisionTreeRegressor(max_depth=10, splitter='random'),
            'SVR': SVR(C=1.0, epsilon=0.2)
        }


        #Fetch models based on the choices provided in the config
        selected_models = [MultiOutputRegressor(model_dict[model]) for model in self.model_choices]
        model_names = self.model_choices
        return zip(model_names, selected_models)

    def fit_predict1(self, model, prediction_choice='test'):

        """ 
         
          
            """
        results = {}
        model_name = model.estimator.__class__.__name__
        print(f"Training and predicting with model: {model_name}")
        #Call general_fit to train the model 
        model.fit(self.training_data1, self.target_weights)
        #Select the dataset for prediction
        if prediction_choice == 'train':
            data = self.training_data1
            targets = self.target_weights
        else:
            data = self.testing_data
            #targets = np.array_split(self.target_weights.values, len(self.prediction_features))
        #Call general_predict logic
        predictions = model.predict(data)
        results[model_name] = predictions
        
        #Calculate performance metrics 
        cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1)
        n_scores = abs(cross_val_score(model, self.training_data1, self.target_weights, scoring='r2', cv=cv, n_jobs=-1))
        print(f'R^2 score for {model_name}: {np.mean(n_scores)}')
        #Optionally, integrate SHAP for explanations, needs work
        #self.explain_with_shap(model)

        return results


    def fit_predict(self, model, prediction_choice='test'):
        #Train the model on the training data
        model.fit(self.training_data1, self.target_weights)
        
        #Select data for prediction
        if prediction_choice == 'train':
            data = self.training_data1
            targets = self.training_data2  # original weights
        else:
            data = self.testing_data
            #Prepare reshaped prediction input frame
            self.data_choice = self.testing_data
            front = np.array([ np.round(np.average(i), 0) for i in np.array_split(np.array([i[0] for i in self.training_data2]), len(self.testing_data))])
            back = np.array([ np.round(np.average(i), 0) for i in np.array_split(np.array([i[1] for i in self.training_data2]), len(self.testing_data))])
            total = np.array([ np.round(np.average(i), 0) for i in np.array_split(np.array([i[2] for i in self.training_data2]), len(self.testing_data))])
            self.prediction_input = np.array([front, back, total]).transpose()

        #predictions = model.predict(data)
        #Evaluate model performancerain
        cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1)
        n_scores = cross_val_score(model, self.training_data1, self.training_data2, scoring='r2', cv=cv, n_jobs=-1)
        n_scores = abs(n_scores)

        
        mean_scores = cross_val_score(model, self.data_choice, self.prediction_input, scoring= 'r2' , cv=cv, n_jobs=-1)
        mean_scores = abs(mean_scores)
        
        #These are resized target variables for predictions (of actual weights)
        predictions = cross_val_predict(model, self.data_choice, self.prediction_input, cv=5)


        
        #print(predictions)
        #exit()
        return np.array([[round((pred), 0) for pred in lists] for lists in predictions]), n_scores, mean_scores

    

    def explain_with_shap(self, model):
        #Initialize SHAP explainer
        explainer = shap.KernelExplainer(model.predict, self.training_data1)

        #Compute SHAP values
        shap_values = explainer.shap_values(self.training_data1)

        #Visualize SHAP summary plot
        shap.summary_plot(shap_values, self.training_data1, feature_names=self.combined_data.columns)

        #Optionally, save the plots as part of the results
        plt.savefig(f'shap_summary_plot_{model.estimator.__class__.__name__}.png')




