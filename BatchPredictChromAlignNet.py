import pandas as pd
import numpy as np
import os
import time
import sys
from PredictChromAlignNet import prepareDataForPrediction, runPrediction
from parameters import prediction_options, batch_prediction_options
from modelDefinition import getModelVariant
from utilsData import calculateMetrics

"""
Runs the prediction script many times on one data set, over a range of model names,
model variants and repetitions.
Two arguments can be given from the command line when running the script. The first
argument is required and indicates the index of the data set to use. The second
argument is optional and can specify a particular repetition of the model to run.

Outputs:
    ModelTests-On*.csv -- CSV file containing the metrics from each prediction
    ModelTests-On*_Mean.csv -- CSV file containing summary metrics averaged across repetions.
    ./Individual/* -- Individual prediction outcomes can be saved if 'save_individual_predictions'
                      is True. The folder can be changed by modifying 'individual_predictions_save_path'
"""

### Load parameters
data_paths = batch_prediction_options['data_paths']  # Location of the data sets
model_repeats = batch_prediction_options['model_repeats']  # Which repetitions of the model were trained
model_variants = batch_prediction_options['model_variants']  # The variant of the model that was trained (defined in modelDefinition.py)
model_names =  batch_prediction_options['model_names']  # The letter name of the model that was trained (indicating the data sets that were used in the training)

# Select the dataset to perform predictions for. The integer provided in an index
# which corresponds to the list of data sets in data_options
dataset_number = int(sys.argv[1])
data_path = data_paths[dataset_number]

calculate_f1_metric = prediction_options['calculate_f1_metric']  # If True, the F1 score is calculated as a metric as well as the true and false positives
calculate_auc_metric = prediction_options['calculate_auc_metric']  # If True, the AUC is calculated as a metric as well
calculate_average_precision_metric = prediction_options['calculate_average_precision_metric']  # If True, the average precision is calculated as a metric as well
calculate_metrics_for_components = prediction_options['calculate_metrics_for_components']  # If True, metrics are calculated for the subnetwork outputs
results_path = prediction_options['results_path']  # Path to save the summary output

# Second system input of repetition number. If it's present then need to modify the save_name variable
# If it's not present then run the prediction using all repetitions
save_names = batch_prediction_options['save_names']  # Gives the name of the csv file to be saved with the metrics from each prediction
if len(sys.argv) > 2:
    repeat = int(sys.argv[2])
    save_name = os.path.join(results_path, 
                             'rep' + '{:02d}'.format(repeat) 
                             + '-' + save_names[dataset_number])
    model_repeats = [repeat]
else:
    save_name = os.path.join(results_path, save_names[dataset_number])

# The individual prediction outputs can be saved
if batch_prediction_options['save_individual_predictions']:
    individual_predictions_save_path = batch_prediction_options['individual_predictions_save_path']
    if individual_predictions_save_path is None:
        results_path_individual = results_path  # Save in the main results folder
    else:
        results_path_individual = os.path.join(results_path, individual_predictions_save_path) # Save in a subfolder
    os.makedirs(results_path_individual, exist_ok = True)



# Check input, make sure we're using the correct data file name and where 
# the results will be saved to. 
print('Predicting for data: ')
print(data_path)
print('Results will be saved to: ')
print(save_name)
sys.stdout.flush()  # Using manual flush to force the printing, for situations when we are checking the log file mid-calculation. 


# Determine the names of the metrics to save in the summary dataframe
metric_names = ['True Positives', 'False Positives - Ignore Neg Idx', 'False Positives']
mass_metric_names = ['TP-Mass', 'FP-IgnNeg-Mass', 'FP-Mass']
peak_metric_names = ['TP-Peak', 'FP-IgnNeg-Peak', 'FP-Peak']
chrom_metric_names = ['TP-Chrom', 'FP-IgnNeg-Chrom', 'FP-Chrom']
if calculate_f1_metric:
    metric_names.extend(['Recall', 'Precision', 'F1'])
    mass_metric_names.extend(['Recall-Mass', 'Precision-Mass', 'F1-Mass'])
    peak_metric_names.extend(['Recall-Peak', 'Precision-Peak', 'F1-Peak'])
    chrom_metric_names.extend(['Recall-Chrom', 'Precision-Chrom', 'F1-Chrom'])
if calculate_auc_metric:
    metric_names.append('AUC')
    mass_metric_names.append('AUC-Mass')
    peak_metric_names.append('AUC-Peak')
    chrom_metric_names.append('AUC-Chrom')
if calculate_average_precision_metric:
    metric_names.append('Average Precision')
    mass_metric_names.append('AP-Mass')
    peak_metric_names.append('AP-Peak')
    chrom_metric_names.append('AP-Chrom')
if calculate_metrics_for_components:
    metric_names.extend(mass_metric_names + peak_metric_names + chrom_metric_names)
metric_len = len(peak_metric_names)

    
### Initialise
if batch_prediction_options['continue_from_saved'] and os.path.isfile(save_name):
    df = pd.read_csv(save_name, index_col = 0)
    metrics_list = list(df[metric_names].values)
    prediction_times_list = list(df['Prediction Times'])
    model_fullname_list = list(df['Model Name'])
    repetition_list = list(df['Repetition'])
    check_saved = True

else:
    metrics_list = []
    prediction_times_list = []
    model_fullname_list = []
    repetition_list = []
    check_saved = False
    
### Predict
# The variant loop is first, so the data only need to be loaded once for each variant (some variants do not use the peak data, hence the reloading between variants)
for i in model_variants:
    print('\n\n')  # This is just so the log file is nice   
    print('===============\nFor model variant', i)   
    
    # Get the correct model variant and extract the ignore_peak_profile attribute
    chrom_align_model = getModelVariant(i)
    ignore_peak_profile = getattr(chrom_align_model, 'ignore_peak_profile')
    
    prediction_data, comparisons, info_df, peak_df_orig, peak_intensity = prepareDataForPrediction(data_path, ignore_peak_profile)
            
    for name in model_names:
        model_fullname = name + '-' + '{:02d}'.format(i)  # Full name of the model - eg 'H-01'
        
        for repeat in model_repeats:
            # Check if this repetition has already been done
            if check_saved:
                if ((df['Model Name'] == model_fullname) & (df['Repetition'] == repeat)).any():
                    continue
            
            # Combine the model name, variant and repetition to get the name of the model file to load
            model_file = "{}-{}-{:02d}-r{:02d}".format(batch_prediction_options['model_prefix'], name, i, repeat)
            print('---\nModel used: ', model_file)  
    
            model_fullname_list.append(model_fullname)
            repetition_list.append(repeat)
            
            if batch_prediction_options['save_individual_predictions']:
                predictions_save_name = '{}/{}_{}_Prediction.csv'.format(results_path_individual, model_file, batch_prediction_options['dataset_name'][dataset_number])
            else:
                predictions_save_name = None
            
            # Loads the saved model and runs the prediction
            predict_time = time.time()
            predictions = runPrediction(prediction_data, model_file, verbose = batch_prediction_options['verbose_prediction'],
                                           predictions_save_name = predictions_save_name, comparisons = comparisons)
            
            prediction_times_list.append(round((time.time() - predict_time)/60, 2))  # Prediction time in minutes
            
            # Get the metrics of the prediction results
            metrics = calculateMetrics(predictions, info_df, comparisons, calculate_f1 = calculate_f1_metric,
                                       calculate_auc = calculate_auc_metric, calculate_average_precision = calculate_average_precision_metric,
                                       calculate_for_components = calculate_metrics_for_components, print_metrics = True)
            
            # Modify the returned metrics to include nan values for peak encoder (so the columns line up between different model variants)
            if calculate_metrics_for_components and ignore_peak_profile:  
                metrics[metric_len * 2 : metric_len * 2] = [np.nan] * metric_len

            metrics_list.append(metrics)
            sys.stdout.flush()
            
        df = pd.DataFrame(metrics_list, columns = metric_names)
        df['Prediction Times'] = prediction_times_list
        df['Model Name'] = model_fullname_list
        df['Repetition'] = repetition_list
        df.dropna(axis='columns', how = 'all', inplace = False).to_csv(save_name)  # Saves the summary output after every prediction
        
        
if len(sys.argv) == 2:  # No selection of model repeat
    # Saves a mean output at the end (averaged over the model repetitions)
    df.groupby('Model Name').mean().dropna(axis='columns', how = 'all', inplace = False).to_csv(save_name[:-4] + '_Mean.csv')
