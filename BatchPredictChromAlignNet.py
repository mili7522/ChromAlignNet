import pandas as pd
import numpy as np
import os
import time
import sys
from PredictChromAlignNet import prepareDataForPrediction, runPrediction
from parameters import prediction_options, training_options, batch_prediction_options, getDatasetName
from model_definition import getModelVariant
from utils import calculateMetrics


### Options
model_path = prediction_options.get('model_path')
data_paths = training_options.get('datasets')
model_prefix = 'ChromAlignNet-'
results_path = prediction_options.get('results_path')

calculate_f1_metric = prediction_options.get('calculate_f1_metric')
calculate_metrics_for_components = prediction_options.get('calculate_metrics_for_components')

model_repeats = batch_prediction_options.get('model_repeats')
model_variants = batch_prediction_options.get('model_variants')
model_names =  batch_prediction_options.get('model_names')  

verbose_prediction = batch_prediction_options.get('verbose_prediction') 
save_individual_predictions = batch_prediction_options.get('save_individual_predictions')

dataset_number = int(sys.argv[1])

# Second system input of repetition number, thus need to modify the save_name variable
save_names = batch_prediction_options.get('save_names')
if len(sys.argv) > 2:
    repeat = int(sys.argv[2])
    save_name = os.path.join(results_path, 
                             'rep' + '{:02d}'.format(repeat) 
                             + '-' + save_names[dataset_number])
    model_repeats = [repeat]
else:
    save_name = os.path.join(results_path, save_names[dataset_number])

data_path = data_paths[dataset_number]

if save_individual_predictions:
    individual_predictions_save_path = batch_prediction_options.get('individual_predictions_save_path')
    results_path = os.path.join(results_path, individual_predictions_save_path)
    os.makedirs(results_path, exist_ok = True)


### Load and pre-process data
metrics_list = []
prediction_times_list = []
model_fullname_list = []


# XRW
# Check input, make sure we're using the correct data file name and where 
# the results will be saved to. 
#   Using manual flush to force the printing, for situations when we are 
# checking the log file mid-calculation. 
print('Predicting for data: ')
print(data_path)
print('Results will be saved to: ')
print(save_name)
sys.stdout.flush()


# Determine names of metrics for dataframe
metric_names = ['True Positives', 'False Positives - Ignore Neg Idx', 'False Positives']
if calculate_f1_metric:
    metric_names.extend(['Recall', 'Precision', 'F1'])
    if calculate_metrics_for_components:
        metric_names.extend(['TP-Mass', 'FP-IgnNeg-Mass', 'FP-Mass', 'Recall-Mass', 'Precision-Mass', 'F1-Mass',
                             'TP-Peak', 'FP-IgnNeg-Peak', 'FP-Peak', 'Recall-Peak', 'Precision-Peak', 'F1-Peak',
                             'TP-Chrom', 'FP-IgnNeg-Chrom', 'FP-Chrom', 'Recall-Chrom', 'Precision-Chrom', 'F1-Chrom'])
elif calculate_metrics_for_components:
    metric_names.extend(['TP-Mass', 'FP-IgnNeg-Mass', 'FP-Mass', 'TP-Peak', 'FP-IgnNeg-Peak', 'FP-Peak',
                         'TP-Chrom', 'FP-IgnNeg-Chrom', 'FP-Chrom'])

### Predict
# changed the order of the loops. variant first, so the data only need to be loaded once for the variant. model name, and then the repeats. 
for i in model_variants:
    print('\n\n')     # this is just so the log file is nice   
    print('===============\nFor model variant', i)   
    
    chrom_align_model = getModelVariant(i)
    ignore_peak_profile = getattr(chrom_align_model, 'ignore_peak_profile')
    
    prediction_data, comparisons, info_df, peak_df_orig, peak_intensity = prepareDataForPrediction(data_path, ignore_peak_profile)
            
    for name in model_names:
        for repeat in model_repeats:
    
            model_file = model_prefix + name + '-' + '{:02d}'.format(i) + '-r' + '{:02d}'.format(repeat)     # for full model name now    # XRW 08-10
            print('---\nModel used: ', model_file)  
    
            model_fullname_list.append(name + '-' + '{:02d}'.format(i))  
            
            if save_individual_predictions:
                predictions_save_name = '{}/{}_{}_Prediction.csv'.format(results_path, model_file, getDatasetName(data_path))
            else:
                predictions_save_name = None
            
            predict_time = time.time()
            predictions = runPrediction(prediction_data, model_path, model_file, verbose = verbose_prediction,
                                           predictions_save_name = predictions_save_name, comparisons = comparisons)
            
            prediction_times_list.append(round((time.time() - predict_time)/60, 2))  # Currently in minutes
            
            ###
            metrics = calculateMetrics(predictions, info_df, comparisons, calculate_f1 = calculate_f1_metric,
                                       calculate_for_components = calculate_metrics_for_components, print_metrics = True)
            if calculate_metrics_for_components and ignore_peak_profile:  # Modify the returned metrics to include nan values for peak encoder
                if calculate_f1_metric:
                    metrics[12:12] = [np.nan] * 6
                else:
                    metrics[6:6] = [np.nan] * 3
            metrics_list.append(metrics)
            sys.stdout.flush()
            
        df = pd.DataFrame(metrics_list, columns = metric_names)
        df['Prediction Times'] = prediction_times_list
        df['Model Name'] = model_fullname_list
        df.to_csv(save_name)

df.groupby('Model Name').mean().to_csv(save_name[:-4] + '_Mean.csv')
