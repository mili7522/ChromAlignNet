import pandas as pd
import numpy as np
import os
import time
import sys
from PredictChromAlignNet import prepareDataForPrediction, runPrediction
from parameters import prediction_options, batch_prediction_options
from model_definition import getModelVariant
from utils import calculateMetrics


### Load parameters
data_paths = batch_prediction_options['data_paths']
model_repeats = batch_prediction_options['model_repeats']
model_variants = batch_prediction_options['model_variants']
model_names =  batch_prediction_options['model_names']

dataset_number = int(sys.argv[1])
data_path = data_paths[dataset_number]

calculate_f1_metric = prediction_options['calculate_f1_metric']
calculate_metrics_for_components = prediction_options['calculate_metrics_for_components']
results_path = prediction_options['results_path']

# Second system input of repetition number, thus need to modify the save_name variable
save_names = batch_prediction_options['save_names']
if len(sys.argv) > 2:
    repeat = int(sys.argv[2])
    save_name = os.path.join(results_path, 
                             'rep' + '{:02d}'.format(repeat) 
                             + '-' + save_names[dataset_number])
    model_repeats = [repeat]
else:
    save_name = os.path.join(results_path, save_names[dataset_number])


if batch_prediction_options['save_individual_predictions']:
    individual_predictions_save_path = batch_prediction_options['individual_predictions_save_path']
    if individual_predictions_save_path is not None:
        results_path_individual = os.path.join(results_path, individual_predictions_save_path)
    os.makedirs(results_path_individual, exist_ok = True)


###
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
    
            model_file = "{}-{}-{:02d}-r{:02d}".format(batch_prediction_options['model_prefix'], name, i, repeat)
            print('---\nModel used: ', model_file)  
    
            model_fullname_list.append(name + '-' + '{:02d}'.format(i))  
            
            if batch_prediction_options['save_individual_predictions']:
                predictions_save_name = '{}/{}_{}_Prediction.csv'.format(results_path_individual, model_file, batch_prediction_options['dataset_name'])
            else:
                predictions_save_name = None
            
            predict_time = time.time()
            predictions = runPrediction(prediction_data, model_file, verbose = batch_prediction_options['verbose_prediction'],
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
