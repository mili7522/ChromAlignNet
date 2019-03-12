import pandas as pd
import os
#import matplotlib.pyplot as plt
#import matplotlib.colors as colors
import time
#import numpy as np
#import itertools
import sys
#import keras.backend as K
#from keras.models import load_model
from PredictChromAlignNet import prepareDataForPrediction, runPrediction, getDistanceMatrix, assignGroups, alignTimes, printConfusionMatrix
from utils import getRealGroupAssignments
from parameters import prediction_options, training_options, batch_prediction_options
from model_definition import getModelVariant


### Options
model_path = prediction_options.get('model_path')
data_paths = training_options.get('datasets')
model_prefix = 'ChromAlignNet-'
results_path = prediction_options.get('results_path')
real_groups_available = prediction_options.get('real_groups_available')

model_repeats = batch_prediction_options.get('model_repeats')
model_variants = batch_prediction_options.get('model_variants')
model_names =  batch_prediction_options.get('model_names')  

verbose_prediction = batch_prediction_options.get('verbose_prediction') 


j = int(sys.argv[1])

# Second system input of repetition number, thus need to modify the save_name variable
save_names = batch_prediction_options.get('save_names')
if len(sys.argv) > 2:
    repeat = int(sys.argv[2])
    save_name = os.path.join(results_path, 
                             'rep' + '{:02d}'.format(repeat) 
                             + '-' + save_names[j])
    model_repeats = [repeat]
else:
    save_name = os.path.join(results_path, save_names[j])

data_path = data_paths[j]

individual_predictions_save_path = 'Individual'
os.makedirs(os.path.join(results_path, individual_predictions_save_path), exist_ok = True)


### Load and pre-process data
confusion_matrices = []
prediction_times = []
model_fullname_list = []  
#model_variant_list = []      # no longer needed
#model_name_list = []         # no longer needed


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
            # model_name_list.append(model_names[i+1])   # XRW -- for full models -- no longer needed
            # model_variant_list.append(i)   # XRW -- for sub-models  -- no longer needed
            
            predictions_save_name = '{}/{}/{}_{}_Prediction.csv'.format(results_path, individual_predictions_save_path, model_file, data_path.split('-')[1])  # TODO: Make saving optional?
            
            predict_time = time.time()
            prediction_all = runPrediction(prediction_data, model_path, model_file, verbose = verbose_prediction,
                                           predictions_save_name = predictions_save_name, comparisons = comparisons)
            prediction = prediction_all[0]
            
            print('Time to predict:', round((time.time() - predict_time)/60, 2), 'min')  # Duplicates what's inside runPrediction
            prediction_times.append(round((time.time() - predict_time)/60, 2))
            
            ### 
            confusion_matrices.append(printConfusionMatrix(prediction, info_df, comparisons) + 
                                      printConfusionMatrix(prediction_all[1], info_df, comparisons) + 
                                      printConfusionMatrix(prediction_all[2], info_df, comparisons))  # TODO: Make this more general
            sys.stdout.flush()  
            

        cm_df = pd.DataFrame(confusion_matrices, columns = ['True Positives', 'False Positives - Ignore Neg Idx', 'False Positives', 'Recall', 'Precision', 'F1',
                                                            'TP-Mass', 'FP-IgnNeg-Mass', 'FP-Mass', 'Recall-Mass', 'Precision-Mass', 'F1-Mass',
                                                            'TP-Chrom', 'FP-IgnNeg-Chrom', 'FP-Chrom', 'Recall-Chrom', 'Precision-Chrom', 'F1-Chrom'])  # TODO: Generalise
        df = pd.concat([cm_df, pd.DataFrame(prediction_times, columns = ['Prediction Times'])], axis = 1)
        # df['Model Name']   # no longer needed
        # df['Model Variant'] = model_variant_list   # no longer needed
        df['Model Name'] = model_fullname_list
        df.to_csv(save_name)

df.groupby('Model Name').mean().to_csv(save_name[:-4] + '_Mean.csv')
