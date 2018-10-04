import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import time
import numpy as np
import itertools
import sys
import keras.backend as K
from keras.models import load_model
from PredictChromAlignNet import prepareDataForPrediction, runPrediction, getDistanceMatrix, assignGroups, alignTimes, printConfusionMatrix
from utils import getRealGroupAssignments

## Changed the normalisation behaviour to fit the training file 2018-04-30-TrainClassifierSiamese-MultiFolderTraining
## Provided a maximum cut off time for the peak comparison to limit the number of combinations
## Allow for repeats of each model

#%% Options

ignore_negatives = True  # Ignore groups assigned with a negative index?
time_cutoff = 3  # Three minutes
model_repeats = range(1,2)
model_numbers = [20, 21, 26] # range(1, 28)
model_names = {1:'A', 2:'B', 3:'C', 4:'D', 5:'E', 6:'F', 7:'G'}   # XRW
noPeakProfileModels = [2, 20, 22, 24, 26]  # Models where ignore_peak_profile = True

#%% Load and pre-process data

confusion_matrices = []
prediction_times = []
model_number = []

results_path = 'results'
if os.path.isdir(results_path) == False:
    os.makedirs(results_path)


j = int(sys.argv[1])
save_names = ['ModelTests-OnAir103.csv',
             'ModelTests-OnAir115.csv',
             'ModelTests-OnAir143.csv',
             'ModelTests-OnBreath103.csv',
             'ModelTests-OnBreath115.csv',
             'ModelTests-OnBreath73.csv',
             'ModelTests-OnBreath88.csv']
save_name = os.path.join(results_path,save_names[j])

model_path = 'SavedModels/'

model_files = 'ChromAlignNet-A-'

data_paths = ['../Data/2018-04-22-ExtractedPeaks-Air103-WithMassSlice/',
             '../Data/2018-04-30-ExtractedPeaks-Air115-WithMassSlice/',
             '../Data/2018-04-30-ExtractedPeaks-Air143-WithMassSlice/',
             '../Data/2018-05-01-ExtractedPeaks-Breath115-WithMassSlice/',
             '../Data/2018-05-01-ExtractedPeaks-Breath103-WithMassSlice/',
             '../Data/2018-05-14-ExtractedPeaks-Breath73-WithMassSlice-All/',
             '../Data/2018-05-14-ExtractedPeaks-Breath88-WithMassSlice-All/']

data_path = data_paths[j]

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


#%% Predict
info_file = 'PeakData-WithGroup.csv'
if os.path.isfile(os.path.join(data_path, info_file)):
    real_groups_available = True
else:
    info_file = 'PeakData.csv'
    real_groups_available = False
sequence_file = 'WholeSequence.csv'



for repeat in model_repeats:
    for i in model_numbers:
#    for i in range(1,7):
#    for i in range(len(model_files)):   # XRW -- also need to clean up this bit more
        prediction_data, comparisons, info_df, peak_df_orig, peak_df_max = prepareDataForPrediction(data_path, info_file, sequence_file, ignore_peak_profile = True if i in noPeakProfileModels else False)
        model_file = model_files + '{:02d}'.format(i) + '-r' + '{:02d}'.format(repeat)     # for submodel
        print('Model used: ', model_file)   #XRW
#        model_file = model_files[i] + '-r' + '{:02d}'.format(repeat)   # XRW
        
        # model_number.append(model_names[i+1])   # XRW -- for full models
        model_number.append(i)   # XRW -- for sub-models
        
        predict_time = time.time()
        
        prediction = runPrediction(prediction_data, model_path, model_file)
        
        print('Time to predict:', round((time.time() - predict_time)/60, 2), 'min')
        prediction_times.append(round((time.time() - predict_time)/60, 2))
    
        #%% Group and cluster
        
        clusterTime = time.time()
        
        distance_matrix = getDistanceMatrix(comparisons, prediction, clip = 10)
        
        groups = assignGroups(distance_matrix, threshold = 2)
        
        
        
        if real_groups_available:
            real_groups = getRealGroupAssignments(info_df)
        
        
        print('Time to cluster:', round((time.time() - clusterTime)/60, 2), 'min')
        
        #%% Plot spectrum and peaks
                  
        alignTimes(groups, info_df, 'AlignedTime')
        if real_groups_available:
            alignTimes(real_groups, info_df, 'RealAlignedTime')
        
        
        if real_groups_available:
            confusion_matrices.append(printConfusionMatrix(prediction, info_df, comparisons))
        

    cm_df = pd.DataFrame(confusion_matrices, columns = ['True Positives', 'False Positives', 'False Positives - Ignore Neg Indices'])
    
    df = pd.concat([cm_df, pd.DataFrame(prediction_times, columns = ['Prediction Times'])], axis = 1)
    
    df['Model Number'] = model_number
    
    df.to_csv(save_name)