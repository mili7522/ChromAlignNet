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
from PredictChromAlignNet import prepareDataForPrediction, runPrediction, getDistanceMatrix, assignGroups, getRealGroupAssignments, alignTimes, groupOverlap, printConfusionMatrix

## Changed the normalisation behaviour to fit the training file 2018-04-30-TrainClassifierSiamese-MultiFolderTraining
## Provided a maximum cut off time for the peak comparison to limit the number of combinations
## Allow for repeats of each model

#%% Options

ignoreNegatives = True  # Ignore groups assigned with a negative index?
timeCutOff = 3  # Three minutes
modelRepeats = range(1,2)
modelNumbers = [20, 21, 26] # range(1, 28)
modelNames = {1:'A', 2:'B', 3:'C', 4:'D', 5:'E', 6:'F', 7:'G'}   # XRW
noPeakProfileModels = [2, 20, 22, 24, 26]  # Models where ignorePeakProfile = True

#%% Load and pre-process data

groupOverlaps = []
confusionMatrices = []
predictionTimes = []
modelNumber = []

resultsPath = 'results'
if os.path.isdir(resultsPath) == False:
    os.makedirs(resultsPath)


j = int(sys.argv[1])
saveNames = ['ModelTests-OnAir103.csv',
             'ModelTests-OnAir115.csv',
             'ModelTests-OnAir143.csv',
             'ModelTests-OnBreath103.csv',
             'ModelTests-OnBreath115.csv',
             'ModelTests-OnBreath73.csv',
             'ModelTests-OnBreath88.csv']
saveName = os.path.join(resultsPath,saveNames[j])

modelPath = 'SavedModels/'
#modelFiles = ['2018-05-21-Siamese_Net-A-01',
#              '2018-05-21-Siamese_Net-B-01',
#              '2018-05-21-Siamese_Net-C-01',
#              '2018-05-21-Siamese_Net-D-01',
#              '2018-05-21-Siamese_Net-E-01',
#              '2018-05-21-Siamese_Net-F-01']

modelFiles = 'ChromAlignNet-A-'

dataPaths = ['../Data/2018-04-22-ExtractedPeaks-Air103-WithMassSlice/',
             '../Data/2018-04-30-ExtractedPeaks-Air115-WithMassSlice/',
             '../Data/2018-04-30-ExtractedPeaks-Air143-WithMassSlice/',
             '../Data/2018-05-01-ExtractedPeaks-Breath115-WithMassSlice/',
             '../Data/2018-05-01-ExtractedPeaks-Breath103-WithMassSlice/',
             '../Data/2018-05-14-ExtractedPeaks-Breath73-WithMassSlice-All/',
             '../Data/2018-05-14-ExtractedPeaks-Breath88-WithMassSlice-All/']

dataPath = dataPaths[j]

# XRW
# Check input, make sure we're using the correct data file name and where 
# the results will be saved to. 
#   Using manual flush to force the printing, for situations when we are 
# checking the log file mid-calculation. 

print('Predicting for data: ')
print(dataPath)
print('Results will be saved to: ')
print(saveName)
sys.stdout.flush()


#%% Predict
infoFile = 'PeakData-WithGroup.csv'
if os.path.isfile(os.path.join(dataPath, infoFile)):
    realGroupsAvailable = True
else:
    infoFile = 'PeakData.csv'
    realGroupsAvailable = False
sequenceFile = 'WholeSequence.csv'



for repeat in modelRepeats:
    for i in modelNumbers:
#    for i in range(1,7):
#    for i in range(len(modelFiles)):   # XRW -- also need to clean up this bit more
        prediction_data, comparisons, infoDf, peakDfMax, peakDfOrig = prepareDataForPrediction(dataPath, infoFile, sequenceFile, ignorePeakProfile = True if i in noPeakProfileModels else False)
        modelFile = modelFiles + '{:02d}'.format(i) + '-r' + '{:02d}'.format(repeat)     # for submodel
        print('Model used: ', modelFile)   #XRW
#        modelFile = modelFiles[i] + '-r' + '{:02d}'.format(repeat)   # XRW
        
        # modelNumber.append(modelNames[i+1])   # XRW -- for full models
        modelNumber.append(i)   # XRW -- for sub-models
        
        predictTime = time.time()
        
        prediction = runPrediction(prediction_data, modelPath, modelFile)
        
        print('Time to predict:', round((time.time() - predictTime)/60, 2), 'min')
        predictionTimes.append(round((time.time() - predictTime)/60, 2))
    
        #%% Group and cluster
        
        clusterTime = time.time()
        
        distanceMatrix = getDistanceMatrix(comparisons, prediction, clip = 10)
        
        groups = assignGroups(distanceMatrix, threshold = 2)
        
        
        
        if realGroupsAvailable:
            realGroups = getRealGroupAssignments(infoDf)
        
        
        print('Time to cluster:', round((time.time() - clusterTime)/60, 2), 'min')
        
        #%% Plot spectrum and peaks
                  
        alignTimes(groups, infoDf, 'AlignedTime')
        if realGroupsAvailable:
            alignTimes(realGroups, infoDf, 'RealAlignedTime')
        
        
        if realGroupsAvailable:
    #        print("Group Overlap:", round(groupOverlap(groups, realGroups),4))
            groupOverlaps.append(round(groupOverlap(groups, realGroups),4))
            print('---')
            confusionMatrices.append(printConfusionMatrix(prediction, infoDf, comparisons))
        

    CM_DF = pd.DataFrame(confusionMatrices, columns = ['True Positives', 'False Positives', 'False Positives - Ignore Neg Indices'])
    
    df = pd.concat([CM_DF, pd.DataFrame(groupOverlaps, columns = ['Group Overlaps']),
                                        pd.DataFrame(predictionTimes, columns = ['Prediction Times'])], axis = 1)
    
    df['Model Number'] = modelNumber
    
    df.to_csv(saveName)