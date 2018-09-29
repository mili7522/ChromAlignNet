import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import time
import numpy as np
import itertools
import sys
from keras.models import load_model
import keras.backend as K

## Changed the normalisation behaviour to fit the training file 2018-04-30-TrainClassifierSiamese-MultiFolderTraining
## Provided a maximum cut off time for the peak comparison to limit the number of combinations
## Allow for repeats of each model

#%% Options

ignoreNegatives = True  # Ignore groups assigned with a negative index?
timeCutOff = 3  # Three minutes

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

modelPath = 'Saved Models/'
#modelFiles = ['2018-05-21-Siamese_Net-A-01',
#              '2018-05-21-Siamese_Net-B-01',
#              '2018-05-21-Siamese_Net-C-01',
#              '2018-05-21-Siamese_Net-D-01',
#              '2018-05-21-Siamese_Net-E-01',
#              '2018-05-21-Siamese_Net-F-01']

modelFiles = '2018-05-28-Siamese_Net-C-'

dataPaths = ['../Data/2018-04-22-ExtractedPeaks-Air103-WithMassSlice/',
             '../Data/2018-04-30-ExtractedPeaks-Air115-WithMassSlice/',
             '../Data/2018-04-30-ExtractedPeaks-Air143-WithMassSlice/',
             '../Data/2018-05-01-ExtractedPeaks-Breath115-WithMassSlice/',
             '../Data/2018-05-01-ExtractedPeaks-Breath103-WithMassSlice/',
             '../Data/2018-05-14-ExtractedPeaks-Breath73-WithMassSlice-All/',
             '../Data/2018-05-14-ExtractedPeaks-Breath88-WithMassSlice-All/']

dataPath = dataPaths[j]


#%% Predict


modelRepeats = ['f', 'g', 'h', 'i', 'j']

for repeat in modelRepeats:
    for i in range(1,20):
#    for i in range(1,7):
        modelFile = modelFiles + '{:02d}'.format(i) + repeat
#        modelFile = modelFiles[i-1] + repeat
        
        modelNumber.append(i)
        
        predictTime = time.time()
        
        ### Load model
        
        K.clear_session()
        
        siamese_net = load_model(os.path.join(modelPath, modelFile) + '.h5')
        
        if i == 2:
            prediction = siamese_net.predict([dataMassProfile1, dataMassProfile2,
    #                                      dataPeakProfile1.reshape((samples, max_peak_seq_length, 1)),
    #                                      dataPeakProfile2.reshape((samples, max_peak_seq_length, 1)),
                                          sequenceProfile1.reshape((samples, sequence_length, 1)),
                                          sequenceProfile2.reshape((samples, sequence_length, 1)),
                                          dataTimeDiff])
        else:
            prediction = siamese_net.predict([dataMassProfile1, dataMassProfile2,
                                              dataPeakProfile1.reshape((samples, max_peak_seq_length, 1)),
                                              dataPeakProfile2.reshape((samples, max_peak_seq_length, 1)),
                                              sequenceProfile1.reshape((samples, sequence_length, 1)),
                                              sequenceProfile2.reshape((samples, sequence_length, 1)),
                                              dataTimeDiff])
        
        prediction = prediction[0]  # Only take the main outcome
        
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
            confusionMatrices.append(printConfusionMatrix())
        

    CM_DF = pd.DataFrame(confusionMatrices, columns = ['True Positives', 'False Positives', 'False Negatives', 'True Negatives'])
    
    df = pd.concat([CM_DF, pd.DataFrame(groupOverlaps, columns = ['Group Overlaps']),
                                        pd.DataFrame(predictionTimes, columns = ['Prediction Times'])], axis = 1)
    
    df['Model Number'] = modelNumber
    
    df.to_csv(saveName)