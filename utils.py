import pandas as pd
import numpy as np
import os

def loadData(dataPath, infoFile = 'PeakData-WithGroup.csv', sequenceFile = 'WholeSequence.csv'):

    infoDf = pd.read_csv(os.path.join(dataPath, infoFile), index_col = 0)
    sequenceDf = pd.read_csv(os.path.join(dataPath, sequenceFile), index_col = 0)
    
    ### Load peak and mass slice profiles
    peakFiles = []
    massProfileFiles = []
    for f in os.listdir(dataPath):
        if f.endswith('.txt'):
            peakFiles.append(f)
            
        if f.endswith('.tsv'):
            massProfileFiles.append(f)
    
    peakFiles.sort()
    dfs = []
    for file in peakFiles:
        df = pd.read_csv(os.path.join(dataPath,file), header = None)
        dfs.append(df)
    peakDf = pd.concat(dfs, axis = 1)
    
    massProfileFiles.sort()
    dfs = []
    for file in massProfileFiles:
        df = pd.read_csv(os.path.join(dataPath,file), header = None)
        dfs.append(df)
    massProfileDf = pd.concat(dfs, axis = 1)
    
    del dfs
    del df
    
    ### Pre-process Data - Normalise peak height and remove abnormal samples
    peakDf = peakDf - np.min(peakDf)
    peakDf.fillna(0, inplace = True)
    
    peakDfOrig = peakDf.copy()
    peakDfOrig = peakDfOrig.transpose()
    peakDfOrig.reset_index(inplace = True, drop = True)

    # Normalise peaks
    peakDfMax = peakDf.max(axis=0)
    peakDf = peakDf.divide(peakDfMax, axis=1)
    peakDf = peakDf.transpose()
    peakDf.reset_index(inplace = True, drop = True)
    
    
    massProfileDf = massProfileDf - np.min(massProfileDf)
    massProfileDf.fillna(0, inplace = True)
    
    massProfileDfMax = massProfileDf.max(axis=0)
    massProfileDf = massProfileDf.divide(massProfileDfMax, axis=1)
    massProfileDf = massProfileDf.transpose()
    massProfileDf.reset_index(inplace = True, drop = True)
    
    
    idx = sequenceDf > 0
    sequenceDf[idx] = np.log2(sequenceDf[idx])
    sequenceDf = sequenceDf.transpose()
    
    
    peakDfMax.reset_index(inplace = True, drop = True)

    return infoDf, peakDf, massProfileDf, sequenceDf, peakDfOrig, peakDfMax






def printShapes():
    pass
#     print('trainingTime1:', trainingTime1.shape)
#     print('trainingTime2:', trainingTime2.shape)
#     print('trainingPeakProfile1:', trainingPeakProfile1.shape)
#     print('trainingPeakProfile2:', trainingPeakProfile2.shape)
#     print('trainingMassProfile1:', trainingMassProfile1.shape)
#     print('trainingMassProfile2:', trainingMassProfile2.shape)
#     print('trainingSequenceProfile1:', trainingSequenceProfile1.shape)
#     print('trainingSequenceProfile2:', trainingSequenceProfile2.shape)
#     print('trainingY:', trainingY.shape)
#     print('---')
#     print('testingTime1:', testingTime1.shape)
#     print('testingTime2:', testingTime2.shape)
#     print('testingPeakProfile1:', testingPeakProfile1.shape)
#     print('testingPeakProfile2:', testingPeakProfile2.shape)
#     print('testingMassProfile1:', testingMassProfile1.shape)
#     print('testingMassProfile2:', testingMassProfile2.shape)
#     print('testingSequenceProfile1:', testingSequenceProfile1.shape)
#     print('testingSequenceProfile2:', testingSequenceProfile2.shape)
#     print('testingY:', testingY.shape)
#     print('testingComparisions:', testingComparisions.shape)
