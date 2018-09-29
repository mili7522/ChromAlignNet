import pandas as pd
import numpy as np
import itertools
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


# def printShapes():
#     print('dataTimeDiff:', dataTimeDiff.shape)
#     print('dataPeakProfile1:', dataPeakProfile1.shape)
#     print('dataPeakProfile2:', dataPeakProfile2.shape)
#     print('dataMassProfile1:', dataMassProfile1.shape)
#     print('dataMassProfile2:', dataMassProfile2.shape)
#     print('sequenceProfile1:', sequenceProfile1.shape)
#     print('sequenceProfile2:', sequenceProfile2.shape)


def getChromatographSegmentDf(infoDf, sequenceDf, sequence_length):
    peaks = len(infoDf)
    surroundsDf = np.zeros((peaks, sequence_length))
    peakTimes = infoDf['peakMaxTime']
    files = infoDf['File'].apply(str)
    timeIdx = np.argmin(np.abs(peakTimes.values.reshape((1,-1)) - sequenceDf.columns.values.reshape((-1,1))), axis = 0)
    
    for i in range(peaks):
        seq = np.zeros(sequence_length)
        t = timeIdx[i] - sequence_length // 2
        if t < 0:
            seq[-t:] = sequenceDf.loc[files.iloc[i]].iloc[:(timeIdx[i] + sequence_length // 2)].copy()
        else:
            insert = sequenceDf.loc[files.iloc[i]].iloc[(timeIdx[i] - sequence_length // 2): (timeIdx[i] + sequence_length // 2)].copy()
            seq[:len(insert)] = insert    
        
        idx = seq > 0
        seq[idx] = seq[idx] - np.min(seq[idx])
        surroundsDf[i] = seq
    
    return pd.DataFrame(surroundsDf)


def generateCombinationIndices(infoDf, timeCutOff = None, returnY = True, shuffle = False, setRandomSeed = None):
    if setRandomSeed is not None:
        # Set seed for repeatability in tests and to get the same dataset when continuing training from checkpoint
        np.random.seed(setRandomSeed)

    comparisons = np.array(list(itertools.combinations(infoDf.index, 2)))

    if timeCutOff is not None:
        x1 = comparisons[:,0]
        x2 = comparisons[:,1]
        x1Time = infoDf.loc[x1]['peakMaxTime'].values
        x2Time = infoDf.loc[x2]['peakMaxTime'].values
        dataTimeDiff = abs(x1Time - x2Time)
        withinTimeCutOff = dataTimeDiff < timeCutOff
        comparisons = comparisons[withinTimeCutOff]

    x1 = comparisons[:,0]
    x2 = comparisons[:,1]

    if returnY:
        x1_group = infoDf.loc[x1,'Group']
        x2_group = infoDf.loc[x2,'Group']
        new_x1 = []
        new_x2 = []
        y = []
        groups = infoDf['Group'].unique()
        for group in groups:
            x1_in_group = x1_group == group
            x2_in_group = x2_group == group
            same_group = np.flatnonzero(np.logical_and(x1_in_group, x2_in_group))
            different_group = np.flatnonzero(np.logical_and(x1_in_group, np.logical_not(x2_in_group)))
            # Select a subset of the cases where groups are different, to keep positive and negative training examples balanced
            different_group = np.random.choice(different_group, size = len(same_group))
            new_x1.extend(x1[same_group])
            new_x2.extend(x2[same_group])
            y.extend([1] * len(same_group))
            new_x1.extend(x1[different_group])
            new_x2.extend(x2[different_group])
            y.extend([0] * len(same_group))

        assert len(new_x1) == len(new_x2) == len(y)



        return np.array(new_x1), np.array(new_x2), y

    return comparisons