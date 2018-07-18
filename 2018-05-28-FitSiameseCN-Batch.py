import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import time
import numpy as np
import itertools
import sys

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

j = int(sys.argv[1])
saveNames = ['../2018-05-25-ModelTests-OnAir103-2.csv',
             '../2018-05-25-ModelTests-OnAir115-2.csv',
             '../2018-05-25-ModelTests-OnAir143-2.csv',
             '../2018-05-25-ModelTests-OnBreath115-2.csv',
             '../2018-05-25-ModelTests-OnBreath103-2.csv',
             '../2018-05-25-ModelTests-OnBreath73-2.csv',
             '../2018-05-25-ModelTests-OnBreath88-2.csv']
saveName = saveNames[j]

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


#%%

loadTime = time.time()

### Load infoDf
try:
    infoFile = 'PeakData-WithGroup.csv'
    infoDf = pd.read_csv(os.path.join(dataPath, infoFile), index_col = 0)
    realGroupsAvailable = True
except FileNotFoundError:
    infoFile = 'PeakData.csv'
    infoDf = pd.read_csv(os.path.join(dataPath, infoFile), index_col = 0)
    realGroupsAvailable = False

### Load sequenceDf
sequenceFile = 'WholeSequence.csv'
sequenceDf = pd.read_csv(os.path.join(dataPath, sequenceFile), index_col = 0)
idx = sequenceDf > 0
sequenceDf[idx] = np.log2(sequenceDf[idx])
sequenceDf = sequenceDf.transpose()
sequence_length = 600  # Specified by the model

### Load peak and mass slice profiles
peakFiles = []
massProfileFiles = []
code = []
for f in os.listdir(dataPath):
    if f.endswith('.txt'):
        peakFiles.append(f)
        
    if f.endswith('.tsv'):
        massProfileFiles.append(f)

peakFiles.sort()
dfs = []
for i, file in enumerate(peakFiles):
    df = pd.read_csv(os.path.join(dataPath,file), header = None)
    dfs.append(df)
peakDf = pd.concat(dfs, axis = 1)

massProfileFiles.sort()
dfs = []
for i, file in enumerate(massProfileFiles):
    df = pd.read_csv(os.path.join(dataPath,file), header = None)
    dfs.append(df)
massProfileDf = pd.concat(dfs, axis = 1)

del dfs
del df

### Pre-process Data - Normalise peak height and remove abnormal somples
peakDf = peakDf - np.min(peakDf)
peakDf.fillna(0, inplace = True)
#peakDf[peakDf < 0] = 0

peakDfOrig = peakDf; peakDfOrig = peakDfOrig.transpose(); peakDfOrig.reset_index(inplace = True, drop = True)
peakDfMax = peakDf.max(axis=0)
peakDf = peakDf.divide(peakDfMax, axis=1)
peakDf = peakDf.transpose()
peakDf.reset_index(inplace = True, drop = True)
peakDfMax.reset_index(inplace = True, drop = True)


massProfileDf = massProfileDf - np.min(massProfileDf)
massProfileDf.fillna(0, inplace = True)
#massProfileDf[peakDf < 0] = 0

massProfileDfMax = massProfileDf.max(axis=0)
massProfileDf = massProfileDf.divide(massProfileDfMax, axis=1)
massProfileDf = massProfileDf.transpose()
massProfileDf.reset_index(inplace = True, drop = True)


if ignoreNegatives and realGroupsAvailable:
    negatives = infoDf['Group'] < 0
    infoDf = infoDf[~negatives]
    peakDf = peakDf[~negatives]
    peakDfOrig = peakDfOrig[~negatives]
    peakDfMax = peakDfMax[~negatives]
    massProfileDf = massProfileDf[~negatives]
    infoDf.reset_index(inplace = True, drop = False)
    peakDf.reset_index(inplace = True, drop = True)
    peakDfOrig.reset_index(inplace = True, drop = True)
    peakDfMax.reset_index(inplace = True, drop = True)
    massProfileDf.reset_index(inplace = True, drop = True)
    print("Negative index ignored: {}".format(np.sum(negatives)))

keepIndex = (pd.notnull(peakDf).all(1)) & (pd.notnull(massProfileDf).all(1))
#infoDf = infoDf[keepIndex]
#peakDf = peakDf[keepIndex]
#massProfileDf = massProfileDf[keepIndex]


print("Dropped rows: {}".format(np.sum(keepIndex == False)))
print(np.flatnonzero(keepIndex == False))


# Create surroundsDf  (Doing it down here to avoid having to reset the index if ignoring negatives)
peaks = len(peakDf)
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
surroundsDf = pd.DataFrame(surroundsDf)


#%% Generate data combinations

comparisons = np.array(list(itertools.combinations(infoDf[keepIndex].index, 2)))
x1 = comparisons[:,0]
x2 = comparisons[:,1]

x1Time = infoDf.loc[x1]['peakMaxTime'].values
x2Time = infoDf.loc[x2]['peakMaxTime'].values
dataTimeDiff = abs(x1Time - x2Time)
withinTimeCutOff = dataTimeDiff < timeCutOff

comparisons = comparisons[withinTimeCutOff]
x1 = comparisons[:,0]
x2 = comparisons[:,1]
dataTimeDiff = dataTimeDiff[withinTimeCutOff]
x1Time = x1Time[withinTimeCutOff]
x2Time = x2Time[withinTimeCutOff]
dataPeakProfile1 = peakDf.loc[x1].values
dataPeakProfile2 = peakDf.loc[x2].values
dataMassProfile1 = massProfileDf.loc[x1].values
dataMassProfile2 = massProfileDf.loc[x2].values
sequenceProfile1 = surroundsDf.loc[x1].values
sequenceProfile2 = surroundsDf.loc[x2].values


samples, max_peak_seq_length = dataPeakProfile1.shape
_, max_mass_seq_length = dataMassProfile1.shape
_, sequence_length = sequenceProfile1.shape


print('Number of samples:', samples)
print('Max peak sequence length:', max_peak_seq_length)
print('Max mass sequence length:', max_mass_seq_length)
print('Surrounds sequence length:', sequence_length)


def printShapes():
    print('dataTimeDiff:', dataTimeDiff.shape)
    print('dataPeakProfile1:', dataPeakProfile1.shape)
    print('dataPeakProfile2:', dataPeakProfile2.shape)
    print('dataMassProfile1:', dataMassProfile1.shape)
    print('dataMassProfile2:', dataMassProfile2.shape)
    print('sequenceProfile1:', sequenceProfile1.shape)
    print('sequenceProfile2:', sequenceProfile2.shape)


print('Time to load and generate samples:', round((time.time() - loadTime)/60, 2), 'min')

#%% Define functions

def getDistances(prediction):
    distances = 1 / prediction
    return distances
    
def getDistanceMatrix(comparisons, prediction, clip = 10):
    
    distances = getDistances(prediction)
    
    maxIndex = np.max(comparisons) + 1
    
    distanceMatrix = np.empty((maxIndex, maxIndex))
    distanceMatrix.fill(clip)  # Clip value
    
    for i, (x1, x2) in enumerate(comparisons):
        distanceMatrix[x1, x2] = min(distances[i], clip)
        distanceMatrix[x2, x1] = min(distances[i], clip)
    
    for i in range(maxIndex):
        distanceMatrix[i,i] = 0
    
    return distanceMatrix

def assignGroups(distanceMatrix, threshold = 2):
    import scipy.spatial
    import scipy.cluster
    
    sqform = scipy.spatial.distance.squareform(distanceMatrix)
    mergings = scipy.cluster.hierarchy.linkage(sqform, method = 'average')
#        plt.figure()
#        dn = scipy.cluster.hierarchy.dendrogram(mergings, leaf_font_size = 3)
#        plt.savefig(dataPath + 'Dendrogram.png', dpi = 300, format = 'png', bbox_inches = 'tight')
    labels = scipy.cluster.hierarchy.fcluster(mergings, threshold, criterion = 'distance')
    
    groups = {}
    for i in range(max(labels)):
        groups[i] = set(np.where(labels == i + 1)[0])  # labels start at 1
    
    return groups


def getRealGroupAssignments(infoDf):
    groups = {}
    for group, indexes in infoDf.groupby('Group').groups.items():
        groups[group] = set(indexes)
    return groups


def alignTimes(groups, infoDf, alignTo):
    infoDf[alignTo] = infoDf['peakMaxTime']
    for group in groups.values():
        times = infoDf.loc[group, 'peakMaxTime']
        averageTime = np.mean(times)
        infoDf.loc[group, alignTo] = averageTime


def IOU(set1, set2):
    '''
    Intersection Over Union
    '''
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    
    return len(intersection) / len(union)


def groupOverlap(assignedGroups, realGroups):
    '''
    Gets maximum total IOU between the groups by a greedy approach
    Returns total IOU divided by the number of real groups
    '''
    remainingAssignedGroups = list(assignedGroups.keys())
    remainingRealGroups = set(realGroups.keys())
    
    iouDict = {}
    groupMatchDict = {}   
    iouSum = 0
    def getRemainingIOU():
        for g1 in remainingAssignedGroups:
            bestIOU = -np.inf
            for g2 in remainingRealGroups:
                iou = IOU(assignedGroups[g1], realGroups[g2])
                
                if iou > bestIOU:
                    bestIOU = iou
                    iouDict[g1] = iou
                    groupMatchDict[g1] = g2
    
    def findMaxIOU():
        maxIOU = -np.inf
        for g1 in remainingAssignedGroups:
            iou = iouDict[g1]
            if iou > maxIOU:
                maxIOU = iou
                maxIOUIndex = g1
        return maxIOU, maxIOUIndex
    
    # First pass to remove any iou = 1
    getRemainingIOU()
    remove = []
    for g1, iou in iouDict.items():
        if iou == 1:
            remove.append(g1)
            iouSum += 1
    for r in remove:
        remainingAssignedGroups.remove(r)
        remainingRealGroups.remove(groupMatchDict[r])
    
    while len(remainingAssignedGroups) > 0:
        getRemainingIOU()
        maxIOU, maxIOUIndex = findMaxIOU()
        iouSum += maxIOU
        remainingAssignedGroups.remove(maxIOUIndex)
        remainingRealGroups.remove(groupMatchDict[maxIOUIndex])
        if len(remainingRealGroups) == 0:
            break

    return (iouSum / len(realGroups) + iouSum / len(assignedGroups)) / 2
        
        
def printConfusionMatrix():
    p = np.round(prediction).astype(int).reshape((-1))
    g1 = infoDf.loc[x1]['Group'].values
    g2 = infoDf.loc[x2]['Group'].values
    keep = (g1 != -1) & (g2 != -1)  # Ignore index of -1
    truth = (g1[keep] == g2[keep])
    p = p[keep]
    
    TP = np.mean(p[truth])
    FP = np.mean(p[~truth])
    FN = np.mean(p[truth] == 0)
    TN = np.mean(p[~truth] == 0)
    return (TP, FP, FN, TN)
#%% Predict
from keras.models import load_model
import keras.backend as K

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