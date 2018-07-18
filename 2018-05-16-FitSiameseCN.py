import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import time
import numpy as np
import itertools

## Changed the normalisation behaviour to fit the training file 2018-04-30-TrainClassifierSiamese-MultiFolderTraining
## Provided a maximum cut off time for the peak comparison to limit the number of combinations

#%% Options

ignoreNegatives = True  # Ignore groups assigned with a negative index?
timeCutOff = 1 # Three minutes

#%% Load and pre-process data

loadTime = time.time()

modelPath = 'Saved Models/'
modelFile = '2018-05-21-Siamese_Net-D-01'
#dataPath = '../Data/2018-04-22-ExtractedPeaks-Air103-WithMassSlice/'
#dataPath = '../Data/2018-04-30-ExtractedPeaks-Air115-WithMassSlice/'
#dataPath = '../Data/2018-04-30-ExtractedPeaks-Air143-WithMassSlice/'
dataPath = '../Data/2018-05-01-ExtractedPeaks-Breath115-WithMassSlice/'
#dataPath = '../Data/2018-05-01-ExtractedPeaks-Breath103-WithMassSlice/'
#dataPath = '../Data/2018-05-14-ExtractedPeaks-Breath73-WithMassSlice-All/'
#dataPath = '../Data/2018-05-14-ExtractedPeaks-Breath88-WithMassSlice-All/'

#dataPath = '../Data/2018-04-23-ExtractedPeaks-Air103-FullTime10Files/'
#dataPath = '../Data/2018-05-29-ExtractedPeaks-Breath73-FullTime10Files/'


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
#%% Predict

from keras.models import load_model
import keras.backend as K

#for _ in range(5):
K.clear_session()

predictTime = time.time()

### Load model
siamese_net = load_model(os.path.join(modelPath, modelFile) + '.h5')


prediction = siamese_net.predict([dataMassProfile1, dataMassProfile2,
                                  dataPeakProfile1.reshape((samples, max_peak_seq_length, 1)),
                                  dataPeakProfile2.reshape((samples, max_peak_seq_length, 1)),
                                  sequenceProfile1.reshape((samples, sequence_length, 1)),
                                  sequenceProfile2.reshape((samples, sequence_length, 1)),
                                  dataTimeDiff])

predAll = prediction
prediction = prediction[0]  # Only take the main outcome

#print('Time to predict:', round((time.time() - predictTime)/60, 2), 'min')
print('Time to predict:', time.time() - predictTime, 'sec')

#%% Group and cluster

clusterTime = time.time()

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

distanceMatrix = getDistanceMatrix(comparisons, prediction, clip = 10)

def assignGroups(distanceMatrix, threshold = 2):
    import scipy.spatial
    import scipy.cluster
    
    sqform = scipy.spatial.distance.squareform(distanceMatrix)
    mergings = scipy.cluster.hierarchy.linkage(sqform, method = 'average')
#    plt.figure()
#    dn = scipy.cluster.hierarchy.dendrogram(mergings, leaf_font_size = 3)
#    plt.savefig(dataPath + 'Dendrogram.png', dpi = 300, format = 'png', bbox_inches = 'tight')
    labels = scipy.cluster.hierarchy.fcluster(mergings, threshold, criterion = 'distance')
    
    groups = {}
    for i in range(max(labels)):
        groups[i] = set(np.where(labels == i + 1)[0])  # labels start at 1
    
    return groups

groups = assignGroups(distanceMatrix, threshold = 2)

def assignGroup2(comparisons, prediction, peakIndex):
    import itertools
    import collections
    
    groupCount = itertools.count()
    assignedGroup = {}
    threshold = 0.98
    groups = collections.defaultdict(set)
    for x1 in peakIndex:
        tempGroup = set()
        if x1 in assignedGroup:
            groupNo = assignedGroup[x1]
        else:
            groupNo = None
            tempGroup.add(x1)
            
        rs, cs = np.where(comparisons == x1)  # rows and columns
        for j, r in enumerate(rs):
            if prediction[r] > threshold:
                x2 = comparisons[r, abs(cs[j] - 1)]
                if x2 not in assignedGroup:
                    if groupNo is None:
                        tempGroup.add(x2)
                    else:
                        assignedGroup[x2] = groupNo
                        groups[groupNo].add(x2)
                elif groupNo is None:  # Join x1 and peaks matching with it to the existing group of x2 (This accumulates incorrect assignments. Use a high threshold)
                    groupNo = assignedGroup[x2]
                    groups[groupNo].update(tempGroup)
                    for x in tempGroup:
                        assignedGroup[x] = groupNo
        if groupNo is None:
            groupNo = next(groupCount)
            groups[groupNo].update(tempGroup)
            for x in tempGroup:
                assignedGroup[x] = groupNo
   
    return groups

groups2 = assignGroup2(comparisons, prediction, infoDf.index)



def getRealGroupAssignments(infoDf):
    groups = {}
    for group, indexes in infoDf.groupby('Group').groups.items():
        groups[group] = set(indexes)
    return groups

if realGroupsAvailable:
    realGroups = getRealGroupAssignments(infoDf)


print('Time to cluster:', round((time.time() - clusterTime)/60, 2), 'min')

#%% Plot spectrum and peaks

    
def plotSpectrum(times, fileIndex, maxValues, resolution = 1/300, buffer = 5,
                 minTime = None, maxTime = None, ax = None, clip = 1E4):
    if minTime is None:
        minTime = min(times)
    timeIndex = np.round((times - minTime) / resolution).astype(int)
    if maxTime is None:
        maxTimeIndex = max(timeIndex)
    else:
        maxTimeIndex = np.ceil((maxTime - minTime) / resolution).astype(int)
    
    numberOfFiles = fileIndex.max() + 1
    spectrum = np.zeros((numberOfFiles, maxTimeIndex + buffer * 2))
#    spectrum[fileIndex, timeIndex + buffer] = 1
    spectrum[fileIndex, timeIndex + buffer] = np.clip(maxValues, 0, clip)
#    spectrum[fileIndex, timeIndex + buffer] = maxValues
    
    if ax is None:
        ax = plt.axes()
#    pcm = ax.imshow(spectrum, norm=colors.LogNorm(vmin=1, vmax=maxValues.max()), cmap = 'hot', aspect = 'auto',
    pcm = ax.imshow(spectrum, cmap = 'inferno', aspect = 'auto',
                extent = [minTime - buffer * resolution, maxTime + buffer * resolution, 0, 1])
    ax.set_axis_off()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    return pcm


def alignTimes(groups, infoDf, alignTo):
    infoDf[alignTo] = infoDf['peakMaxTime']
    for group in groups.values():
        times = infoDf.loc[group, 'peakMaxTime']
        averageTime = np.mean(times)
        infoDf.loc[group, alignTo] = averageTime
    

alignTimes(groups, infoDf, 'AlignedTime')
if realGroupsAvailable:
    alignTimes(realGroups, infoDf, 'RealAlignedTime')


def plotSpectrumTogether(infoDf, maxValues, withReal = False, saveName = None):
    minTime = min(infoDf['startTime'])
    maxTime = max(infoDf['endTime'])
    
    if withReal:
        fig, axes = plt.subplots(3,1)
    else:
        fig, axes = plt.subplots(2,1)
    axes[0].set_title('Unaligned', fontdict = {'fontsize': 11})
    plotSpectrum(infoDf.peakMaxTime, infoDf.File, maxValues,
                 minTime = minTime, maxTime = maxTime, ax = axes[0])
    axes[1].set_title('Aligned', fontdict = {'fontsize': 11})
    pcm = plotSpectrum(infoDf.AlignedTime, infoDf.File, maxValues,
                 minTime = minTime, maxTime = maxTime, ax = axes[1])
    if withReal:
        axes[2].set_title('Truth', fontdict = {'fontsize': 11})
        plotSpectrum(infoDf.RealAlignedTime, infoDf.File, maxValues,
                     minTime = minTime, maxTime = maxTime, ax = axes[2])
        
    # Put retention time as x axis on the bottom-most plot
    axes[-1].set_axis_on()
    axes[-1].get_xaxis().set_visible(True)
    axes[-1].spines['top'].set_visible(False)
    axes[-1].spines['right'].set_visible(False)
    axes[-1].spines['left'].set_visible(False)
    axes[-1].set_xlabel('Retention Time (min)', fontdict = {'fontsize': 11})
    
    plt.tight_layout()
#    fig.subplots_adjust(hspace = 0.3, wspace = 10)
#    fig.colorbar(pcm, ax=axes.ravel().tolist(), fraction = 0.05, pad = 0.01)
    
    if saveName is not None:
        plt.savefig(saveName + '.png', dpi = 250, format = 'png', bbox_inches = 'tight')
    else:
        plt.show()

#plotSpectrumTogether(infoDf, peakDfMax, withReal = realGroupsAvailable)
plotSpectrumTogether(infoDf[infoDf['Group'] >= 0], peakDfMax[infoDf['Group'] >= 0], withReal = realGroupsAvailable)


def plotPeaks(times, infoDf, peakDf, minTime, maxTime, resolution = 1/300, buffer = 5):
    '''
    resolution = minutes per index step
    '''
    numberOfFiles = infoDf.File.max() + 1
    timeSteps = np.ceil((maxTime - minTime) / resolution + buffer * 2).astype(int)
    peaks = np.zeros((timeSteps, numberOfFiles))
    for row in infoDf.iterrows():
        peakProfile = peakDf.loc[row[0]]
        peakProfile = peakProfile[np.flatnonzero(peakProfile)]  # Remove the zeros (which were added during the preprocessing)
        peakProfileLength = len(peakProfile)
        stepsFromPeak = np.round((row[1]['peakMaxTime'] - row[1]['startTime']) / resolution).astype(int)
        alignedPeakTime = times.loc[row[0]]
        peakStepsFromBeginning = np.round((alignedPeakTime - minTime) / resolution).astype(int)
        peaks[peakStepsFromBeginning - stepsFromPeak + buffer: peakStepsFromBeginning - stepsFromPeak + peakProfileLength + buffer,
              int(row[1]['File'])] = peakProfile
    
    times = np.linspace(minTime - resolution * buffer, maxTime + resolution * buffer, timeSteps)
    return peaks, times
        
def plotPeaksTogether(infoDf, peakDf, withReal = False, saveName = None):
    minTime = min(infoDf['startTime'])
    maxTime = max(infoDf['endTime'])
    peaks, _ = plotPeaks(infoDf.AlignedTime, infoDf, peakDf, minTime, maxTime)
    orig_peaks, time = plotPeaks(infoDf.peakMaxTime, infoDf, peakDf, minTime, maxTime)
    if withReal:
        real_peaks, time = plotPeaks(infoDf.RealAlignedTime, infoDf, peakDf, minTime, maxTime)
        fig, axes = plt.subplots(3,1)
        axes[2].plot(time, real_peaks)
        axes[2].set_title('Truth', fontdict = {'fontsize': 11})
    else:
        fig, axes = plt.subplots(2,1)
    axes[0].plot(time, orig_peaks)
    axes[0].set_title('Unaligned', fontdict = {'fontsize': 11})
    axes[1].plot(time, peaks)
    axes[1].set_title('Aligned', fontdict = {'fontsize': 11})
    for ax in axes[:-1]:
        ax.set_axis_off()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_xlim(time[0], time[-1])
        
    axes[-1].spines['top'].set_visible(False)
    axes[-1].spines['right'].set_visible(False)
    axes[-1].spines['left'].set_visible(False)
    axes[-1].get_yaxis().set_visible(False)
    axes[-1].set_xlim(time[0], time[-1])
    axes[-1].set_xlabel('Retention Time (min)', fontdict = {'fontsize': 11})
    
    plt.tight_layout()
    fig.subplots_adjust(hspace = 0.3, wspace = 10)
    
    if saveName is not None:
        plt.savefig(saveName + '.png', dpi = 250, format = 'png', bbox_inches = 'tight')
    else:
        plt.show()

#plotPeaksTogether(infoDf[infoDf['Group'] >= 0], peakDf[infoDf['Group'] >= 0], withReal = realGroupsAvailable)
#logPeaks = np.log2(peakDfOrig)
#logPeaks[logPeaks < 0] = 0
plotPeaksTogether(infoDf[infoDf['Group'] >= 0], peakDfOrig[infoDf['Group'] >= 0], withReal = realGroupsAvailable)  # Peaks not normalised


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


def printConfusionMatrix(prediction):
    p = np.round(prediction).astype(int).reshape((-1))
    g1 = infoDf.loc[x1]['Group'].values
    g2 = infoDf.loc[x2]['Group'].values
    keep = (g1 != -1) & (g2 != -1)  # Ignore index of -1
    truth = (g1[keep] == g2[keep])
    p = p[keep]
    print('True positives: {} / {} = {:.3f}'.format(np.sum(p[truth]), np.sum(truth), np.mean(p[truth])))
    print('False positives: {} / {} = {:.3f}'.format(np.sum(p[~truth]), np.sum(~truth), np.mean(p[~truth])))
    print('False negatives: {} / {} = {:.3f}'.format(np.sum(p[truth] == 0), np.sum(truth), np.mean(p[truth] == 0)))
    print('True negatives: {} / {} = {:.3f}'.format(np.sum(p[~truth] == 0), np.sum(~truth), np.mean(p[~truth] == 0)))

if realGroupsAvailable:
    print("Group Overlap:", round(groupOverlap(groups, realGroups),4))
    print('---')
    printConfusionMatrix(prediction)

def getWrongCases(saveName = None):
    wrongCases = comparisons[(p != truth)]
    if saveName is not None:
        np.savetxt(saveName + '.doc', wrongCases, fmt = '%d', delimiter='    ')
    return wrongCases