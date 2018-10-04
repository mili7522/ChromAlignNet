import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import time
import sys
import os
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model
from utils import loadData, getChromatographSegmentDf, generateCombinationIndices
from model_definition import getModelVariant
from parameters import prediction_options


## Changed the normalisation behaviour to fit the training file 2018-04-30-TrainClassifierSiamese-MultiFolderTraining
## Provided a maximum cut off time for the peak comparison to limit the number of combinations

# Load parameters
ignoreNegatives = prediction_options.get('ignoreNegatives')
timeCutOff = prediction_options.get('timeCutOff')

modelPath = prediction_options.get('modelPath')
modelFile = prediction_options.get('modelFile')

dataPath = prediction_options.get('dataPath')
infoFile = prediction_options.get('infoFile')
sequenceFile = prediction_options.get('sequenceFile')


modelVariant = int(modelFile.split('-')[2])
ChromAlignModel = getModelVariant(modelVariant)
ignorePeakProfile = getattr(ChromAlignModel, 'ignorePeakProfile')


if os.path.isfile(os.path.join(dataPath, infoFile)):
    realGroupsAvailable = True
else:
    infoFile = 'PeakData.csv'
    realGroupsAvailable = False



### Load peak and mass slice profiles
def prepareDataForPrediction(dataPath, infoFile, sequenceFile, ignorePeakProfile = ignorePeakProfile):
    loadTime = time.time()

    infoDf, peakDf, massProfileDf, sequenceDf, peakDfOrig, peakDfMax = loadData(dataPath, infoFile, sequenceFile)

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

    surroundsDf = getChromatographSegmentDf(infoDf, sequenceDf, sequence_length = 600)

    #%% Generate data combinations
    comparisons = generateCombinationIndices(infoDf[keepIndex], timeCutOff = timeCutOff, returnY = False)
    x1 = comparisons[:,0]
    x2 = comparisons[:,1]

    x1Time = infoDf.loc[x1]['peakMaxTime'].values
    x2Time = infoDf.loc[x2]['peakMaxTime'].values
    dataTimeDiff = abs(x1Time - x2Time)
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

    print('Time to load and generate samples:', round((time.time() - loadTime)/60, 2), 'min')
    print('\n')   # XRW
    print('===============\nPredictions:\n---')   # XRW
    sys.stdout.flush()   # XRW


    if ignorePeakProfile:
        prediction_data = [dataMassProfile1, dataMassProfile2,
                            sequenceProfile1.reshape((samples, sequence_length, 1)),
                            sequenceProfile2.reshape((samples, sequence_length, 1)),
                            dataTimeDiff]
    else:
        prediction_data = [dataMassProfile1, dataMassProfile2,
                            dataPeakProfile1.reshape((samples, max_peak_seq_length, 1)),
                            dataPeakProfile2.reshape((samples, max_peak_seq_length, 1)),
                            sequenceProfile1.reshape((samples, sequence_length, 1)),
                            sequenceProfile2.reshape((samples, sequence_length, 1)),
                            dataTimeDiff]

    return prediction_data, comparisons, infoDf, peakDfMax, peakDfOrig

def runPrediction(prediction_data, modelPath, modelFile):
    #%% Predict
    K.clear_session()
    predictTime = time.time()
    ### Load model
    loading = os.path.join(modelPath, modelFile) + '.h5'
    print(loading)
    model = load_model(loading)

    prediction = model.predict(prediction_data,
                                verbose = 1)
    predAll = prediction
    prediction = prediction[0]  # Only take the main outcome

    #print('Time to predict:', round((time.time() - predictTime)/60, 2), 'min')
    print('Time to predict:', time.time() - predictTime, 'sec')
    return prediction


#%% Group and cluster
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
#    plt.figure()
#    dn = scipy.cluster.hierarchy.dendrogram(mergings, leaf_font_size = 3)
#    plt.savefig(dataPath + 'Dendrogram.png', dpi = 300, format = 'png', bbox_inches = 'tight')
    labels = scipy.cluster.hierarchy.fcluster(mergings, threshold, criterion = 'distance')
    
    groups = {}
    for i in range(max(labels)):
        groups[i] = set(np.where(labels == i + 1)[0])  # labels start at 1
    
    return groups


def getRealGroupAssignments(infoDf):
    groups = {}
    for group, indexes in infoDf.groupby('Group').groups.items():
        if group < 0: continue  # Don't align negative groups. Leave them with their original times
        groups[group] = set(indexes)
    return groups


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


def plotPeaks(times, infoDf, peakDf, minTime, maxTime, resolution = 1/300, buffer = 10):
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


def printConfusionMatrix(prediction, infoDf, comparisons):
    x1 = comparisons[:,0]
    x2 = comparisons[:,1]
    p = np.round(prediction).astype(int).reshape((-1))
    g1 = infoDf.loc[x1]['Group'].values
    g2 = infoDf.loc[x2]['Group'].values

    keep = (g1 >= 0) & (g2 >= 0)  # Ignore negative indices
    truth = (g1 == g2)
    truth_ignore_neg = (g1[keep] == g2[keep])
    p_ignore_neg = p[keep]

    print('True positives: {} / {} = {:.3f}'.format(np.sum(p_ignore_neg[truth_ignore_neg]), np.sum(truth_ignore_neg), np.mean(p_ignore_neg[truth_ignore_neg])))
    print('False positives - ignore negative indices: {} / {} = {:.3f}'.format(np.sum(p_ignore_neg[~truth_ignore_neg]), np.sum(~truth_ignore_neg), np.mean(p_ignore_neg[~truth_ignore_neg])))
    print('False positives: {} / {} = {:.3f}'.format(np.sum(p[~truth]), np.sum(~truth), np.mean(p[~truth])))
    
    TP = np.mean(p_ignore_neg[truth_ignore_neg])
    FP_ignore_neg = np.mean(p_ignore_neg[~truth_ignore_neg])
    FP = np.mean(p[~truth])

    return (TP, FP_ignore_neg, FP)

# def getWrongCases(saveName = None):
#     wrongCases = comparisons[(p != truth)]
#     if saveName is not None:
#         np.savetxt(saveName + '.doc', wrongCases, fmt = '%d', delimiter='    ')
#     return wrongCases


####

if __name__ == "__main__":
    prediction_data, comparisons, infoDf, peakDfMax, peakDfOrig = prepareDataForPrediction(dataPath, infoFile, sequenceFile)
    prediction = runPrediction(prediction_data, modelPath, modelFile)

    distanceMatrix = getDistanceMatrix(comparisons, prediction, clip = 10)

    groups = assignGroups(distanceMatrix, threshold = 2)

    alignTimes(groups, infoDf, 'AlignedTime')
    if realGroupsAvailable:
        realGroups = getRealGroupAssignments(infoDf)
        alignTimes(realGroups, infoDf, 'RealAlignedTime')
        print("Group Overlap:", round(groupOverlap(groups, realGroups),4))
        print('---')
        printConfusionMatrix(prediction, infoDf, comparisons)

    plotSpectrumTogether(infoDf, peakDfMax, withReal = realGroupsAvailable, saveName = 'SpectrumAll')
    if not ignoreNegatives:
        plotSpectrumTogether(infoDf[infoDf['Group'] >= 0], peakDfMax[infoDf['Group'] >= 0], withReal = realGroupsAvailable, saveName = 'SpectrumNonNeg')

    #plotPeaksTogether(infoDf[infoDf['Group'] >= 0], peakDf[infoDf['Group'] >= 0], withReal = realGroupsAvailable)
    #logPeaks = np.log2(peakDfOrig)
    #logPeaks[logPeaks < 0] = 0
    plotPeaksTogether(infoDf, peakDfOrig, withReal = realGroupsAvailable, saveName = 'PeaksAll')
    if not ignoreNegatives:
        plotPeaksTogether(infoDf[infoDf['Group'] >= 0], peakDfOrig[infoDf['Group'] >= 0], withReal = realGroupsAvailable, saveName = 'PeaksNonNeg')  # Peaks not normalised