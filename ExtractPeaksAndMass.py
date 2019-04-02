import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from parameters import extraction_options


### Load options
data_path = extraction_options['data_path']
#masses = [39, 41, 42, 43, 55, 56, 57, 58, 71, 72, 74, 85, 100]
masses = [115]
filesToProcess = 10
peaksPerFile = 1000
#timeWindow = (13.8, 15.5)
timeWindow = (0, np.inf)
#timeWindow = (3, 6)
margin = 300
sortByPeakArea = False

save_path = extraction_options['save_path']
os.makedirs(save_path, exist_ok = True)  # Make the save_path if it doesn't exist





# Get files
files = []
for f in os.listdir(data_path):
    if f.endswith('.mat'):
        files.append(f)
#files.sort()
from random import shuffle
shuffle(files)


extractedPeakData = []
fileNo = 0
wholeSequenceData = []

for i in range(min(filesToProcess, len(files))):
    file = files[i]
    
    # Load data from .mat file
    data = loadmat(os.path.join(data_path, file))
    massZ = data['massZ']
    resMSPeak = data['resMSPeak']
    peakData = data['peakData']
    dataRaw = data['dataRaw']
    del data
    
    for mass in masses:
        massInd = np.flatnonzero(massZ == mass)[0]
        data = resMSPeak[massInd][0]
        
        
        try:
            timeWindowStartIndex = np.flatnonzero(data[:,2] > timeWindow[0])[0]  # Col 2 is peakMaxTime
            startInd = np.flatnonzero(dataRaw[0,:] > timeWindow[0])[0]
        except IndexError:  # Start of time window is beyond the range of the data
            wholeSequenceData.append(pd.DataFrame([0], index = [0.]))  # Placeholder column so the files don't get misaligned.
            #Index is made to be of type float to prevent error in pd.merge_asof(direction = nearest)
            continue
        try:
            timeWindowEndIndex = np.flatnonzero(data[:,2] > timeWindow[1])[0]
            endInd = np.flatnonzero(dataRaw[0,:] > timeWindow[1])[0]
        except IndexError:  # End of time window is beyond the range of the data
            timeWindowEndIndex = None
            endInd = None
        
        ## For plotting the whole peak sequence
        startInd = max(0, startInd - margin)
        if endInd is not None:
            endInd = endInd + margin
        wholeSequenceData.append(pd.DataFrame(dataRaw[massInd + 1, startInd:endInd], index = dataRaw[0, startInd:endInd]))
        
        try:
            if sortByPeakArea:
                idx = np.argsort(data[timeWindowStartIndex:timeWindowEndIndex,4])[::-1]  # Sort by peak area
                idx = idx[0:peaksPerFile]
                idx = idx + timeWindowStartIndex
            else:
                try:
                    idx = range(timeWindowStartIndex, timeWindowEndIndex)
                except TypeError:  # timeWindowEndIndex = None
                    idx = range(timeWindowStartIndex, data.shape[0])
            
            startTime = data[idx, 0]
            endTime = data[idx, 1]
            peakMaxTime = data[idx, 2]
            
#            print("File {}: Start - {}, End - {}".format(i, startTime, endTime))
            times = np.concatenate((startTime.reshape((1,-1)), endTime.reshape((1,-1)), peakMaxTime.reshape((1,-1))))
            miniDf = pd.DataFrame(times.T, columns = ['startTime','endTime','peakMaxTime'])
            miniDf['File'] = i
            miniDf['Mass'] = mass
            extractedPeakData.append(miniDf)
            
            for j in range(peaksPerFile):
                startInd = np.flatnonzero(dataRaw[0,:] == startTime[j])[0]
                endInd = np.flatnonzero(dataRaw[0,:] == endTime[j])[0]
                peakMaxInd = np.flatnonzero(dataRaw[0,:] == peakMaxTime[j])[0]
                
                extractedPeak = dataRaw[massInd + 1, startInd:endInd+1]
                massSlice = dataRaw[1:, peakMaxInd]
                plt.plot(extractedPeak)
                np.savetxt(save_path + '{:04d}-Mass={}-File={}-Peak={}.txt'.format(fileNo,mass, i, j), extractedPeak, delimiter = ',')
                np.savetxt(save_path + '{:04d}-Mass={}-File={}-Peak={}-MassSlice.tsv'.format(fileNo,mass, i, j), massSlice, delimiter = '\t')
                fileNo += 1
            

        except IndexError:
            pass

df = pd.concat(extractedPeakData)
df.reset_index(inplace = True)
df.columns = ['PeakNumber', 'startTime', 'endTime', 'peakMaxTime', 'File', 'Mass']
df.to_csv(save_path + 'PeakData.csv')
plt.savefig(save_path + 'FullPeakSequence-{}.png'.format(timeWindow), dpi = 300, format = 'png')

sortIndex = sorted(range(len(wholeSequenceData)), key = lambda x: len(wholeSequenceData[x]), reverse = True)

dfWholeSequence = wholeSequenceData[sortIndex[0]]
for i, idx in enumerate(sortIndex):
    if i == 0:
        continue
    dfWholeSequence = pd.merge_asof(dfWholeSequence, wholeSequenceData[idx], direction = 'nearest', left_index = True, right_index = True)
# Using direction = 'nearest' filles any blank values at the start with the first available value and 
# any blank values at the end with the last available value

#dfWholeSequence.interpolate(method='time', inplace = True)
#dfWholeSequence.fillna(0, inplace = True)
dfWholeSequence.columns = sortIndex
dfWholeSequence.sort_index(axis = 1, inplace = True)  # Rearrange the axis in order of the files
dfWholeSequence.to_csv(save_path + 'WholeSequence.csv')
