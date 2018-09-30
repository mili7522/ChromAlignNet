import pandas as pd
import os
import matplotlib.pyplot as plt
import scipy
import numpy as np
import fnmatch

#dataPath = '../Data/2018-04-22-ExtractedPeaks-Air103-WithMassSlice'
#dataPath = '../Data/2018-04-30-ExtractedPeaks-Air115-WithMassSlice'
#dataPath = '../Data/2018-04-30-ExtractedPeaks-Air143-WithMassSlice'
#dataPath = '../Data/2018-04-23-ExtractedPeaks-Air103-FullTime10Files'
#dataPath = '../Data/2018-05-01-ExtractedPeaks-Breath115-WithMassSlice'
dataPath = 'Data/2018-05-01-ExtractedPeaks-Breath103-WithMassSlice'
#dataPath = '../Data/2018-04-23-ExtractedPeaks-Air103-FullTime10Files'
#dataPath = '../Data/2018-05-14-ExtractedPeaks-Breath73-WithMassSlice-All'
#dataPath = '../Data/2018-05-14-ExtractedPeaks-Breath88-WithMassSlice-All'

infoFile = 'PeakData.csv'
sequenceFile = 'WholeSequence.csv'

infoDf = pd.read_csv(os.path.join(dataPath, infoFile), index_col = 0)
sequenceDf = pd.read_csv(os.path.join(dataPath, sequenceFile), index_col = 0)

### Load peak and mass slice profiles
peakFiles = []
massProfileFiles = []
code = []
for f in os.listdir(dataPath):
    if fnmatch.fnmatch(f, '*.txt'):
        peakFiles.append(f)                
        nameSplit = f[4:-4].split('-')  # Skip ID in front
        code.append('-'.join([x.split('=')[-1] for x in nameSplit]))
    if fnmatch.fnmatch(f, '*.tsv'):
        massProfileFiles.append(f)                
        
peakFiles.sort()
dfs = []
for i, file in enumerate(peakFiles):
    df = pd.read_csv(os.path.join(dataPath,file), header = None)
    dfs.append(df)
peakDf = pd.concat(dfs, axis = 1)
peakDf.columns = code

massProfileFiles.sort()
dfs = []
for i, file in enumerate(massProfileFiles):
    df = pd.read_csv(os.path.join(dataPath,file), header = None)
    dfs.append(df)
massProfileDf = pd.concat(dfs, axis = 1)
massProfileDf.columns = code

del dfs
del df

### Pre-process Data
peakDf = peakDf.transpose()
peakDf.reset_index(inplace = True, drop = True)
peakDf.fillna(0, inplace = True)
massProfileDf = massProfileDf.transpose()
massProfileDf.reset_index(inplace = True, drop = True)


def plotByIndex(index = None, margin = 100, plotLogSequence = True, readClipboard = False, plotAsSubplots = False):
    if plotAsSubplots:
        fig, axes = plt.subplots(2,2)
    else:
        axes = np.array([[None] * 2, [plt] * 2], dtype=np.object)
    
    if index is None:
        if readClipboard:
            index = pd.read_clipboard(header = None).squeeze().tolist()
        else:
            index = []
            while True:
                i = input("Index:")
                if i == '': break
                else: index.append(int(i))
    print(infoDf.loc[index])
    peakDf.loc[index].transpose().plot(ax = axes[0,0])
    if plotAsSubplots:
        axes[0,0].ticklabel_format(scilimits = (0,3))
        axes[0,0].set_title('Peak profile', fontdict = {'fontsize': 18})
    else:
        plt.title('Peak profile')
        plt.show()
        
    massProfileDf.loc[index].transpose().plot(ax = axes[0,1])
    if plotAsSubplots:
        axes[0,1].ticklabel_format(scilimits = (0,3))
        axes[0,1].set_title('Mass spectrum at the time of peak maximum', fontdict = {'fontsize': 18})
        axes[0,1].set_xlabel('m/z', fontdict = {'fontsize': 12})
    else:
        plt.title('Mass spectrum at the time of peak maximum')
        plt.show()
    
    sequenceIdx = np.argmin(np.abs(sequenceDf.index - np.mean(infoDf.loc[index]['peakMaxTime'])).values)
    axes[1,0].plot(sequenceDf.iloc[max(0,sequenceIdx - margin) : sequenceIdx + margin], 'gray', alpha = 0.2, label = '_nolegend_')
    for i, file in enumerate(infoDf.loc[index]['File']):
        p = axes[1,0].plot(sequenceDf.iloc[max(0,sequenceIdx - margin) : sequenceIdx + margin][[str(file)]], linewidth=3, label = index[i])
        # Plot line to the top of the peak at 'peakMaxTime'. Helps keep track of which peak to look at
        axes[1,0].plot((infoDf.loc[index[i]]['peakMaxTime'], infoDf.loc[index[i]]['peakMaxTime']),
                  (0, max(peakDf.loc[index[i]])), color = p[-1].get_color(), label = '_nolegend_')
    axes[1,0].legend()
    axes[1,0].ticklabel_format(scilimits = (0,3))
    if plotAsSubplots:
        axes[1,0].set_title('Chromatograph segment', fontdict = {'fontsize': 18})
        axes[1,0].set_xlabel('Retention Time (min)', fontdict = {'fontsize': 12})
    else:
        plt.title('Chromatograph segment')
        plt.show()
    
    if plotLogSequence:
        axes[1,1].plot(sequenceDf.iloc[max(0,sequenceIdx - margin) : sequenceIdx + margin], 'gray', alpha = 0.2, label = '_nolegend_')
        for i, file in enumerate(infoDf.loc[index]['File']):
            segment = sequenceDf.iloc[max(0,sequenceIdx - margin) : sequenceIdx + margin][[str(file)]]
            segment = segment[segment != 0]
            p = axes[1,1].semilogy(segment, linewidth=3, label = index[i])
            # Plot line to the top of the peak at 'peakMaxTime'. Helps keep track of which peak to look at
            axes[1,1].semilogy((infoDf.loc[index[i]]['peakMaxTime'], infoDf.loc[index[i]]['peakMaxTime']),
                      (np.min(segment), max(peakDf.loc[index[i]])), color = p[-1].get_color(), label = '_nolegend_')
        axes[1,1].legend()
        if plotAsSubplots:
            axes[1,1].set_title('Chromatograph segment - log scale', fontdict = {'fontsize': 18})
            axes[1,1].set_xlabel('Retention Time (min)', fontdict = {'fontsize': 12})
        else:
            plt.title('Chromatograph segment - log scale')
            plt.show()
    
#    plt.savefig('A.png', dpi = 250, format = 'png', bbox_inches = 'tight')