import pandas as pd
import os
import matplotlib.pyplot as plt
import scipy
import numpy as np
import fnmatch

#data_path = '../Data/2018-04-22-ExtractedPeaks-Air103-WithMassSlice'
#data_path = '../Data/2018-04-30-ExtractedPeaks-Air115-WithMassSlice'
#data_path = '../Data/2018-04-30-ExtractedPeaks-Air143-WithMassSlice'
#data_path = '../Data/2018-04-23-ExtractedPeaks-Air103-FullTime10Files'
#data_path = '../Data/2018-05-01-ExtractedPeaks-Breath115-WithMassSlice'
data_path = 'Data/2018-05-01-ExtractedPeaks-Breath103-WithMassSlice'
#data_path = '../Data/2018-04-23-ExtractedPeaks-Air103-FullTime10Files'
#data_path = '../Data/2018-05-14-ExtractedPeaks-Breath73-WithMassSlice-All'
#data_path = '../Data/2018-05-14-ExtractedPeaks-Breath88-WithMassSlice-All'

info_file = 'PeakData.csv'
sequence_file = 'WholeSequence.csv'

info_df = pd.read_csv(os.path.join(data_path, info_file), index_col = 0)
chromatogram_df = pd.read_csv(os.path.join(data_path, sequence_file), index_col = 0)

### Load peak and mass slice profiles
peakFiles = []
massProfileFiles = []
code = []
for f in os.listdir(data_path):
    if fnmatch.fnmatch(f, '*.txt'):
        peakFiles.append(f)                
        nameSplit = f[4:-4].split('-')  # Skip ID in front
        code.append('-'.join([x.split('=')[-1] for x in nameSplit]))
    if fnmatch.fnmatch(f, '*.tsv'):
        massProfileFiles.append(f)                
        
peakFiles.sort()
dfs = []
for i, file in enumerate(peakFiles):
    df = pd.read_csv(os.path.join(data_path,file), header = None)
    dfs.append(df)
peak_df = pd.concat(dfs, axis = 1)
peak_df.columns = code

massProfileFiles.sort()
dfs = []
for i, file in enumerate(massProfileFiles):
    df = pd.read_csv(os.path.join(data_path,file), header = None)
    dfs.append(df)
mass_profile_df = pd.concat(dfs, axis = 1)
mass_profile_df.columns = code

del dfs
del df

### Pre-process Data
peak_df = peak_df.transpose()
peak_df.reset_index(inplace = True, drop = True)
peak_df.fillna(0, inplace = True)
mass_profile_df = mass_profile_df.transpose()
mass_profile_df.reset_index(inplace = True, drop = True)


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
    print(info_df.loc[index])
    peak_df.loc[index].transpose().plot(ax = axes[0,0])
    if plotAsSubplots:
        axes[0,0].ticklabel_format(scilimits = (0,3))
        axes[0,0].set_title('Peak profile', fontdict = {'fontsize': 18})
    else:
        plt.title('Peak profile')
        plt.show()
        
    mass_profile_df.loc[index].transpose().plot(ax = axes[0,1])
    if plotAsSubplots:
        axes[0,1].ticklabel_format(scilimits = (0,3))
        axes[0,1].set_title('Mass spectrum at the time of peak maximum', fontdict = {'fontsize': 18})
        axes[0,1].set_xlabel('m/z', fontdict = {'fontsize': 12})
    else:
        plt.title('Mass spectrum at the time of peak maximum')
        plt.show()
    
    sequenceIdx = np.argmin(np.abs(chromatogram_df.index - np.mean(info_df.loc[index]['peakMaxTime'])).values)
    axes[1,0].plot(chromatogram_df.iloc[max(0,sequenceIdx - margin) : sequenceIdx + margin], 'gray', alpha = 0.2, label = '_nolegend_')
    for i, file in enumerate(info_df.loc[index]['File']):
        p = axes[1,0].plot(chromatogram_df.iloc[max(0,sequenceIdx - margin) : sequenceIdx + margin][[str(file)]], linewidth=3, label = index[i])
        # Plot line to the top of the peak at 'peakMaxTime'. Helps keep track of which peak to look at
        axes[1,0].plot((info_df.loc[index[i]]['peakMaxTime'], info_df.loc[index[i]]['peakMaxTime']),
                  (0, max(peak_df.loc[index[i]])), color = p[-1].get_color(), label = '_nolegend_')
    axes[1,0].legend()
    axes[1,0].ticklabel_format(scilimits = (0,3))
    if plotAsSubplots:
        axes[1,0].set_title('Chromatograph segment', fontdict = {'fontsize': 18})
        axes[1,0].set_xlabel('Retention Time (min)', fontdict = {'fontsize': 12})
    else:
        plt.title('Chromatograph segment')
        plt.show()
    
    if plotLogSequence:
        axes[1,1].plot(chromatogram_df.iloc[max(0,sequenceIdx - margin) : sequenceIdx + margin], 'gray', alpha = 0.2, label = '_nolegend_')
        for i, file in enumerate(info_df.loc[index]['File']):
            segment = chromatogram_df.iloc[max(0,sequenceIdx - margin) : sequenceIdx + margin][[str(file)]]
            segment = segment[segment != 0]
            p = axes[1,1].semilogy(segment, linewidth=3, label = index[i])
            # Plot line to the top of the peak at 'peakMaxTime'. Helps keep track of which peak to look at
            axes[1,1].semilogy((info_df.loc[index[i]]['peakMaxTime'], info_df.loc[index[i]]['peakMaxTime']),
                      (np.min(segment), max(peak_df.loc[index[i]])), color = p[-1].get_color(), label = '_nolegend_')
        axes[1,1].legend()
        if plotAsSubplots:
            axes[1,1].set_title('Chromatograph segment - log scale', fontdict = {'fontsize': 18})
            axes[1,1].set_xlabel('Retention Time (min)', fontdict = {'fontsize': 12})
        else:
            plt.title('Chromatograph segment - log scale')
            plt.show()
    
#    plt.savefig('A.png', dpi = 250, format = 'png', bbox_inches = 'tight')