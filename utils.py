import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import os

def loadData(data_path, info_file = 'PeakData-WithGroup.csv', sequence_file = 'WholeSequence.csv', take_chromatogram_log = True):

    info_df = pd.read_csv(os.path.join(data_path, info_file), index_col = 0)
    info_df.dropna(axis = 1, how = 'all', inplace = True)  # Some empty columns are sometimes imported. Drop these
    chromatogram_df = pd.read_csv(os.path.join(data_path, sequence_file), index_col = 0)
    
    ### Load peak and mass spectra
    peakFiles = []
    massProfileFiles = []
    for f in os.listdir(data_path):
        if f.endswith('.txt'):
            peakFiles.append(f)
            
        if f.endswith('.tsv'):
            massProfileFiles.append(f)
    
    peakFiles.sort()
    dfs = []
    for file in peakFiles:
        df = pd.read_csv(os.path.join(data_path,file), header = None)
        dfs.append(df)
    peak_df = pd.concat(dfs, axis = 1)

    massProfileFiles.sort()
    dfs = []
    for file in massProfileFiles:
        df = pd.read_csv(os.path.join(data_path,file), header = None)
        dfs.append(df)
    mass_profile_df = pd.concat(dfs, axis = 1)

    del dfs
    del df
    
    ### Pre-process Data - Normalise peak height and remove abnormal samples
    peak_df = peak_df - np.min(peak_df)
    peak_df.fillna(0, inplace = True)

    peak_df_orig = peak_df.copy()
    peak_df_orig = peak_df_orig.transpose()

    # Normalise peaks
    peak_df_max = peak_df.max(axis=0)
    peak_df = peak_df.divide(peak_df_max, axis=1)
    peak_df = peak_df.transpose()


    mass_profile_df = mass_profile_df - np.min(mass_profile_df)
    mass_profile_df.fillna(0, inplace = True)

    mass_profile_df_max = mass_profile_df.max(axis=0)
    mass_profile_df = mass_profile_df.divide(mass_profile_df_max, axis=1)
    mass_profile_df = mass_profile_df.transpose()


    if take_chromatogram_log:
        idx = chromatogram_df > 0
        chromatogram_df[idx] = np.log2(chromatogram_df[idx])
    chromatogram_df = chromatogram_df.transpose()
    
    # Index starts off as all 0s due to concatonation and transposing. Resets to consecutive integers
    peak_df.reset_index(inplace = True, drop = True)
    peak_df_orig.reset_index(inplace = True, drop = True)
    mass_profile_df.reset_index(inplace = True, drop = True)
    peak_df_max.reset_index(inplace = True, drop = True)

    return info_df, peak_df, mass_profile_df, chromatogram_df, peak_df_orig, peak_df_max


def getChromatographSegmentDf(info_df, chromatogram_df, segment_length):
    peaks = len(info_df)
    chrom_seg_df = np.zeros((peaks, segment_length))
    peak_times = info_df['peakMaxTime']
    files = info_df['File'].apply(str)
    time_idx = np.argmin(np.abs(peak_times.values.reshape((1,-1)) - chromatogram_df.columns.values.reshape((-1,1))), axis = 0)
    
    for i in range(peaks):
        seq = np.zeros(segment_length)
        t = time_idx[i] - segment_length // 2
        if t < 0:
            seq[-t:] = chromatogram_df.loc[files.iloc[i]].iloc[:(time_idx[i] + segment_length // 2)].copy()
        else:
            insert = chromatogram_df.loc[files.iloc[i]].iloc[(time_idx[i] - segment_length // 2): (time_idx[i] + segment_length // 2)].copy()
            seq[:len(insert)] = insert    
        
        idx = seq > 0
        seq[idx] = seq[idx] - np.min(seq[idx])
        chrom_seg_df[i] = seq
    
    return pd.DataFrame(chrom_seg_df)


def generateCombinationIndices(info_df, time_cutoff = None, return_y = True, random_seed = None):
    if random_seed is not None:
        # Set seed to get the same dataset when continuing training from checkpoint
        np.random.seed(random_seed)

    comparisons = np.array(list(itertools.combinations(info_df.index, 2)))

    if time_cutoff is not None:
        x1 = comparisons[:,0]
        x2 = comparisons[:,1]
        x1_time = info_df.loc[x1]['peakMaxTime'].values
        x2_time = info_df.loc[x2]['peakMaxTime'].values
        data_time_diff = abs(x1_time - x2_time)
        within_time_cutoff = data_time_diff < time_cutoff
        comparisons = comparisons[within_time_cutoff]

    x1 = comparisons[:,0]
    x2 = comparisons[:,1]

    if return_y:
        # Redraw training examples to ensure that the number of negative examples matches the number of positive examples for each group
        x1_group = info_df.loc[x1,'Group']
        x2_group = info_df.loc[x2,'Group']
        new_x1 = []
        new_x2 = []
        y = []
        selected_for_different_group = np.zeros((len(x1)), dtype = bool)  # Avoid repetitions in the training set
        groups = info_df['Group'].unique()
        for group in groups:
            x1_in_group = (x1_group == group).values
            x2_in_group = (x2_group == group).values
            same_group = x1_in_group & x2_in_group
            different_group = (x1_in_group | x2_in_group) & (~same_group) & (~selected_for_different_group)
            # Convert boolean values into indices
            same_group = np.flatnonzero(same_group)
            different_group = np.flatnonzero(different_group)
            d_x1_times = info_df.loc[x1[different_group]]['peakMaxTime'].values
            d_x2_times = info_df.loc[x2[different_group]]['peakMaxTime'].values
            d_time_diff = np.abs(d_x1_times - d_x2_times)
            d_time_diff_inv = 1/(d_time_diff + 1E-4) ** 2
            p = d_time_diff_inv / np.sum(d_time_diff_inv)
            # Select a subset of the cases where groups are different, to keep positive and negative training examples balanced
            different_group = np.random.choice(different_group, size = len(same_group), replace = False, p = p)
            
            selected_for_different_group[different_group] = 1
            new_x1.extend(x1[same_group])
            new_x2.extend(x2[same_group])
            y.extend([1] * len(same_group))
            new_x1.extend(x1[different_group])
            new_x2.extend(x2[different_group])
            y.extend([0] * len(same_group))

        assert len(new_x1) == len(new_x2) == len(y)

        return np.array(new_x1), np.array(new_x2), y
    
    else:
        return comparisons


def getRealGroupAssignments(info_df):
    groups = {}
    for group, indexes in info_df.groupby('Group').groups.items():
        if group < 0: continue  # Don't align negative groups. Leave them with their original times
        groups[group] = set(indexes)
    return groups


def plotSpectrum(times, fileIndex, maxValues, resolution = 1/300, buffer = 5,
                 minTime = None, maxTime = None, ax = None, clip = 1E4):
    if minTime is None:
        minTime = min(times)
    timeIndex = np.round((times - minTime) / resolution).astype(int)
    if maxTime is None:
        maxTimeIndex = max(timeIndex)
    else:
        maxTimeIndex = np.ceil((maxTime - minTime) / resolution).astype(int)
    
    number_of_files = fileIndex.max() + 1
    spectrum = np.zeros((number_of_files, maxTimeIndex + buffer * 2))
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

def plotSpectrumTogether(info_df, maxValues, with_real = False, save_name = None):
    minTime = min(info_df['startTime'])
    maxTime = max(info_df['endTime'])
    
    if with_real:
        fig, axes = plt.subplots(3,1)
    else:
        fig, axes = plt.subplots(2,1)
    axes[0].set_title('Unaligned', fontdict = {'fontsize': 11})
    plotSpectrum(info_df.peakMaxTime, info_df.File, maxValues,
                 minTime = minTime, maxTime = maxTime, ax = axes[0])
    axes[1].set_title('Aligned', fontdict = {'fontsize': 11})
    pcm = plotSpectrum(info_df.AlignedTime, info_df.File, maxValues,
                 minTime = minTime, maxTime = maxTime, ax = axes[1])
    if with_real:
        axes[2].set_title('Truth', fontdict = {'fontsize': 11})
        plotSpectrum(info_df.RealAlignedTime, info_df.File, maxValues,
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
    
    if save_name is not None:
        plt.savefig(save_name + '.png', dpi = 250, format = 'png', bbox_inches = 'tight')
        plt.savefig(save_name + '.eps', format = 'eps', bbox_inches = 'tight')
    else:
        plt.show()


def plotPeaks(times, info_df, peak_df, minTime, maxTime, resolution = 1/300, buffer = 10):
    '''
    resolution = minutes per index step
    '''
    numberOfFiles = info_df.File.max() + 1
    timeSteps = np.ceil((maxTime - minTime) / resolution + buffer * 2).astype(int)
    peaks = np.zeros((timeSteps, numberOfFiles))
    for row in info_df.iterrows():
        peak = peak_df.loc[row[0]]
        peak = peak[np.flatnonzero(peak)]  # Remove the zeros (which were added during the preprocessing)
        peak_length = len(peak)
        stepsFromPeak = np.round((row[1]['peakMaxTime'] - row[1]['startTime']) / resolution).astype(int)
        alignedPeakTime = times.loc[row[0]]
        peakStepsFromBeginning = np.round((alignedPeakTime - minTime) / resolution).astype(int)
        peaks[peakStepsFromBeginning - stepsFromPeak + buffer: peakStepsFromBeginning - stepsFromPeak + peak_length + buffer,
                int(row[1]['File'])] = peak
    
    times = np.linspace(minTime - resolution * buffer, maxTime + resolution * buffer, timeSteps)
    return peaks, times


def plotPeaksTogether(info_df, peak_df, with_real = False, save_name = None):
    minTime = min(info_df['startTime'])
    maxTime = max(info_df['endTime'])
    peaks, _ = plotPeaks(info_df['AlignedTime'], info_df, peak_df, minTime, maxTime)
    orig_peaks, time = plotPeaks(info_df['peakMaxTime'], info_df, peak_df, minTime, maxTime)
    if with_real:
        real_peaks, time = plotPeaks(info_df['RealAlignedTime'], info_df, peak_df, minTime, maxTime)
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
    
    if save_name is not None:
        plt.savefig(save_name + '.png', dpi = 250, format = 'png', bbox_inches = 'tight')
        plt.savefig(save_name + '.eps', format = 'eps', bbox_inches = 'tight')
    else:
        plt.show()