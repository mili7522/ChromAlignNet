import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import os
import scipy.spatial
import scipy.cluster

def loadData(data_path, info_file = 'PeakData-WithGroup.csv', sequence_file = 'WholeSequence.csv', take_chromatogram_log = True):
    ''' Loads data into pandas dataframes
    Output:
        info_df - Dataframe of peak information. Number of rows = number of peaks
                  Columns are peak number, times (start, end and max value), file number, mass number and group number (if one is assigned)
                  Peak number is assigned from 0 within each chromatogram file
        peak_df - Dataframe of the peak profile of each peak. Normalised to max of 1. Number of rows = number of peaks
                  Each peak is assumed to be in its own .txt file, as created by the 'ExtractPeaksAndMass.py' script
        mass_profile_df - Dataframe of the mass spectrum associated with each peak's time of max value.
                  Each mass spectrum is assumed to be in its own .tsv file, as created by the 'ExtractPeaksAndMass.py' script
        chromatogram_df - Dataframe of each file's whole chromatogram. Number of rows = number of files
        peak_df_orig - Unnormalised peak_df
        peak_intensity - Max chromatogram reading from each peak. Number of rows = number of peaks

    Keyword arguments:
        data_path - String: Path of data folder
        info_file - String: Name of the file which will be loaded into info_df
        sequence_file - String: Name of the file which contains the whole chromatogram sequences for all files
        take_chromatogram_log - Boolean: whether the chromatogram_df is transformed by taking the log (base 2)
    '''
    info_df = pd.read_csv(os.path.join(data_path, info_file), index_col = 0)
    info_df.dropna(axis = 1, how = 'all', inplace = True)  # Some empty columns are sometimes imported. Drop these
    chromatogram_df = pd.read_csv(os.path.join(data_path, sequence_file), index_col = 0)
    
    ### Load peak and mass spectra (from individual files into one DataFrame)
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

    
    ### Pre-process the data - Subtract baseline and normalise height (for both peaks and mass spectrum)
    peak_df = peak_df - np.min(peak_df)
    peak_df.fillna(0, inplace = True)

    peak_df_orig = peak_df.copy()
    peak_df_orig = peak_df_orig.transpose()

    peak_intensity = peak_df.max(axis=0)
    peak_df = peak_df.divide(peak_intensity, axis=1)
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
    
    # The index starts off as all 0s due to concatonation and transposing. Reset this to consecutive integers
    peak_df.reset_index(inplace = True, drop = True)
    peak_df_orig.reset_index(inplace = True, drop = True)
    mass_profile_df.reset_index(inplace = True, drop = True)
    peak_intensity.reset_index(inplace = True, drop = True)

    return info_df, peak_df, mass_profile_df, chromatogram_df, peak_df_orig, peak_intensity


def getChromatographSegmentDf(info_df, chromatogram_df, segment_length):
    ''' Generates a DataFrame of chromatogram segments
    Outputs:
        chrom_seg_df - DataFrame containing chromatogram segments centred at each peak, extracted from the corresponding unaligned chromatograms

    Keyword arguments:
        info_df - DataFrame containing information about each peak, in particular the peak time and the file number
        chromatogram_df - DataFrame containing the full chromatograms from all files. Column titles are the times of each measurement
        segment_length - Int: total length of the resulting chromatogram segments
    '''
    peaks = len(info_df)  # Number of peaks
    chrom_seg_df = np.zeros((peaks, segment_length))
    peak_times = info_df['peakMaxTime']  # Peak time is defined as the time of maximum peak intensity
    files = info_df['File'].apply(str)  # File associated with each peak
    time_idx = np.argmin(np.abs(peak_times.values.reshape((1,-1)) - chromatogram_df.columns.values.reshape((-1,1))), axis = 0)  # Get the index value corresponding to the peak time
    
    for i in range(peaks):
        seq = np.zeros(segment_length)
        t = time_idx[i] - segment_length // 2  # Start at half the segment_length before the peak index
        if t < 0:  # If the start of the segment would be before the start of the full chromatogram, those unfilled values remain as zero
            seq[-t:] = chromatogram_df.loc[files.iloc[i]].iloc[:(time_idx[i] + segment_length // 2)].copy()
        else:
            insert = chromatogram_df.loc[files.iloc[i]].iloc[(time_idx[i] - segment_length // 2): (time_idx[i] + segment_length // 2)].copy()
            seq[:len(insert)] = insert  # This works even if len(insert) < segment_length because it reaches the end of the full chromatogram
        
        idx = seq > 0  # Ignore any sensor errors, which are reported as zero
        seq[idx] = seq[idx] - np.min(seq[idx])  # Subtract away the (non-zero) baseline value in each segment
        chrom_seg_df[i] = seq
    
    return pd.DataFrame(chrom_seg_df)


def generateCombinationIndices(info_df, time_cutoff = None, return_y = True, random_seed = None):
    ''' Generates pairwise training examples
    Outputs:
        comparisons - Numpy array. May be returned as two columns - x1 and x2
                      Contains the peak IDs of the two pairwise peaks. The peak ID is gives by the corresponding row index in info_df
        y - Only returned if return_y is true. A Numpy array equal to 1 where the two peaks belong to the same group, or 0 otherwise

    Keyword arguments:
        info_df - DataFrame of peak information
        time_cutoff - Int or Float: maximum time (min) between two pairwise peaks to still be considered a valid training/testing example
        return_y - Boolean: To return y (for training) or not return y (for prediction)
        random_seed - Int
    '''
    if random_seed is not None:
        # Set seed to either ensure randomness or to get the same dataset when continuing training from checkpoint
        np.random.seed(random_seed)

    comparisons = np.array(list(itertools.combinations(info_df.index, 2)))  # Generates all pairs of peaks (peak ID is given by info_df.index)

    if time_cutoff is not None:  # Only get the pairs that fall within the time_cutoff are included
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
        # This provides balance when training
        x1_group = info_df.loc[x1,'Group']
        x2_group = info_df.loc[x2,'Group']
        new_x1 = []
        new_x2 = []
        y = []
        selected_for_different_group = np.zeros((len(x1)), dtype = bool)  # Keep track of included combinations to avoid repetitions in the training set
        groups = info_df['Group'].unique()
        for group in groups:
            if group < 0: continue  # Prevents combinations where both groups have negative indices
            x1_in_group = (x1_group == group).values
            x2_in_group = (x2_group == group).values
            same_group = x1_in_group & x2_in_group
            different_group = (x1_in_group | x2_in_group) & (~same_group) & (~selected_for_different_group)
            # Convert boolean values into indices
            same_group = np.flatnonzero(same_group)
            different_group = np.flatnonzero(different_group)
            d_x1_times = info_df.loc[x1[different_group]]['peakMaxTime'].values
            d_x2_times = info_df.loc[x2[different_group]]['peakMaxTime'].values
            # Bias the selection of peaks which are close together as negative examples, to make the training examples more difficult
            d_time_diff = np.abs(d_x1_times - d_x2_times)
            d_time_diff_inv = 1/(d_time_diff + 1E-4) ** 2
            p = d_time_diff_inv / np.sum(d_time_diff_inv)
            # Select a subset of the cases where groups are different, to keep positive and negative training examples balanced
            different_group = np.random.choice(different_group, size = len(same_group), replace = False, p = p)
            
            selected_for_different_group[different_group] = 1  # Update which of the negative examples have already been selected
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
    ''' Identifies the group which has been manually assigned to each peak
    Outputs:
        group - Dictionary of group IDs, each containing a set of the peaks belonging to that group (peak ID given by the row index of the peak)
    '''
    groups = {}
    for group, indexes in info_df.groupby('Group').groups.items():
        if group < 0: continue  # Don't align negative groups. Leave them with their original times
        groups[group] = set(indexes)
    return groups


def plotSpectrum(times, files, peak_intensity, resolution = 1/300, buffer = 5,
                 min_time = None, max_time = None, ax = None, clip = 1E4):
    ''' Plots the spectrum of peak across the different chromatograms
    Output:
        pcm - matplotlib axis
    
    Keyword arguments:
        times - pandas Series giving the times of the peaks
        files - pandas Series of the file each peak belonged to
        peak_intensity - pandas Series of the maximum values of each peak
        resolution - minutes per time index step for the chromatogram
        buffer - Int: Extra time steps to add to each end of the spectrum output
        min_time - Float: Minumum time to draw the spectrum from (excluding buffer). Helps align several spectrum together
        max_time - Float: Maximum time to draw the spectrum from (excluding buffer)
        ax - matplotlib axis to draw the spectrum into
        clip - Int or Float: Maximum value of the intensity. Values above this are clipped
    '''
    if min_time is None:
        min_time = min(times)
    timeIndex = np.round((times - min_time) / resolution).astype(int)
    if max_time is None:
        max_time_index = max(timeIndex)
    else:
        max_time_index = np.ceil((max_time - min_time) / resolution).astype(int)
    
    number_of_files = files.max() + 1
    spectrum = np.zeros((number_of_files, max_time_index + buffer * 2))
#    spectrum[files, timeIndex + buffer] = 1
    spectrum[files, timeIndex + buffer] = np.clip(peak_intensity, 0, clip)
#    spectrum[files, timeIndex + buffer] = peak_intensity
    
    if ax is None:
        ax = plt.axes()
#    pcm = ax.imshow(spectrum, norm=colors.LogNorm(vmin=1, vmax=peak_intensity.max()), cmap = 'hot', aspect = 'auto',
    pcm = ax.imshow(spectrum, cmap = 'inferno', aspect = 'auto',
                extent = [min_time - buffer * resolution, max_time + buffer * resolution, 0, 1])
    ax.set_axis_off()  # Turn off the display of the axis lines and ticks
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    return pcm

def plotSpectrumTogether(info_df, peak_intensity, with_real = False, save_name = None):
    ''' Plots several spectra stacked together, to compare the prediction output with the input and groundtruth
    Keyword arguments:
        info_df - DataFrame containing information about each peak, including aligned and unaligned peak times and file number
        peak_intensity - pandas Series of the maximum values of each peak
        with_real - Boolean: To include the groundtruth as a third spectrum or not
        save_name - None or String: Name to save the figure
    '''
    # Get min_time and max_time to pass into each call of plotSpectrum, so that each spectrum is aligned
    min_time = min(info_df['startTime'])
    max_time = max(info_df['endTime'])
    
    if with_real:
        fig, axes = plt.subplots(3,1)
    else:
        fig, axes = plt.subplots(2,1)
    axes[0].set_title('Unaligned', fontdict = {'fontsize': 11})
    plotSpectrum(info_df['peakMaxTime'], info_df['File'], peak_intensity,
                 min_time = min_time, max_time = max_time, ax = axes[0])
    axes[1].set_title('Aligned', fontdict = {'fontsize': 11})
    pcm = plotSpectrum(info_df['AlignedTime'], info_df['File'], peak_intensity,
                 min_time = min_time, max_time = max_time, ax = axes[1])
    if with_real:
        axes[2].set_title('Truth', fontdict = {'fontsize': 11})
        plotSpectrum(info_df['RealAlignedTime'], info_df['File'], peak_intensity,
                     min_time = min_time, max_time = max_time, ax = axes[2])
        
    # Put retention time as x axis on the bottom-most plot
    axes[-1].set_axis_on()
    axes[-1].get_xaxis().set_visible(True)  # Only set the bottom axis line to be visible
    axes[-1].spines['top'].set_visible(False)
    axes[-1].spines['right'].set_visible(False)
    axes[-1].spines['left'].set_visible(False)
    axes[-1].set_xlabel('Retention Time (min)', fontdict = {'fontsize': 11})
    
    plt.tight_layout()
#    fig.subplots_adjust(hspace = 0.3, wspace = 10)
#    fig.colorbar(pcm, ax=axes.ravel().tolist(), fraction = 0.05, pad = 0.01)
    
    if save_name is not None:
        plt.savefig(save_name + '.png', dpi = 250, format = 'png', bbox_inches = 'tight')
        plt.savefig(save_name + '.eps', dpi = 500, format = 'eps', bbox_inches = 'tight')
    else:
        plt.show()


def plotPeaks(times, info_df, peak_df, min_time, max_time, resolution = 1/300, buffer = 10):
    ''' Recreates chromatograms from the individual peaks, each at their associated times
    Output:
        peaks - 2D numpy array with each row as a reconstructed chromatogram
        times - 1D numpy array of the times corresponding to each column of the peaks array
    
    Keyword arguments:
        times - pandas Series giving the times of the peaks
        info_df - DataFrame containing information about each peak, including aligned and unaligned peak times and file number
        peak_df - Dataframe of the peak profile of each peak
        min_time - Float: Minumum time of the chromatogram (excluding buffer)
        max_time - Float: Maximum time of the chromatogram (excluding buffer)
        resolution - minutes per time index step for the chromatogram
        buffer - Int: Extra time steps to add to each end of the output chromatogram
    '''
    number_of_files = info_df['File'].max() + 1
    time_steps = np.ceil((max_time - min_time) / resolution + buffer * 2).astype(int)
    peaks = np.zeros((time_steps, number_of_files))
    for row in info_df.iterrows():
        peak = peak_df.loc[row[0]]  # Peak profile
        peak = peak[np.flatnonzero(peak)]  # Remove the zeros (which were added during the preprocessing)
        peak_length = len(peak)
        steps_from_peak = np.round((row[1]['peakMaxTime'] - row[1]['startTime']) / resolution).astype(int)  # Number of timesteps from the start of the peak profile to its highest intensity
        peak_steps_from_beginning = np.round((times.loc[row[0]] - min_time) / resolution).astype(int)  # Index corresponding to the peak time (highest intensity)
        idx_start = peak_steps_from_beginning - steps_from_peak + buffer
        idx_end = peak_steps_from_beginning - steps_from_peak + peak_length + buffer
        current_values = peaks[idx_start : idx_end, int(row[1]['File'])]
        peaks[idx_start : idx_end, int(row[1]['File'])] = np.maximum(peak, current_values)  # Replace the default zeros of the reconstructed chromatogram with the peak profile at the appropriate time
    
    times = np.linspace(min_time - resolution * buffer, max_time + resolution * buffer, time_steps)
    return peaks, times


def plotPeaksTogether(info_df, peak_df, with_real = False, save_name = None):
    ''' Plots several reconstructed chromatograms stacked together, to compare the prediction output with the input and groundtruth
    Keyword arguments:
        info_df - DataFrame containing information about each peak, including aligned and unaligned peak times and file number
        peak_df - Dataframe of the peak profile of each peak
        with_real - Boolean: To include the groundtruth as a third plot or not
        save_name - None or String: Name to save the figure
    '''
    # Get min_time and max_time to pass into each call of plotPeaks, so that each plot is aligned
    min_time = min(info_df['startTime'])
    max_time = max(info_df['endTime'])
    peaks, _ = plotPeaks(info_df['AlignedTime'], info_df, peak_df, min_time, max_time)
    orig_peaks, time = plotPeaks(info_df['peakMaxTime'], info_df, peak_df, min_time, max_time)
    if with_real:
        real_peaks, _ = plotPeaks(info_df['RealAlignedTime'], info_df, peak_df, min_time, max_time)
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
    
    # Put retention time as x axis on the bottom-most plot
    axes[-1].spines['top'].set_visible(False)  # Only set the bottom axis line to be visible
    axes[-1].spines['right'].set_visible(False)
    axes[-1].spines['left'].set_visible(False)
    axes[-1].get_yaxis().set_visible(False)
    axes[-1].set_xlim(time[0], time[-1])
    axes[-1].set_xlabel('Retention Time (min)', fontdict = {'fontsize': 11})
    
    plt.tight_layout()
    fig.subplots_adjust(hspace = 0.3, wspace = 10)
    
    if save_name is not None:
        plt.savefig(save_name + '.png', dpi = 250, format = 'png', bbox_inches = 'tight')
        plt.savefig(save_name + '.eps', dpi = 250, format = 'eps', bbox_inches = 'tight')
    else:
        plt.show()


### Group and cluster
def getDistances(prediction):
    distances = 1 / (prediction)
    return distances
    

def getDistanceMatrix(comparisons, number_of_peaks, prediction, clip = 10, info_df = None):
    distances = getDistances(prediction)
    
    distance_matrix = np.empty((number_of_peaks, number_of_peaks))
    distance_matrix.fill(clip)  # Clip value
    
    for i, (x1, x2) in enumerate(comparisons):
        if info_df is not None and info_df.loc[x1, 'File'] == info_df.loc[x2, 'File']:
            val = min(distances[i] * 2, clip * 2)
            if np.abs(info_df.loc[x1, 'peakMaxTime'] - info_df.loc[x2, 'peakMaxTime']) > 0.09:
                val = clip * 5
        elif info_df is not None and np.abs(info_df.loc[x1, 'peakMaxTime'] - info_df.loc[x2, 'peakMaxTime']) > 0.4:
            val = min(distances[i] * 2, clip)
        else:
            val = min(distances[i], clip)
        distance_matrix[x1, x2] = distance_matrix[x2, x1] = val
    
    for i in range(number_of_peaks):
        distance_matrix[i,i] = 0
    
    return distance_matrix


def assignGroups(distance_matrix, threshold = 2):
    sqform = scipy.spatial.distance.squareform(distance_matrix)
    mergings = scipy.cluster.hierarchy.linkage(sqform, method = 'average')  # centroid works well? Previously used 'average'
    plt.figure()
    dn = scipy.cluster.hierarchy.dendrogram(mergings, leaf_font_size = 3, color_threshold = threshold)
#    plt.savefig(data_path + 'Dendrogram.png', dpi = 300, format = 'png', bbox_inches = 'tight')
    labels = scipy.cluster.hierarchy.fcluster(mergings, threshold, criterion = 'distance')
    
    groups = {}
    for i in range(max(labels)):
        groups[i] = set(np.where(labels == i + 1)[0])  # labels start at 1
    
    return groups

def postprocessGroups(groups, info_df):
    max_group = len(groups) - 1  # max(groups.keys())
    new_groups = {}
    for i, group in groups.items():
        group_df = info_df.loc[group].copy()
        group_df.sort_values(by = 'peakMaxTime', axis = 0, inplace = True)
        files = group_df['File']
        files_count = dict()
        new_groups[i] = set()
        max_group_increment = 0
        for peak, file in files.iteritems():
            if file in files_count:
                if max_group + files_count[file] not in new_groups:
                    new_groups[max_group + files_count[file]] = set()
                new_groups[max_group + files_count[file]].add(peak)
                if files_count[file] > max_group_increment:
                    max_group_increment += 1
                files_count[file] += 1
            else:
                files_count[file] = 1
                new_groups[i].add(peak)
                
        max_group += max_group_increment
    
    return new_groups
            

def alignTimes(groups, info_df, peak_intensity, align_to):
    info_df[align_to] = info_df['peakMaxTime']
    for group in groups.values():
        times = info_df.loc[group, 'peakMaxTime']
        peak_values = peak_intensity.loc[group]
        average_time = np.average(times, weights = peak_values)
#        average_time = np.mean(times)
        info_df.loc[group, align_to] = average_time
    

def printConfusionMatrix(prediction, info_df, comparisons):
    x1 = comparisons[:,0]
    x2 = comparisons[:,1]
    p = np.round(prediction).astype(int).reshape((-1))
    g1 = info_df.loc[x1]['Group'].values
    g2 = info_df.loc[x2]['Group'].values

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