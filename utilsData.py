import pandas as pd
import numpy as np
import itertools
import os
from sklearn.metrics import roc_auc_score, average_precision_score

### Loading and preparing data
def loadData(data_path, info_file = 'PeakData-WithGroup.csv', sequence_file = 'WholeSequence.csv', take_chromatogram_log = True):
    """
    Loads data into pandas dataframes
    
    Arguments:
        data_path -- String: Path of data folder
        info_file -- String: Name of the file which will be loaded into info_df
        sequence_file -- String: Name of the file which contains the whole chromatogram sequences for all files
        take_chromatogram_log -- Boolean: whether the chromatogram_df is transformed by taking the log (base 2)
    
    Returns:
        info_df -- Dataframe of peak information. Number of rows = number of peaks
                  Columns are peak number, times (start, end and max value), file number, mass number and group number (if one is assigned)
                  Peak number is assigned from 0 within each chromatogram file
        peak_df -- Dataframe of the peak profile of each peak. Normalised to max of 1. Number of rows = number of peaks
                  Each peak is assumed to be in its own .txt file, as created by the 'ExtractPeaksAndMass.py' script
        mass_profile_df -- Dataframe of the mass spectrum associated with each peak's time of max value.
                  Each mass spectrum is assumed to be in its own .tsv file, as created by the 'ExtractPeaksAndMass.py' script
        chromatogram_df -- Dataframe of each file's whole chromatogram. Number of rows = number of files
        peak_df_orig -- Unnormalised peak_df
        peak_intensity -- Max chromatogram reading from each peak. Number of rows = number of peaks
    """
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
    """
    Generates a DataFrame of chromatogram segments

    Arguments:
        info_df -- DataFrame containing information about each peak, in particular the peak time and the file number
        chromatogram_df -- DataFrame containing the full chromatograms from all files. Column titles are the times of each measurement
        segment_length -- Int: total length of the resulting chromatogram segments
    
    Returns:
        chrom_seg_df -- DataFrame containing chromatogram segments centred at each peak, extracted from the corresponding unaligned chromatograms
    """
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


def generateCombinationIndices(info_df, time_cutoff = None, return_y = True, random_seed = None, ignore_same_sample = False):
    """
    Generates pairwise training examples

    Arguments:
        info_df -- DataFrame of peak information
        time_cutoff -- Int or Float: maximum time (min) between two pairwise peaks to still be considered a valid training/testing example
        return_y -- Boolean: To return y (for training) or not return y (for prediction)
        random_seed -- Int
        ignore_same_sample -- If True, removes comparisons between peaks originating from the same sample
        
    Returns:
        comparisons -- Numpy array. May be returned as two columns - x1 and x2
                      Contains the peak IDs of the two pairwise peaks. The peak ID is gives by the corresponding row index in info_df
        y -- Only returned if return_y is true. A Numpy array equal to 1 where the two peaks belong to the same group, or 0 otherwise
    """
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

    if ignore_same_sample:
        x1_file = info_df.loc[comparisons[:,0]]['File'].values
        x2_file = info_df.loc[comparisons[:,1]]['File'].values
        different_sample = x1_file != x2_file
        comparisons = comparisons[different_sample]

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
    """
    Identifies the group which has been manually assigned to each peak (ground truth)
    
    Arguments:
        info_df -- DataFrame containing information about each peak
    
    Returns:
        group -- Dictionary of group IDs, each containing a set of the peaks belonging to that group (peak ID given by the row index of the peak)
    """
    groups = {}
    for group, indexes in info_df.groupby('Group').groups.items():
        if group < 0: continue  # Don't align negative groups. Leave them with their original times
        groups[group] = set(indexes)
    return groups

    

### Measuring performance
def getGroundTruth(comparisons, info_df, ignore_negative_indices = True):
    x1 = comparisons[:,0]
    x2 = comparisons[:,1]
    g1 = info_df.loc[x1]['Group'].values
    g2 = info_df.loc[x2]['Group'].values
#    keep = (g1 >= 0) & (g2 >= 0)  # Ignore negative indices
    keep = (g1 >= 0) | (g2 >= 0)  # Ignores only when both are in the negative group
    truth = (g1 == g2)
    truth_ignore_neg = (g1[keep] == g2[keep])
    if ignore_negative_indices:
        return truth, truth_ignore_neg, keep
    else:
        return truth
    
def calculateMetrics(predictions, info_df, comparisons, calculate_f1 = True, calculate_auc = False,
                     calculate_average_precision = False, calculate_for_components = True, print_metrics = True):
    """
    Calculates a list of metrics to track the performance of the predictions from the network
    
    Arguments:
        predictions -- Numpy array or list of numpy arrays: Column vector(s) of probabilities (from output(s) of ChromAlignNet)
        info_df -- DataFrame containing information about each peak, in particular the assigned group number
        comparisons -- Numpy array with two columns - x1 and x2 - containing the IDs of the two peaks being compared
        calculate_f1 -- If True, calculates the recall, precision and F1 metric as part of the returned metrics
        calculate_auc -- If True, calculates the AUC as part of the returned metrics
        calculate_average_precision -- If True, calculates the average precision as part of the returned metrics
        calculate_for_components -- If True, calculates the metrics for each sub-network as well as the main output
                                    Needs the 'predictions' argument to be a list of numpy arrays containing all outputs from the network
        print_metrics -- If True, the metrics are printed to the console as well as returned
    
    Returns:
        metrics -- a list of metrics
    """
    metrics = []
    
    truth, truth_ignore_neg, keep = getGroundTruth(comparisons, info_df, ignore_negative_indices = True)
    
    if calculate_for_components:
        names = ['chromatogram', 'mass', 'main']
        if len(predictions) == 4:
            names[1:1] = ['peak']
    else:
        if not isinstance(predictions, list):
            predictions = [predictions]
        else:
            predictions = [predictions[0]]
            
    for prediction in predictions:
        p = np.round(prediction).astype(int).ravel()
        p_ignore_neg = p[keep]
    
        TP = np.mean(p_ignore_neg[truth_ignore_neg])
        FP_ignore_neg = np.mean(p_ignore_neg[~truth_ignore_neg])  # No longer valid
        FP = np.mean(p[~truth])  # This is used in the paper
        metrics.extend([TP, FP_ignore_neg, FP])
        
        if print_metrics:
            if calculate_for_components:
                print('\nMetrics from', names.pop(), 'output:')
            print('True positives: {} / {} = {:.3f}'.format(np.sum(p_ignore_neg[truth_ignore_neg]),
                  np.sum(truth_ignore_neg), np.mean(p_ignore_neg[truth_ignore_neg])))
            print('False positives - ignore negative idx: {} / {} = {:.3f}'.format(np.sum(p_ignore_neg[~truth_ignore_neg]),
                  np.sum(~truth_ignore_neg), np.mean(p_ignore_neg[~truth_ignore_neg])))
            print('False positives: {} / {} = {:.3f}'.format(np.sum(p[~truth]), np.sum(~truth), np.mean(p[~truth])))
    
        if calculate_f1:
            recall = TP
            precision = np.sum(p_ignore_neg[truth_ignore_neg]) / np.sum(p_ignore_neg)
            f1 = 2 * (precision * recall) / (precision + recall)
            metrics.extend([recall, precision, f1])
            if print_metrics:
                print('Recall: {:.3f}'.format(recall))
                print('Precision: {} / {} = {:.3f}'.format(np.sum(p_ignore_neg[truth_ignore_neg]), np.sum(p_ignore_neg), precision))
                print('F1: {:.3f}'.format(f1))
                
        if calculate_auc:
            roc_auc = roc_auc_score(truth_ignore_neg, prediction[keep])
            metrics.append(roc_auc)
#            metrics.append(roc_auc_score(truth, prediction))
            if print_metrics:
                print('AUC: {:.3f}'.format(roc_auc))
#                print('AUC not ignoring neg: {:.3f}'.format(roc_auc_score(truth, prediction)))
        if calculate_average_precision:
            average_precision = average_precision_score(truth_ignore_neg, prediction[keep])
            metrics.append(average_precision)
            if print_metrics:
                print('Average Precision: {:.3f}'.format(average_precision))
                
    return metrics


def getIncorrectExamples(prediction, info_df, comparisons, ignore_neg = True, sample_size = None):
    """
    Gets a sample of the incorrect predictions, returning the peak IDs of the incorrect pairs
    
    Arguments:
        predictions -- Numpy array or list of numpy arrays: Column vector(s) of probabilities (from output(s) of ChromAlignNet)
        info_df -- DataFrame containing information about each peak, in particular the assigned group number
        comparisons -- Numpy array with two columns - x1 and x2 - containing the IDs of the two peaks being compared
        ignore_neg --If True, peaks corresponding with a negative group will be ignored
        sample_size -- The number of incorrect examples to return, or None (which returns all incorrect examples)
    
    Returns:
        incorrect_array -- Numpy array of size (sample_size, 2), where the first column refers to
                           the peak ID of x1 and the second column refers to x2.
                           Each row gives the pair of peaks involved in an incorrect prediction
    """
    
    truth, truth_ignore_neg, keep = getGroundTruth(comparisons, info_df, ignore_negative_indices = True)
    
    p = np.round(prediction).astype(int).ravel()
    if ignore_neg:
        truth = truth_ignore_neg
        p = p[keep]
        comparisons = comparisons[keep]
    incorrect = p != truth
    incorrect_array = comparisons[incorrect][:sample_size]
    return incorrect_array



def printLastValues(history, std = None, kind = 'loss'):
    """
    Prints to the console the last values of the history
    
    Arguments:
        history -- DataFrame of the history over epoches of the model,
                   with the loss and accuracy of the various components as columns
        std -- None or a DataFrame giving the standard deviation of the history over epoches
        kind -- The component(s) of the history to print, as a string or a list of strings
    """
    def formatHistoryLabels(label):
        """
        Formats the labels of the different output contained within the model history
        
        Arguments:
            label -- The history label to format, as a string
        
        Returns:
            formatted_label -- The label formatted for printing to the console or for use in a plot legend
        """
        if label == 'loss':
            return 'Total Loss'
        label_components = label.split('_')
        output = []
        if label_components[0] == 'val':
            output.append('Validation')
        output.append(label_components[-3].capitalize())
        if label_components[-3] != 'main':
            output.append('Encoder')
        output.append('Loss' if label_components[-1] == 'loss' else 'Accuracy')
        
        formatted_label = ' '.join(output)
        return formatted_label
    
    print("Last Values:")
    if isinstance(kind, str):
        kind = [kind]
    formatted_labels = []
    for k in kind:
        end = ' ± {:.4f}'.format(std.iloc[-1][k]) if std is not None else ""
        formatted_labels.append(formatHistoryLabels(k))
        print('{}: {:.4f}'.format(formatted_labels[-1], history.iloc[-1][k]) + end)
    return formatted_labels