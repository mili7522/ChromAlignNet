import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from utils import loadData, plotSpectrumTogether, plotPeaksTogether, getRealGroupAssignments
from utils import getDistanceMatrix, assignGroups, alignTimes, calculateMetrics, postprocessGroups
from parameters import prediction_options

data_path = prediction_options.get('data_path')
info_file = prediction_options.get('info_file')
sequence_file = prediction_options.get('sequence_file')
real_groups_available = prediction_options.get('real_groups_available')
prediction_file = prediction_options.get('predictions_save_name')
calculate_f1_metric = prediction_options.get('calculate_f1_metric')
ignore_same_sample = prediction_options.get('ignore_same_sample')

def plotAlignments(prediction, comparisons, info_df, peak_df_orig, peak_intensity, print_metrics = True, ignore_same_sample = False):
    """
    Performs alignment and plots the outcome
    Given the output from ChromAlignNet, which corresponds to pairwise probabilities of alignment between peaks,
    groups of formed such that the peaks in each group will be aligned together.
    The aligned spectrum and peaks are then plotted in comparison with the raw data and ground truth alignment (if available)
    
    Arguments:
        prediction -- Numpy array giving a column vector of probabilities (from the main output of ChromAlignNet)
        comparisons -- Numpy array with two columns - x1 and x2 - containing the IDs of the two peaks being compared
        info_df -- DataFrame containing information about each peak
        peak_df_orig -- DataFrame containing the unnormalised intensities along the profile of each peak
        peak_intensity -- DataFrame containing the maximum intensity of each peak
        print_metrics -- If True, prints metrics related to the prediction outcome (TP, FP, F1, etc)
        ignore_same_sample -- If True, removes comparisons between peaks originating from the same sample
    """
    if ignore_same_sample:
        x1_file = info_df.loc[comparisons[:,0]]['File'].values
        x2_file = info_df.loc[comparisons[:,1]]['File'].values
        different_sample = x1_file != x2_file
        comparisons = comparisons[different_sample]
        prediction = prediction[different_sample]
    
    # getDistanceMatrix calculates a matrix of distances between each peak using the pairwise probabilities
    distance_matrix, probability_matrix = getDistanceMatrix(comparisons, info_df.index.max() + 1, prediction, clip = 50, info_df = info_df)
    # assignGroups uses a hierachical clustering algorithm to assign groups
    groups = assignGroups(distance_matrix, threshold = 2)
    # Because of common patterns of false positives, postprocessing separates out peaks which belong to the same sample into different groups
    groups = postprocessGroups(groups, info_df)
    
    alignTimes(groups, info_df, peak_intensity, 'AlignedTime')
    if real_groups_available:
        real_groups = getRealGroupAssignments(info_df)
        alignTimes(real_groups, info_df, peak_intensity, 'RealAlignedTime')
        if print_metrics:
            calculateMetrics(prediction, info_df, comparisons, calculate_for_components = False, calculate_f1 = calculate_f1_metric, print_metrics = True)

    plotSpectrumTogether(info_df, peak_intensity, with_real = real_groups_available, save_name = None)
#    plotPeaksTogether(info_df, peak_df_orig, with_real = real_groups_available, save_name = '../figures/alignment_plot')
    # if want to save the full spectra data (original, aligned and truth)
    plotPeaksTogether(info_df, peak_df_orig, with_real = real_groups_available, save_name = None, save_data = False)



def plotPeaksByIndex(index = None, margin = 100, plot_log_sequence = True, read_clipboard = False, plot_as_subplots = False):
    """
    Plots several views of one or more peaks
    Views of the peak profiles, mass spectra and chromatogram segments are shown
    
    Arguments:
        index -- None or a list of IDs of the peaks to be plotted (peak IDs correspond to the index of the info_df DataFrame)
                 If index is None then input is given from the console and are added successively to a list until a blank input is given
        margin -- The number of time steps to either side of the average retention time to plot in the chromatogram segment figure
        plot_log_sequence -- If True, produces an additional figure of the chromatogram segment on a semi-log plot
        read_clipboard -- If True, the clipboard is read to get the list of peak ID values
        plot_as_subplots -- If True, produces subplots in one figure instead of separate figures
    """
    if plot_as_subplots:
        fig, axes = plt.subplots(2,2)
    else:
        axes = np.array([[None] * 2, [plt] * 2], dtype=np.object)
    
    if index is None:
        if read_clipboard:
            index = pd.read_clipboard(header = None).squeeze().tolist()
        else:
            index = []
            while True:
                i = input("Index:")
                if i == '': break
                else: index.append(int(i))
    print(info_df.loc[index])
    peak_df_orig.loc[index].transpose().plot(ax = axes[0,0])
    if plot_as_subplots:
        axes[0,0].ticklabel_format(scilimits = (0,3))
        axes[0,0].set_title('Peak profile', fontdict = {'fontsize': 18})
    else:
        plt.title('Peak profile')
        
    mass_profile_df.loc[index].transpose().plot(ax = axes[0,1])
    if plot_as_subplots:
        axes[0,1].ticklabel_format(scilimits = (0,3))
        axes[0,1].set_title('Mass spectrum at the time of peak maximum', fontdict = {'fontsize': 18})
        axes[0,1].set_xlabel('m/z', fontdict = {'fontsize': 12})
    else:
        plt.title('Mass spectrum at the time of peak maximum')
        plt.figure()
    
    chrom_idx = np.argmin(np.abs(chromatogram_df.columns - np.mean(info_df.loc[index]['peakMaxTime'])).values)
    axes[1,0].plot(chromatogram_df.iloc[:, max(0,chrom_idx - margin) : chrom_idx + margin].transpose(), 'gray', alpha = 0.2, label = '_nolegend_')
    for i, file in enumerate(info_df.loc[index]['File']):
        p = axes[1,0].plot(chromatogram_df.iloc[file, max(0,chrom_idx - margin) : chrom_idx + margin].transpose(), linewidth=3, label = index[i])
        # Plot line to the top of the peak at 'peakMaxTime'. Helps keep track of which peak to look at
        axes[1,0].plot((info_df.loc[index[i]]['peakMaxTime'], info_df.loc[index[i]]['peakMaxTime']),
                  (0, max(peak_df_orig.loc[index[i]])), color = p[-1].get_color(), label = '_nolegend_')
    axes[1,0].legend()
    axes[1,0].ticklabel_format(scilimits = (0,3))
    if plot_as_subplots:
        axes[1,0].set_title('Chromatogram segment', fontdict = {'fontsize': 18})
        axes[1,0].set_xlabel('Retention Time (min)', fontdict = {'fontsize': 12})
    else:
        plt.title('Chromatogram segment')
        plt.figure()
    
    if plot_log_sequence:
        axes[1,1].plot(chromatogram_df.iloc[:, max(0,chrom_idx - margin) : chrom_idx + margin].transpose(), 'gray', alpha = 0.2, label = '_nolegend_')
        for i, file in enumerate(info_df.loc[index]['File']):
            segment = chromatogram_df.iloc[file, max(0,chrom_idx - margin) : chrom_idx + margin].transpose()
            segment = segment[segment != 0]
            p = axes[1,1].semilogy(segment, linewidth=3, label = index[i])
            # Plot line to the top of the peak at 'peakMaxTime'. Helps keep track of which peak to look at
            axes[1,1].semilogy((info_df.loc[index[i]]['peakMaxTime'], info_df.loc[index[i]]['peakMaxTime']),
                      (np.min(segment), max(peak_df_orig.loc[index[i]])), color = p[-1].get_color(), label = '_nolegend_')
        axes[1,1].legend()
        if plot_as_subplots:
            axes[1,1].set_title('Chromatogram segment - log scale', fontdict = {'fontsize': 18})
            axes[1,1].set_xlabel('Retention Time (min)', fontdict = {'fontsize': 12})
            plt.show()
        else:
            plt.title('Chromatogram segment - log scale')



def plotPerformanceByModel(filename = None, use_false_pos_ignore_neg = True):
    """
    Plots a bar chart comparing the true positives and false positives of a variety of models
    Uses the output from the batch prediction script
    
    Arguments:
        filename -- Name of a csv file created by the batch prediction script, defined as a string
        use_false_pos_ignore_neg -- If True, the 'False Positives - Ignore Neg Idx' column is used in place of the 'False Positives' column
    """
    # TODO: Better loading of filenames
    df = pd.read_csv(filename, index_col = 0)

    # Select appropriate false positives column
    if use_false_pos_ignore_neg:
        del df['False Positives']
        df = df.rename(columns = {'False Positives - Ignore Neg Idx': 'False Positives'})

    g = df[['True Positives', 'False Positives', 'Model Name']].groupby('Model Name')
    mean = g.mean()
    std = g.std()

    ax = plt.axes()
    ax.set_aspect(10)

    p = ax.plot((-1,19), [mean.iloc[0]['True Positives']] * 2, '--', alpha = 1, linewidth = 1, zorder = -1)
    ax.plot((-1,19), [mean.iloc[0]['True Positives'] + std.iloc[0]['True Positives']] * 2, ':', alpha = 1, linewidth = 1, zorder = -1, color = p[-1].get_color())
    ax.plot((-1,19), [mean.iloc[0]['True Positives'] - std.iloc[0]['True Positives']] * 2, ':', alpha = 1, linewidth = 1, zorder = -1, color = p[-1].get_color())
    p = ax.plot((-1,19), [mean.iloc[0]['False Positives']] * 2, '--', alpha = 1, linewidth = 1, zorder = -1)
    ax.plot((-1,19), [mean.iloc[0]['False Positives'] + std.iloc[0]['False Positives']] * 2, ':', alpha = 1, linewidth = 1, zorder = -1, color = p[-1].get_color())
    ax.plot((-1,19), [mean.iloc[0]['False Positives'] - std.iloc[0]['False Positives']] * 2, ':', alpha = 1, linewidth = 1, zorder = -1, color = p[-1].get_color())

    mean.plot.bar(width = 0.75, ax = ax, zorder = 1,
        yerr = std.values.T, error_kw = {'elinewidth': 0.5, 'capthick': 0.5, 'ecolor': 'black', 'capsize': 1})
    plt.ylim(0, 1)
    plt.legend(loc='lower left', fontsize = 'small', framealpha = 1, frameon = True)
    plt.xlabel('Model Name')

    plt.show()


def plotProbabilityMatrix(threshold = None, sort_by_group = True, highlight_negative_group = True):
    """
    Plots a heatmap representation of the 2D probability matrix arranged by peak time (and ground truth group if 'sort_by_group' = True)
    
    Arguments:
        threshold -- None or a Float between 0 and 1.
                     If it is a float, the plot shows the outcome of a binary classifier, with the decision threshold set at this values
        sort_by_group -- If True, the peaks will be sorted by ascending ground truth groups as well as peak retention time.
                         Red lines will be drawn to show the boundary between different groups
                         If False, sorting will only occur by peak retention time. No red lines will be drawn
        highlight_negative_group -- If True, peaks belonging to groups with negative group ID will be highlighted
                                    by setting their values in the lower triangle of the matrix to 0
    """
    if sort_by_group:
        #Assign unique numbers to the -1 group peaks
        max_group = info_df['Group'].max()
        neg_groups = info_df['Group'] < 0
        info_df['New_Group'] = info_df['Group']
        info_df.loc[neg_groups, 'New_Group'] = range(max_group + 1, max_group + 1 + neg_groups.sum())
        
        sorted_groups = info_df.groupby('New_Group').mean().sort_values('peakMaxTime').index  # Gives the ordering of the groups sorted by average peak RT
        sorted_groups_dict = dict(zip(sorted_groups, range(len(sorted_groups))))  # Mapping between the old group ID and new group ID based on the sort order
        idx_arrangement = pd.concat([info_df['New_Group'].map(sorted_groups_dict), info_df['peakMaxTime']], axis = 1).sort_values(by = ['New_Group', 'peakMaxTime']).index # Sort by Group, then peakMaxTime
    else:
        idx_arrangement = info_df.sort_values('peakMaxTime').index
    idx_arrangement_dict = dict(zip(idx_arrangement, range(len(idx_arrangement)) ))
    
    number_of_peaks = len(info_df)
    
    prediction_matrix = np.zeros((number_of_peaks, number_of_peaks))
    np.fill_diagonal(prediction_matrix, 1)
    
    for i, (x1, x2) in enumerate(comparisons):
        x1_ = idx_arrangement_dict[x1]
        x2_ = idx_arrangement_dict[x2]
        prediction_matrix[x1_, x2_] = prediction_matrix[x2_, x1_] = prediction[i]
        if highlight_negative_group and info_df.loc[x1, 'Group'] < 0 and info_df.loc[x2, 'Group'] < 0:  # Highlight peaks with a negative group by not plotting below the diagonal when both peaks are negative
            if x1_ < x2_:
                prediction_matrix[x2_, x1_] = 0
            else:
                prediction_matrix[x1_, x2_] = 0

    if threshold is not None:
        prediction_matrix = prediction_matrix > threshold
    
    max_time = info_df['peakMaxTime'].max()
    min_time = info_df['peakMaxTime'].min()
    plt.imshow(prediction_matrix, extent = None if sort_by_group else (min_time, max_time, max_time, min_time))
    
    dataset_name = prediction_file.split('_')[-2]
    if threshold is not None:
        plt.title('{}\nProbability Threshold: {}'.format(dataset_name, threshold))
    else:
        plt.title('{}'.format(dataset_name))
        plt.colorbar(label = 'Probability')
        
    if sort_by_group:
        new_idx_groups = info_df.loc[idx_arrangement, 'Group'].values
        group_change_idx = np.flatnonzero( np.diff(new_idx_groups) )  # Gives last idx before (non-negative) group change, zero indexed
        for x in group_change_idx:
            plt.axvline(x=x+0.5, c = 'red', linewidth = 1, alpha = 1)
            plt.axhline(y=x+0.5, c = 'red', linewidth = 1, alpha = 1)
        plt.xlabel('Index')
        plt.ylabel('Index')
    else:
        plt.xlabel('Min')
        plt.ylabel('Min')



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


def plotHistory(kind = 'loss', all_reps = False):
    """
    Plots the history over epoches of the model
    
    Arguments:
        kind -- The component(s) of the history to plot, defined as a string or a list of strings
                If a list of strings is used, multiple component will be drawn within the same figure
                The available components are:
                    loss
                    main_prediction_loss, mass_prediction_loss, peak_prediction_loss, chromatogram_prediction_loss,
                    val_main_prediction_loss, val_mass_prediction_loss, val_peak_prediction_loss, val_chromatogram_prediction_loss,
                    main_prediction_acc, mass_prediction_acc, peak_prediction_acc, chromatogram_prediction_acc,
                    val_main_prediction_acc, val_mass_prediction_acc, val_peak_prediction_acc, val_chromatogram_prediction_acc
        all_reps -- If True, plots the mean and standard deviation across all (10) repetitions of the model
    """
    if kind == 'acc':
        kind = ['main_prediction_acc', 'val_main_prediction_acc']
    if isinstance(kind, str):
        kind = [kind]
    if all_reps:
        histories = []
        for rep in range(1,11):
            f = '{}/{}-r{:02}-History.csv'.format(prediction_options['model_path'], prediction_options['model_file'][:-4], rep)
            df = pd.read_csv(f, index_col = 0)
            histories.append(df)
        df = pd.concat(histories, axis = 0)
        g = df.groupby(df.index)
        history = g.mean()
        std = g.std()
        formatted_labels = printLastValues(history, std, kind = kind)
        error_kw = {'elinewidth': 1, 'capthick': 1, 'capsize': 2, 'errorevery': len(history) // 50}
    else:
        history_file = '{}/{}-History.csv'.format(prediction_options['model_path'], prediction_options['model_file'])
        history = pd.read_csv(history_file, index_col = 0)
        formatted_labels = printLastValues(history, None, kind = kind)
        error_kw = {}
    history[kind].plot(linewidth = 2, yerr = std[kind] if all_reps else None, **error_kw)
    plt.legend(formatted_labels)
    plt.xlabel('Epochs')


def plotSubnetworkHistory(kind = 'acc', all_reps = False):
    """
    Plots the history of each sub-network using the plotHistory function
    
    Arguments:
        kind -- which component of the history to plot, defined as a string ('acc', 'loss', 'val_acc' or 'val_loss')
        all_reps -- If True, plots the mean and standard deviation across all (10) repetitions of the model
    """
    if kind.endswith('acc'):
        k = ['main_prediction_acc', 'mass_prediction_acc', 'peak_prediction_acc', 'chromatogram_prediction_acc']
    if kind.endswith('loss'):
        k = ['main_prediction_loss', 'mass_prediction_loss', 'peak_prediction_loss', 'chromatogram_prediction_loss']
    if kind.startswith('val'):
        k = ['val_' + i for i in k]
    plotHistory(k, all_reps)
    plt.ylabel('Accuracy' if kind.endswith('acc') else 'Loss')


if __name__ == "__main__":
    info_df, peak_df, mass_profile_df, chromatogram_df, peak_df_orig, peak_intensity = loadData(data_path, info_file, sequence_file, take_chromatogram_log = False)
    
#    try:
#        prediction = pd.read_csv(prediction_file, usecols = [2]).values
#        comparisons = pd.read_csv(prediction_file, usecols = [0,1]).values
#    except FileNotFoundError:
#        results_path = prediction_options.get('results_path')
#        individual_predictions_save_path = batch_prediction_options.get('individual_predictions_save_path')
#        prediction_file = os.path.join(results_path, individual_predictions_save_path, prediction_file.lstrip('results/') )  # TODO: Improve?
#        prediction = pd.read_csv(prediction_file, usecols = [2]).values
#        comparisons = pd.read_csv(prediction_file, usecols = [0,1]).values
    
    
#    plotAlignments(prediction, comparisons, info_df, peak_df_orig, peak_intensity)
#    plotPeaksByIndex([2], margin = 100, plot_log_sequence = True, read_clipboard = False, plot_as_subplots = False)
#    plotPerformanceByModel('results/ModelTests-OnField73.csv')
#    plotProbabilityMatrix(threshold = None, sort_by_group = True, highlight_negative_group = True)
#    plotHistory('loss', True)
    plotSubnetworkHistory('val_acc', False)

