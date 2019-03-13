import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from utils import loadData, plotSpectrumTogether, plotPeaksTogether, getRealGroupAssignments
from utils import getDistanceMatrix, assignGroups, alignTimes, calculateMetrics, postprocessGroups
from parameters import prediction_options, batch_prediction_options

data_path = prediction_options.get('data_path')
info_file = prediction_options.get('info_file')
sequence_file = prediction_options.get('sequence_file')
real_groups_available = prediction_options.get('real_groups_available')
prediction_file = prediction_options.get('predictions_save_name')
calculate_f1_metric = prediction_options.get('calculate_f1_metric')
calculate_metrics_for_components = prediction_options.get('calculate_metrics_for_components')

def plotAlignments(prediction, comparisons, info_df, peak_df_orig, peak_intensity, print_metrics = True):

    distance_matrix = getDistanceMatrix(comparisons, info_df.index.max() + 1, prediction, clip = 50, info_df = info_df)
    groups = assignGroups(distance_matrix, threshold = 2)
    groups = postprocessGroups(groups, info_df)
    alignTimes(groups, info_df, peak_intensity, 'AlignedTime')
    if real_groups_available:
        real_groups = getRealGroupAssignments(info_df)
        alignTimes(real_groups, info_df, peak_intensity, 'RealAlignedTime')
        if print_metrics:
            calculateMetrics(prediction, info_df, comparisons, calculate_for_components = calculate_metrics_for_components, calculate_f1 = calculate_f1_metric, print_metrics = True)

    plotSpectrumTogether(info_df, peak_intensity, with_real = real_groups_available, save_name = None)
#    plotPeaksTogether(info_df, peak_df_orig, with_real = real_groups_available, save_name = '../figures/alignment_plot')
    # if want to save the full spectra data (original, aligned and truth)
    plotPeaksTogether(info_df, peak_df_orig, with_real = real_groups_available, save_name = None, save_data = False)



def plotPeaksByIndex(index = None, margin = 100, plot_log_sequence = True, read_clipboard = False, plot_as_subplots = False):
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
        plt.show()
        
    mass_profile_df.loc[index].transpose().plot(ax = axes[0,1])
    if plot_as_subplots:
        axes[0,1].ticklabel_format(scilimits = (0,3))
        axes[0,1].set_title('Mass spectrum at the time of peak maximum', fontdict = {'fontsize': 18})
        axes[0,1].set_xlabel('m/z', fontdict = {'fontsize': 12})
    else:
        plt.title('Mass spectrum at the time of peak maximum')
        plt.show()
    
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
        plt.show()
    
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
            plt.show()



def plotPerformanceByModel(filename = None, use_false_pos_ignore_neg = True):
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


if __name__ == "__main__":
    info_df, peak_df, mass_profile_df, chromatogram_df, peak_df_orig, peak_intensity = loadData(data_path, info_file, sequence_file, take_chromatogram_log = False)
    
    try:
        prediction = pd.read_csv(prediction_file, usecols = [2]).values
        comparisons = pd.read_csv(prediction_file, usecols = [0,1]).values
    except FileNotFoundError:
        results_path = prediction_options.get('results_path')
        individual_predictions_save_path = batch_prediction_options.get('individual_predictions_save_path')
        prediction_file = os.path.join(results_path, individual_predictions_save_path, prediction_file.lstrip('results/') )
        prediction = pd.read_csv(prediction_file, usecols = [2]).values
        comparisons = pd.read_csv(prediction_file, usecols = [0,1]).values
    
    plotAlignments(prediction, comparisons, info_df, peak_df_orig, peak_intensity)
#    plotPeaksByIndex([2,10,17])
#    plotPerformanceByModel('results/ModelTests-OnField73.csv')
#    plotProbabilityMatrix(threshold = None, sort_by_group = True, highlight_negative_group = True)

