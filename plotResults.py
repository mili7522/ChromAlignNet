import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from utils import loadData, plotSpectrumTogether, plotPeaksTogether, getRealGroupAssignments
from utils import getDistanceMatrix, assignGroups, alignTimes, printConfusionMatrix, postprocessGroups
from parameters import prediction_options

data_path = prediction_options.get('data_path')
info_file = prediction_options.get('info_file')
sequence_file = prediction_options.get('sequence_file')
real_groups_available = prediction_options.get('real_groups_available')

info_df, peak_df, mass_profile_df, chromatogram_df, peak_df_orig, peak_intensity = loadData(data_path, info_file, sequence_file, take_chromatogram_log = False)

def plotAlignments():
    prediction_file = prediction_options.get('predictions_save_name')
    prediction = pd.read_csv(prediction_file, usecols = [2]).values
    comparisons = pd.read_csv(prediction_file, usecols = [0,1]).values

    distance_matrix = getDistanceMatrix(comparisons, info_df.index.max() + 1, prediction, clip = 50, info_df = info_df)
    groups = assignGroups(distance_matrix, threshold = 2)
    groups = postprocessGroups(groups, info_df)
    alignTimes(groups, info_df, peak_intensity, 'AlignedTime')
    if real_groups_available:
        real_groups = getRealGroupAssignments(info_df)
        alignTimes(real_groups, info_df, peak_intensity, 'RealAlignedTime')
        printConfusionMatrix(prediction, info_df, comparisons)

    plotSpectrumTogether(info_df, peak_intensity, with_real = real_groups_available, save_name = '../figures/alignment_image')
#    plotPeaksTogether(info_df, peak_df_orig, with_real = real_groups_available, save_name = '../figures/alignment_plot')
    # if want to save the full spectra data (original, aligned and truth)
    plotPeaksTogether(info_df, peak_df_orig, with_real = real_groups_available, save_name = '../figures/alignment_plot', save_data = True)



def plotByIndex(index = None, margin = 100, plot_log_sequence = True, read_clipboard = False, plot_as_subplots = False):
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
        df = df.rename(columns = {'False Positives - Ignore Neg Indices': 'False Positives'})

    g = df[['True Positives', 'False Positives', 'Model Number']].groupby('Model Number')
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

    mean.index = ['3.{:02d}'.format(x) for x in mean.index]
    mean.plot.bar(width = 0.75, ax = ax, zorder = 1,
        yerr = std.values.T, error_kw = {'elinewidth': 0.5, 'capthick': 0.5, 'ecolor': 'black', 'capsize': 1})
    plt.ylim(0, 1)
    plt.legend(loc='lower left', fontsize = 'small', framealpha = 1, frameon = True)
    plt.xlabel('Model Number')

    plt.show()


if __name__ == "__main__":
#    plotAlignments()
#    plotByIndex([2,10,17])
#    plotPerformanceByModel('results/ModelTests-OnBreath88.csv')

    #Assign unique numbers to the -1 group peaks
    max_group = info_df['Group'].max()
    neg_groups = info_df['Group'] < 0
    info_df['New_Group'] = info_df['Group']
    info_df.loc[neg_groups, 'New_Group'] = range(max_group + 1, max_group + 1 + neg_groups.sum())
    
    sorted_groups = info_df.groupby('New_Group').mean().sort_values('peakMaxTime').index
    sorted_groups_dict = dict(zip(sorted_groups, range(len(sorted_groups))))
#    idx_arrangement = info_df['Group'].map(sorted_groups_dict).argsort()
#    idx_arrangement_dict = dict(zip(idx_arrangement, idx_arrangement.index))
    idx_arrangement = pd.concat([info_df['New_Group'].map(sorted_groups_dict), info_df['peakMaxTime']], axis = 1).sort_values(by = ['New_Group', 'peakMaxTime']).index # Sort by both Group and peakMaxTime
    idx_arrangement_dict = dict(zip(idx_arrangement, range(len(idx_arrangement)) ))
    
    prediction_file = prediction_options.get('predictions_save_name')
    prediction = pd.read_csv(prediction_file, usecols = [2]).values
    comparisons = pd.read_csv(prediction_file, usecols = [0,1]).values
    
    number_of_peaks = comparisons.max() + 1
    
    prediction_matrix = np.zeros((number_of_peaks, number_of_peaks))
    
    for i, (x1, x2) in enumerate(comparisons):
        x1_ = idx_arrangement_dict[x1]
        x2_ = idx_arrangement_dict[x2]
        prediction_matrix[x1_, x2_] = prediction_matrix[x2_, x1_] = prediction[i]
        if info_df.loc[x1, 'Group'] < 0 and info_df.loc[x2, 'Group'] < 0:
            if x1_ < x2_:
                prediction_matrix[x2_, x1_] = 0
            else:
                prediction_matrix[x1_, x2_] = 0

#    group_matrix = np.zeros((number_of_peaks, number_of_peaks))        
#    for i in range(number_of_peaks):
#        i_ = idx_arrangement_dict[i]
#        group_matrix[:, i_] = sorted_groups_dict[info_df.loc[i, 'Group']]

    np.fill_diagonal(prediction_matrix, 1)
    

    new_idx_groups = info_df.loc[idx_arrangement, 'Group'].values
    group_change_idx = np.flatnonzero( np.diff(new_idx_groups) )  # Gives last idx before group change, zero indexed
    # Remove marking negative indices
    
    import matplotlib.pyplot as plt
    
#    prediction_matrix = prediction_matrix > 0.5
    
    plt.imshow(prediction_matrix); plt.colorbar(label = 'Probability')
    for x in group_change_idx:
        plt.axvline(x=x+0.5, c = 'red', linewidth = 1, alpha = 1)
        plt.axhline(y=x+0.5, c = 'red', linewidth = 1, alpha = 1)
#    plt.title('Breath115')