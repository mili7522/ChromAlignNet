import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from utilsData import loadData, getRealGroupAssignments, getIncorrectExamples, calculateMetrics, printLastValues, getGroundTruth
from utilsPlotting import plotSpectrumTogether, plotPeaksTogether, plotPeaksByIndex
from utilsAlignment import getDistanceMatrix, assignGroups, alignTimes, postprocessGroups
from parameters import prediction_options, batch_prediction_options


def loadPredictions(prediction_file):
    """
    Loads saved prediction output from the model with corresponding comparison file
    Takes into account batch prediction that might have saved individual prediction output into a subfolder
    
    Arguments:
        prediction_file -- Path and filename of the saved location file, defined as a string
        
    Outputs:
        prediction -- Numpy array giving a column vector of probabilities (from the main output of ChromAlignNet)
        comparisons -- Numpy array with two columns - x1 and x2 - containing the IDs of the two peaks being compared
    """
    try:
        prediction = pd.read_csv(prediction_file, usecols = [2]).values
        comparisons = pd.read_csv(prediction_file, usecols = [0,1]).values
    except FileNotFoundError as e:
        try:
            if batch_prediction_options['save_individual_predictions']:
                results_path = prediction_options['results_path']
                individual_predictions_save_path = batch_prediction_options['individual_predictions_save_path']
                if individual_predictions_save_path is not None:
                    prediction_file = prediction_file.replace(results_path.rstrip('/\\'), os.path.join(results_path, individual_predictions_save_path))
            prediction = pd.read_csv(prediction_file, usecols = [2]).values
            comparisons = pd.read_csv(prediction_file, usecols = [0,1]).values
        except:
            raise e
    print("File loaded:", prediction_file)

    return prediction, comparisons


def plotAlignments(prediction, comparisons, info_df, peak_df_orig, peak_intensity, print_metrics = True):
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
    """
    
    if prediction_options['ignore_same_sample']:
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
    real_groups_available = 'Group' in info_df
    if real_groups_available:
        real_groups = getRealGroupAssignments(info_df)
        alignTimes(real_groups, info_df, peak_intensity, 'RealAlignedTime')
        if print_metrics:
            calculateMetrics(prediction, info_df, comparisons, calculate_for_components = False,
                             calculate_f1 = prediction_options['calculate_f1_metric'],
                             calculate_auc = prediction_options['calculate_auc_metric'],
                             print_metrics = True)

    plotSpectrumTogether(info_df, peak_intensity, with_real = real_groups_available, save_name = None)
#    plotPeaksTogether(info_df, peak_df_orig, with_real = real_groups_available, save_name = '../figures/alignment_plot')
    # if want to save the full spectra data (original, aligned and truth)
    plotPeaksTogether(info_df, peak_df_orig, with_real = real_groups_available, save_name = None, save_data = False)


def plotMetricsByModel(dataset_name = None, use_false_pos_ignore_neg = True, metrics = ['True Positives', 'False Positives'],
                       use_range = False, plot_aspect = 6):
    """
    Plots a bar chart comparing the performance (multiple metrics) of a variety of models on one data set
    Uses the output from the batch prediction script
    
    Arguments:
        dataset_name -- Name of the data set the models are being compared on, as a string, or None
                        If None, the name given in prediction_options is used as the default
        use_false_pos_ignore_neg -- If True, the 'False Positives - Ignore Neg Idx' column is used in place of the 'False Positives' column
        metrics -- 
        use_range -- If True, the range is plotted as asymmetric error bars instead of the standard deviation
        plot_aspect -- None, or a number. Defines the aspect ratio of the plot. If None, the default is used.
    """
    dataset_name = dataset_name or prediction_options['dataset_name']  # Use the name given in prediction_options as the default
    filename = os.path.join(prediction_options['results_path'], 'ModelTests-On{}.csv'.format(dataset_name))
    df = pd.read_csv(filename, index_col = 0)

    if isinstance(metrics, str):
        metrics = [metrics]

    # Select appropriate false positives column
    if use_false_pos_ignore_neg and 'False Positives' in metrics:
        del df['False Positives']
        df = df.rename(columns = {'False Positives - Ignore Neg Idx': 'False Positives'})

    if 'TP - FP' in metrics:  # Calculate the difference between the true positives and false positives
        df['TP - FP'] = df['True Positives'] - df['False Positives']

    plotColumns(df, metrics, use_range, reference_model = 0, plot_aspect = plot_aspect)


def plotPerformanceOnDatasetsByModel(datasets = None, metric = 'True Positives', use_range = False,
                                     plot_aspect = None, reference_model = None):
    """
    Plots a bar chart comparing one performance metric of a variety of models on multiple data sets
    Uses the outputs from the batch prediction script
    
    Arguments:
        datasets -- None, or a list of Ints corresponding to the data sets to plot
        metric -- Name of the metric to plot, as a string
    """
    datasets = datasets or range(len(batch_prediction_options['dataset_name']))

    f = batch_prediction_options['save_names']
    names = batch_prediction_options['dataset_name']
    
    dfs = []
    for i in datasets:
        df = pd.read_csv(os.path.join(prediction_options['results_path'], f[i]), index_col = 0)
        df = df[[metric] + ['Model Name']].rename(columns = {metric: names[i]}).set_index('Model Name')
        dfs.append(df)
    df = pd.concat(dfs, axis = 1)
    df.reset_index(inplace = True)
    
    names = [names[i] for i in datasets]
    plotColumns(df, names, use_range, reference_model = reference_model, plot_aspect = plot_aspect)
    

def plotColumns(df, columns, use_range = False, reference_model = 0, plot_aspect = None):
    """
    Plots a bar chart showing the mean and standard deviation (or range) of each
    columns of the batch prediction results. Bars are grouped by the different model names
    Helper function for plotMetricsByModel and plotPerformanceOnDatasetsByModel
    
    Arguments:
        df -- pandas DataFrame 
        columns -- List of strings. Each element is a column name of the DataFrame,
                   and will be plotted as a bar for each of the model names
        use_range  -- If True, the range is plotted as asymmetric error bars instead of the standard deviation
        reference_model
        plot_aspect
    """
    g = df[columns + ['Model Name']].groupby('Model Name')
    mean = g.mean()
    
    if use_range:
        lower = np.expand_dims((mean - g.min()).values.T, axis = 1)
        upper = np.expand_dims((g.max() - mean).values.T, axis = 1)
        std = np.concatenate((lower, upper), axis = 1)
    else:
        std = np.expand_dims(g.std().values.T, axis = 1)

    ax = plt.axes()
    if plot_aspect is not None:
        ax.set_aspect(plot_aspect)

    # Plot horizontal lines highlighting the mean and standard deviation (or range) of the reference model (usually should be the first model, for comparison)
    if reference_model is not None:
        j = reference_model
        for i, column in enumerate(columns):
            p = ax.plot((-1,19), [mean.iloc[j][column]] * 2, '--', alpha = 1, linewidth = 1, zorder = -1)
            ax.plot((-1,19), [mean.iloc[j][column] + std[i][-1][j]] * 2, ':', alpha = 1, linewidth = 1, zorder = -1, color = p[-1].get_color())
            ax.plot((-1,19), [mean.iloc[j][column] - std[i][0][j]] * 2, ':', alpha = 1, linewidth = 1, zorder = -1, color = p[-1].get_color())

    # Plot the performance of each model as a bar chart with error bars
    mean.plot.bar(width = 0.75, ax = ax, zorder = 1,
        yerr = std.squeeze(), error_kw = {'elinewidth': 0.5, 'capthick': 1, 'ecolor': 'black', 'capsize': 2})
    plt.ylim(0, 1)
    plt.legend(loc='lower left', fontsize = 'small', framealpha = 1, frameon = True)
    
    plt.tight_layout()
    plt.show()



def plotProbabilityMatrix(prediction, comparisons, info_df, threshold = None, sort_by_group = True, highlight_negative_group = True):
    """
    Plots a heatmap representation of the 2D probability matrix with the order of the peaks
    arranged by their RT (and also the ground truth group if 'sort_by_group' = True)
    
    Arguments:
        prediction -- Numpy array giving a column vector of probabilities (from the main output of ChromAlignNet)
        comparisons -- Numpy array with two columns - x1 and x2 - containing the IDs of the two peaks being compared
        info_df -- DataFrame containing information about each peak, in particular the peak times and their assigned ground truth group
        threshold -- None or a Float between 0 and 1.
                     If it is a float, the plot shows the outcome of a binary classifier, with the decision threshold set at this value
        sort_by_group -- If True, the peaks will be sorted by ascending ground truth groups as well as peak retention time.
                         Red lines will be drawn to show the boundary between different groups
                         If False, sorting will only occur by peak retention time. No red lines will be drawn
        highlight_negative_group -- If True, peaks belonging to groups with negative group ID will be highlighted
                                    by setting their values in the lower triangle of the probability matrix to 0
    """
    if sort_by_group:
        # Assign unique group numbers to the -1 group peaks, so that they can be sorted independently
        max_group = info_df['Group'].max()
        neg_groups = info_df['Group'] < 0
        info_df['New_Group'] = info_df['Group']
        info_df.loc[neg_groups, 'New_Group'] = range(max_group + 1, max_group + 1 + neg_groups.sum())
        
        sorted_groups = info_df.groupby('New_Group').mean().sort_values('peakMaxTime').index  # This gives the order of the groups when sorted by average peak RT
        sorted_groups_dict = dict(zip(sorted_groups, range(len(sorted_groups))))  # Mapping between the old group ID and the new group ID based on the sort order
        tmp_df = pd.concat([info_df['New_Group'].map(sorted_groups_dict), info_df['peakMaxTime']], axis = 1)
        idx_arrangement = tmp_df.sort_values(by = ['New_Group', 'peakMaxTime']).index  # Sort first by Group, then by peakMaxTime
    else:
        idx_arrangement = info_df.sort_values('peakMaxTime').index
    # This gives a new ordering for the peaks by mapping from the previous peak ID
    # (the index of the info_df) to the new ID based on the sort order
    idx_arrangement_dict = dict( zip( idx_arrangement, range(len(idx_arrangement)) ) )
    
    # Create an empty matrix to store the pairwise probabilities of each peak matching each other peak
    number_of_peaks = len(info_df)
    probability_matrix = np.zeros((number_of_peaks, number_of_peaks))
    np.fill_diagonal(probability_matrix, 1)  # Diagonals are assigned a probability of 100%
    
    # Fill the probability matrix using the prediction output from ChromAlignNet
    for i, (x1, x2) in enumerate(comparisons):
        x1_ = idx_arrangement_dict[x1]  # Get the new peak ID from the old
        x2_ = idx_arrangement_dict[x2]
        probability_matrix[x1_, x2_] = probability_matrix[x2_, x1_] = prediction[i]
        # If highlight_negative_group is True, combinations where both peaks are from negative groups
        # are only plotted in the upper diagonal (the value in the lower diagonal is set to 0)
        if highlight_negative_group and info_df.loc[x1, 'Group'] < 0 and info_df.loc[x2, 'Group'] < 0:
            if x1_ < x2_:
                probability_matrix[x2_, x1_] = 0
            else:
                probability_matrix[x1_, x2_] = 0

    if threshold is not None:
        probability_matrix = probability_matrix > threshold
    
    # Plot the probability matrix as a heatmap
    plt.imshow(probability_matrix)
    plt.xlabel('Index')
    plt.ylabel('Index')
    
    dataset_name = prediction_options['dataset_name']
    if threshold is not None:
        plt.title('{}\nProbability Threshold: {}'.format(dataset_name, threshold))
    else:
        plt.title('{}'.format(dataset_name))
        plt.colorbar(label = 'Probability')
        
    if sort_by_group:  # Split each group with red lines
        new_idx_groups = info_df.loc[idx_arrangement, 'Group'].values
        group_change_idx = np.flatnonzero( np.diff(new_idx_groups) )  # This gives the last idx values
        # before group change (zero indexed). 'Group' instead of 'New_Group' is used so that consecutive peaks
        # from negative groups are treated as part of the same block
        for x in group_change_idx:
            plt.axvline(x = x + 0.5, c = 'red', linewidth = 1, alpha = 1)
            plt.axhline(y = x + 0.5, c = 'red', linewidth = 1, alpha = 1)
    


def plotHistory(measure = 'loss', all_reps = False, model_file = None, ax = None):
    """
    Plots the history over epoches of the model.
    One or more measures (eg. the loss) can be selected.
    If all_reps is set to True, the average across the repetitions is plotted instead. 
    
    Arguments:
        model_file -- None, or the name of the model file, ie something like 'ChromAlignNet-H-01-r01'
                      If None, the one specified in the prediction options will be used
        measure -- The component(s) of the history to plot, defined as a string or a list of strings
                   If a list of strings is used, multiple component will be drawn within the same figure
                   The available components are:
                       loss
                       main_prediction_loss, mass_prediction_loss, peak_prediction_loss, chromatogram_prediction_loss,
                       val_main_prediction_loss, val_mass_prediction_loss, val_peak_prediction_loss, val_chromatogram_prediction_loss,
                       main_prediction_acc, mass_prediction_acc, peak_prediction_acc, chromatogram_prediction_acc,
                       val_main_prediction_acc, val_mass_prediction_acc, val_peak_prediction_acc, val_chromatogram_prediction_acc
        all_reps -- If True, plots the mean and standard deviation across all repetitions of the model
    """
    model_file = model_file or prediction_options['model_file']  # Uses what's in the preferences as the default
    if measure == 'acc':
        measure = ['main_prediction_acc', 'val_main_prediction_acc']
    if isinstance(measure, str):
        measure = [measure]  # Makes sure that measure is a list
    
    # If all_reps is true, replace the '-r01' part of the file name to each of the repetitions in turn
    # The mean and standard deviation are taken, and error bars will be drawn
    if all_reps:
        histories = []
        for rep in batch_prediction_options['model_repeats']:
            f = '{}/{}-r{:02}-History.csv'.format(prediction_options['model_path'], model_file[:-4], rep)
            df = pd.read_csv(f, index_col = 0)
            histories.append(df)
        df = pd.concat(histories, axis = 0)
        g = df.groupby(df.index)
        history = g.mean()
        std = g.std()
        formatted_labels = printLastValues(history, std, kind = measure)
        error_kw = {'elinewidth': 1, 'capthick': 1, 'capsize': 2, 'errorevery': len(history) // 50}  # Draw just 50 error bars across the plot
    else:
        history_file = '{}/{}-History.csv'.format(prediction_options['model_path'], model_file)
        history = pd.read_csv(history_file, index_col = 0)
        formatted_labels = printLastValues(history, None, kind = measure)
        error_kw = {}  # No error bar options
    
    # Draw the plot
    history[measure].plot(linewidth = 2, yerr = std[measure] if all_reps else None, ax = ax, **error_kw)
    plt.legend(formatted_labels)
    plt.xlabel('Epochs')


def plotHistoryAcrossModels(model_names, measure = 'loss', all_reps = True):
    """
    Plots the history over epochs of several models by making multiple calls to
    the plotHistory function.
    Only one measure should be specified, to avoid confusion, since the legend of
    the plot refers to model names.
    If all_reps is set to True, the average across the repetitions is plotted instead.
    
    Arguments:
        model_names -- List of the names of the models to plot. The model prefix
                       and repetition number can be specified but is unnecessary.
                       Do not include the model_path - it is taken from the parameters
        measure -- The component of the history to plot. Should be a string, not a list, to avoid confusion
        all_reps -- If True, plots the mean and standard deviation across all repetitions of the models
    """
    # Format the model names correctly so that they can be recognised by plotHistory
    new_model_names = []
    for model_name in model_names:
        if not model_name.startswith('ChromAlignNet'):
            model_name = 'ChromAlignNet-' + model_name
        if model_name[-3] != 'r':
            model_name = model_name + '-r01'  # Default to the first repetition
        new_model_names.append(model_name)
    
    # Do not show the model prefix in the legend. Also if all_reps=True, do not show the rep number
    if all_reps:
        model_short_name = [n[14:-4] for n in new_model_names]
    else:
        model_short_name = [n[14:] for n in new_model_names]

    ax = plt.axes()
    for i, model_file in enumerate(new_model_names):
        print(model_short_name[i], end = ' ')
        plotHistory(measure = measure, all_reps = all_reps, model_file = model_file, ax = ax)
    
    plt.legend(model_short_name)  # Replace the legend with a new one


def plotSubnetworkHistory(measure = 'acc', all_reps = False):
    """
    Plots the history of each sub-network by calling the plotHistory function.
    One of four plots can be drawn, specified by setting the 'measure' parameter.
    The plot can focused on either accuracy or loss, of either the training
    or validation set.
    
    Arguments:
        measure -- The component of the history to plot, defined as a string ('acc', 'loss', 'val_acc' or 'val_loss')
        all_reps -- If True, plots the mean and standard deviation across all repetitions of the model
    """
    if measure.endswith('acc'):
        ms = ['main_prediction_acc', 'mass_prediction_acc', 'peak_prediction_acc', 'chromatogram_prediction_acc']
    if measure.endswith('loss'):
        ms = ['main_prediction_loss', 'mass_prediction_loss', 'peak_prediction_loss', 'chromatogram_prediction_loss']
    if measure.startswith('val'):
        ms = ['val_' + i for i in ms]
    plotHistory(ms, all_reps)
    plt.ylabel('Accuracy' if measure.endswith('acc') else 'Loss')


def plotROC(prediction, comparisons, info_df, ignore_negative_indices = True, show_threshold = False, cmap = 'Spectral'):
    truth, truth_ignore_neg, keep = getGroundTruth(comparisons, info_df, ignore_negative_indices = True)
    if ignore_negative_indices:
        fpr, tpr, thresholds = roc_curve(truth_ignore_neg, prediction[keep], pos_label = 1)
    else:
        fpr, tpr, thresholds = roc_curve(truth, prediction, pos_label = 1)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.3f' % roc_auc)
    plt.legend()
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.title('Receiver Operating Characteristic')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    if show_threshold:
        idx = np.arange(0, len(fpr) - 1, len(fpr) // 15)
        plt.scatter(np.array(fpr)[idx], np.array(tpr)[idx], c = np.array(thresholds)[idx], vmin = 0, vmax = 1, s = 30, zorder = 5, cmap = cmap)
        plt.colorbar(label = 'Threshold')

def plotPrecisionRecallCurve(prediction, comparisons, info_df, ignore_negative_indices = True, show_threshold = False, cmap = 'Spectral'):
    truth, truth_ignore_neg, keep = getGroundTruth(comparisons, info_df, ignore_negative_indices = True)
    if ignore_negative_indices:
        precision, recall, thresholds = precision_recall_curve(truth_ignore_neg, prediction[keep])
    else:
        precision, recall, thresholds = precision_recall_curve(truth, prediction)
    average_precision = average_precision_score(truth_ignore_neg, prediction[keep])
    plt.plot(recall, precision, 'b', label = 'Average Precision = %0.3f' % average_precision)
    plt.legend()
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    if show_threshold:
        idx = np.arange(0, len(precision) - 1, len(precision) // 15)
        plt.scatter(np.array(recall)[idx], np.array(precision)[idx], c = np.array(thresholds)[idx], vmin = 0, vmax = 1, s = 30, zorder = 5, cmap = cmap)
        plt.colorbar(label = 'Threshold')
    


if __name__ == "__main__":
    ## Load data
    info_df, peak_df, mass_profile_df, chromatogram_df, peak_df_orig, peak_intensity = loadData(prediction_options['data_path'], prediction_options['info_file'],
                                                                                                prediction_options['sequence_file'], take_chromatogram_log = False)
    prediction, comparisons = loadPredictions(prediction_options['predictions_save_name'])
    
    ## Make plots
#    plotAlignments(prediction, comparisons, info_df, peak_df_orig, peak_intensity)
#    plt.figure()
    calculateMetrics(prediction, info_df, comparisons, calculate_for_components = False,
                             calculate_f1 = prediction_options['calculate_f1_metric'],
                             calculate_auc = prediction_options['calculate_auc_metric'],
                             print_metrics = True)
    plotROC(prediction, comparisons, info_df)
#    plotPeaksByIndex(info_df, peak_df_orig, mass_profile_df, chromatogram_df, [2], margin = 100, plot_log_sequence = True, read_clipboard = False, plot_as_subplots = False)
#    incorrect = getIncorrectExamples(prediction, info_df, comparisons, ignore_neg = True, sample_size = 5).ravel()
#    incorrect = np.unique(incorrect)
#    plotPeaksByIndex(info_df, peak_df_orig, mass_profile_df, chromatogram_df, incorrect, margin = 100, plot_log_sequence = True, read_clipboard = False, plot_as_subplots = False)
#    

#    plotMetricsByModel(prediction_options['dataset_name'], metrics = ['True Positives', 'False Positives', 'F1'])
#    plotPerformanceOnDatasetsByModel(datasets = [10,11], metric = 'False Positives', reference_model = 0)
#    plotProbabilityMatrix(prediction, comparisons, info_df, threshold = None, sort_by_group = True, highlight_negative_group = True)
#    plotHistory('loss', False)
#    plotSubnetworkHistory('val_acc', False)
#    plotHistoryAcrossModels(model_names = ['H-02', 'H-32', 'I-32'], measure = 'val_main_prediction_acc', all_reps = True)

