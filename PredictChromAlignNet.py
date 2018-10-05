import pandas as pd
import numpy as np
import scipy.spatial
import scipy.cluster
import time
import sys
import os
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model
from utils import loadData, getChromatographSegmentDf, generateCombinationIndices, getRealGroupAssignments, plotSpectrumTogether, plotPeaksTogether
from model_definition import getModelVariant
from parameters import prediction_options


# Load parameters
ignore_negatives = prediction_options.get('ignore_negatives')
time_cutoff = prediction_options.get('time_cutoff')
predictions_save_name = prediction_options.get('predictions_save_name')
model_path = prediction_options.get('model_path')
model_file = prediction_options.get('model_file')
data_path = prediction_options.get('data_path')
info_file = prediction_options.get('info_file')
sequence_file = prediction_options.get('sequence_file')
results_path = prediction_options.get('results_path')


model_variant = int(model_file.split('-')[2])
chrom_align_model = getModelVariant(model_variant)
ignore_peak_profile = getattr(chrom_align_model, 'ignore_peak_profile')


if os.path.isfile(os.path.join(data_path, info_file)):
    real_groups_available = True
else:
    info_file = 'PeakData.csv'
    real_groups_available = False


### Load peak and mass slice profiles
def prepareDataForPrediction(data_path, info_file, sequence_file, ignore_peak_profile = ignore_peak_profile):
    loadTime = time.time()

    info_df, peak_df, mass_profile_df, chromatogram_df, peak_df_orig, peak_df_max = loadData(data_path, info_file, sequence_file)

    if ignore_negatives and real_groups_available:
        negatives = info_df['Group'] < 0
        info_df = info_df[~negatives]
        peak_df = peak_df[~negatives]
        peak_df_orig = peak_df_orig[~negatives]
        peak_df_max = peak_df_max[~negatives]
        mass_profile_df = mass_profile_df[~negatives]
        info_df.reset_index(inplace = True, drop = False)
        peak_df.reset_index(inplace = True, drop = True)
        peak_df_orig.reset_index(inplace = True, drop = True)
        peak_df_max.reset_index(inplace = True, drop = True)
        mass_profile_df.reset_index(inplace = True, drop = True)
        print("Negative index ignored: {}".format(np.sum(negatives)))

    keep_index = (pd.notnull(peak_df).all(1)) & (pd.notnull(mass_profile_df).all(1))

    print("Dropped rows: {}".format(np.sum(keep_index == False)))
    print(np.flatnonzero(keep_index == False))

    chrom_seg_df = getChromatographSegmentDf(info_df, chromatogram_df, segment_length = 600)

    #%% Generate data combinations
    comparisons = generateCombinationIndices(info_df[keep_index], time_cutoff = time_cutoff, return_y = False)
    x1 = comparisons[:,0]
    x2 = comparisons[:,1]

    data_time_1 = info_df.loc[x1]['peakMaxTime'].values
    data_time_2 = info_df.loc[x2]['peakMaxTime'].values
    data_time_diff = abs(data_time_1 - data_time_2)
    data_peak_1 = peak_df.loc[x1].values
    data_peak_2 = peak_df.loc[x2].values
    data_mass_spectrum_1 = mass_profile_df.loc[x1].values
    data_mass_spectrum_2 = mass_profile_df.loc[x2].values
    data_chrom_seg_1 = chrom_seg_df.loc[x1].values
    data_chrom_seg_1 = chrom_seg_df.loc[x2].values

    samples, max_peak_seq_length = data_peak_1.shape
    _, max_mass_seq_length = data_mass_spectrum_1.shape
    _, segment_length = data_chrom_seg_1.shape

    print('Number of samples:', samples)
    print('Max peak sequence length:', max_peak_seq_length)
    print('Max mass spectrum length:', max_mass_seq_length)
    print('Chromatogram segment length:', segment_length)

    print('Time to load and generate samples:', round((time.time() - loadTime)/60, 2), 'min')
    print('\n')   # XRW
    print('===============\nPredictions:\n---')   # XRW
    sys.stdout.flush()   # XRW


    if ignore_peak_profile:
        prediction_data = [data_mass_spectrum_1, data_mass_spectrum_2,
                            data_chrom_seg_1.reshape((samples, segment_length, 1)),
                            data_chrom_seg_1.reshape((samples, segment_length, 1)),
                            data_time_diff]
    else:
        prediction_data = [data_mass_spectrum_1, data_mass_spectrum_2,
                            data_peak_1.reshape((samples, max_peak_seq_length, 1)),
                            data_peak_2.reshape((samples, max_peak_seq_length, 1)),
                            data_chrom_seg_1.reshape((samples, segment_length, 1)),
                            data_chrom_seg_1.reshape((samples, segment_length, 1)),
                            data_time_diff]

    return prediction_data, comparisons, info_df, peak_df_orig, peak_df_max

def runPrediction(prediction_data, model_path, model_file):
    #%% Predict
    K.clear_session()
    predict_time = time.time()
    ### Load model
    loading = os.path.join(model_path, model_file) + '.h5'
    print(loading)
    model = load_model(loading)

    prediction = model.predict(prediction_data, verbose = 1)

    prediction = prediction[0]  # Only take the main outcome

    #print('Time to predict:', round((time.time() - predict_time)/60, 2), 'min')
    print('Time to predict:', time.time() - predict_time, 'sec')
    return prediction


#%% Group and cluster
def getDistances(prediction):
    distances = 1 / prediction
    return distances
    

def getDistanceMatrix(comparisons, peaks, prediction, clip = 10):
    distances = getDistances(prediction)
    
    distance_matrix = np.empty((peaks, peaks))
    distance_matrix.fill(clip)  # Clip value
    
    for i, (x1, x2) in enumerate(comparisons):
        distance_matrix[x1, x2] = min(distances[i], clip)
        distance_matrix[x2, x1] = min(distances[i], clip)
    
    for i in range(peaks):
        distance_matrix[i,i] = 0
    
    return distance_matrix


def assignGroups(distance_matrix, threshold = 2):
    sqform = scipy.spatial.distance.squareform(distance_matrix)
    mergings = scipy.cluster.hierarchy.linkage(sqform, method = 'average')
#    plt.figure()
#    dn = scipy.cluster.hierarchy.dendrogram(mergings, leaf_font_size = 3)
#    plt.savefig(data_path + 'Dendrogram.png', dpi = 300, format = 'png', bbox_inches = 'tight')
    labels = scipy.cluster.hierarchy.fcluster(mergings, threshold, criterion = 'distance')
    
    groups = {}
    for i in range(max(labels)):
        groups[i] = set(np.where(labels == i + 1)[0])  # labels start at 1
    
    return groups


def alignTimes(groups, info_df, align_to):
    info_df[align_to] = info_df['peakMaxTime']
    for group in groups.values():
        times = info_df.loc[group, 'peakMaxTime']
        average_time = np.mean(times)
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


####

if __name__ == "__main__":
    prediction_data, comparisons, info_df, peak_df_orig, peak_df_max = prepareDataForPrediction(data_path, info_file, sequence_file)
    prediction = runPrediction(prediction_data, model_path, model_file)
    if predictions_save_name is not None:
        if os.path.isdir(results_path) == False:
            os.makedirs(results_path)
        predictions_df = pd.DataFrame(np.concatenate((comparisons, prediction), axis = 1), columns = ['x1', 'x2', 'prediction'])
        predictions_df['x1'] = predictions_df['x1'].astype(int)
        predictions_df['x2'] = predictions_df['x2'].astype(int)
        predictions_df.to_csv(predictions_save_name, index = None)

    distance_matrix = getDistanceMatrix(comparisons, info_df.index.max() + 1, prediction, clip = 10)

    groups = assignGroups(distance_matrix, threshold = 2)

    alignTimes(groups, info_df, 'AlignedTime')
    if real_groups_available:
        real_groups = getRealGroupAssignments(info_df)
        alignTimes(real_groups, info_df, 'RealAlignedTime')
        printConfusionMatrix(prediction, info_df, comparisons)


    plotSpectrumTogether(info_df, peak_df_max, with_real = real_groups_available)
    if not ignore_negatives:
        plotSpectrumTogether(info_df[info_df['Group'] >= 0], peak_df_max[info_df['Group'] >= 0], with_real = real_groups_available)


    plotPeaksTogether(info_df, peak_df_orig, with_real = real_groups_available)
    if not ignore_negatives:
        plotPeaksTogether(info_df[info_df['Group'] >= 0], peak_df_orig[info_df['Group'] >= 0], with_real = real_groups_available)  # Peaks not normalised