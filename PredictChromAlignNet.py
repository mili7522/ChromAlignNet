import pandas as pd
import numpy as np
import time
import sys
import os
import keras.backend as K
from keras.models import load_model
from utils import loadData, getChromatographSegmentDf, generateCombinationIndices, getRealGroupAssignments, plotSpectrumTogether, plotPeaksTogether
from utils import getDistanceMatrix, assignGroups, alignTimes, printConfusionMatrix
from model_definition import getModelVariant
from parameters import prediction_options


### Load parameters
### These parameters are loaded here even for any batch prediction scripts
ignore_negatives = prediction_options.get('ignore_negatives')
time_cutoff = prediction_options.get('time_cutoff')
real_groups_available = prediction_options.get('real_groups_available')
info_file = prediction_options.get('info_file')
sequence_file = prediction_options.get('sequence_file')
results_path = prediction_options.get('results_path')
###

### These parameter are loaded individually in any batch prediction scripts, since they may change per model / data source
predictions_save_name = prediction_options.get('predictions_save_name')
model_path = prediction_options.get('model_path')
model_file = prediction_options.get('model_file')
data_path = prediction_options.get('data_path')


model_variant = int(model_file.split('-')[2])
chrom_align_model = getModelVariant(model_variant)
ignore_peak_profile = getattr(chrom_align_model, 'ignore_peak_profile') #

if os.path.isdir(results_path) == False:
    os.makedirs(results_path)


### Load peak and mass slice profiles
def prepareDataForPrediction(data_path, ignore_peak_profile):
    loadTime = time.time()

    info_df, peak_df, mass_profile_df, chromatogram_df, peak_df_orig, peak_intensity = loadData(data_path, info_file, sequence_file)

    keep_index = (pd.notnull(peak_df).all(1)) & (pd.notnull(mass_profile_df).all(1))
    if ignore_negatives and real_groups_available:
        negatives = info_df['Group'] < 0
        keep_index = keep_index & (~negatives)
        print("Negative index ignored: {}".format(np.sum(negatives)))

    print("Dropped rows: {}".format(np.sum(keep_index == False)))
    print(np.flatnonzero(keep_index == False))

    chrom_seg_df = getChromatographSegmentDf(info_df, chromatogram_df, segment_length = 600)

    ### Generate data combinations
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
    data_chrom_seg_2 = chrom_seg_df.loc[x2].values

    samples, max_peak_seq_length = data_peak_1.shape
    _, max_mass_seq_length = data_mass_spectrum_1.shape
    _, segment_length = data_chrom_seg_1.shape

    print('Number of samples:', samples)
    print('Max peak sequence length:', max_peak_seq_length)
    print('Max mass spectrum length:', max_mass_seq_length)
    print('Chromatogram segment length:', segment_length)

    print('Time to load and generate samples:', round((time.time() - loadTime)/60, 2), 'min')
    #print('\n')   
    print('===============\nPredictions:')   # XRW
    sys.stdout.flush()   # XRW


    prediction_data = [data_mass_spectrum_1, data_mass_spectrum_2,
                        data_chrom_seg_1.reshape((samples, segment_length, 1)),
                        data_chrom_seg_2.reshape((samples, segment_length, 1)),
                        data_time_diff]

    if not ignore_peak_profile:  # Insert peak data
        prediction_data[2:2] = [data_peak_1.reshape((samples, max_peak_seq_length, 1)),
                                data_peak_2.reshape((samples, max_peak_seq_length, 1))]

    return prediction_data, comparisons, info_df, peak_df_orig, peak_intensity


def runPrediction(prediction_data, model_path, model_file, verbose = 1, predictions_save_name = None, comparisons = None):
    ### Predict
    K.clear_session()
    predict_time = time.time()
    ### Load model
    loading = os.path.join(model_path, model_file) + '.h5'
    print(loading)
    model = load_model(loading)

    prediction = model.predict(prediction_data, verbose = verbose)
    
    if predictions_save_name is not None and comparisons is not None:
        predictions_df = pd.DataFrame(np.concatenate((comparisons, np.squeeze(prediction).T), axis = 1))
        if predictions_df.shape[1] == 6:
            predictions_df.columns = ['x1', 'x2', 'probability', 'prob_mass', 'prob_peak', 'prob_chrom']
        else:
            predictions_df.columns = ['x1', 'x2', 'probability', 'prob_mass', 'prob_chrom']
        predictions_df['x1'] = predictions_df['x1'].astype(int)
        predictions_df['x2'] = predictions_df['x2'].astype(int)
        predictions_df.to_csv(predictions_save_name, index = None)

#    prediction = prediction[0]  # Only take the main outcome

    #print('Time to predict:', round((time.time() - predict_time)/60, 2), 'min')
    print('Time to predict:', time.time() - predict_time, 'sec')
    return prediction





####

if __name__ == "__main__":
    prediction_data, comparisons, info_df, peak_df_orig, peak_intensity = prepareDataForPrediction(data_path, ignore_peak_profile)
    prediction_all = runPrediction(prediction_data, model_path, model_file, predictions_save_name = predictions_save_name, comparisons = comparisons)
    prediction = prediction_all[0]

    distance_matrix = getDistanceMatrix(comparisons, info_df.index.max() + 1, prediction, clip = 50)

    groups = assignGroups(distance_matrix, threshold = 2)

    alignTimes(groups, info_df, peak_intensity, 'AlignedTime')
    if real_groups_available:
        real_groups = getRealGroupAssignments(info_df)
        printConfusionMatrix(prediction, info_df, comparisons)
        alignTimes(real_groups, info_df, peak_intensity, 'RealAlignedTime')


# TODO: Add option to control plotting
#    if ignore_negatives:
#        plotSpectrumTogether(info_df[info_df['Group'] >= 0], peak_intensity[info_df['Group'] >= 0], with_real = real_groups_available)
#    else:
#        plotSpectrumTogether(info_df, peak_intensity, with_real = real_groups_available)
#
#    if ignore_negatives:
#        plotPeaksTogether(info_df[info_df['Group'] >= 0], peak_df_orig[info_df['Group'] >= 0], with_real = real_groups_available)
#    else:
#        plotPeaksTogether(info_df, peak_df_orig, with_real = real_groups_available)
