import pandas as pd
import numpy as np
import time
import sys
import os
import keras.backend as K
from keras.models import load_model
from utils import loadData, getChromatographSegmentDf, generateCombinationIndices, calculateMetrics
from plotResults import plotAlignments
from model_definition import getModelVariant
from parameters import prediction_options


### Load parameters
### These parameters are loaded here even for any batch prediction scripts
ignore_negatives = prediction_options.get('ignore_negatives')
time_cutoff = prediction_options.get('time_cutoff')
info_file = prediction_options.get('info_file')
sequence_file = prediction_options.get('sequence_file')
results_path = prediction_options.get('results_path')

calculate_f1_metric = prediction_options.get('calculate_f1_metric')
calculate_metrics_for_components = prediction_options.get('calculate_metrics_for_components')
plot_alignment = prediction_options.get('plot_alignment')
ignore_same_sample = prediction_options.get('ignore_same_sample')
###

### These parameter are loaded individually in any batch prediction scripts, since they may change per model / data source
predictions_save_name = prediction_options.get('predictions_save_name')
model_path = prediction_options.get('model_path')
model_file = prediction_options.get('model_file')
data_path = prediction_options.get('data_path')


model_variant = int(model_file.split('-')[2])
chrom_align_model = getModelVariant(model_variant)
ignore_peak_profile = getattr(chrom_align_model, 'ignore_peak_profile')

if os.path.isdir(results_path) == False:
    os.makedirs(results_path)



def prepareDataForPrediction(data_path, ignore_peak_profile):
    """
    Loads data and places into numpy arrays, reading for input into the network
    Similar to the data preparation procedure in the training script
    
    Arguments:
        data_path -- Location of the data set, as a string
        ignore_peak_profile -- If True, the peak profile will not be used in the prediction (this is set to match how the model was trained)
    
    Returns:
        prediction_data -- List of numpy arrays. Passed directly into the network as input data
        comparisons -- Numpy array with two columns - x1 and x2 - containing the IDs of the two peaks being compared
        info_df -- DataFrame containing information about each peak
        peak_df_orig -- DataFrame containing the unnormalised intensities along the profile of each peak
        peak_intensity -- DataFrame containing the maximum intensity of each peak
    """
    load_time = time.time()

    info_df, peak_df, mass_profile_df, chromatogram_df, peak_df_orig, peak_intensity = loadData(data_path, info_file, sequence_file)
    real_groups_available = 'Group' in info_df  # Check if ground truth groups have been assigned

    # Remove null rows and negative indexed groups
    keep_index = (pd.notnull(mass_profile_df).all(1))
    if ignore_negatives and real_groups_available:
        negatives = info_df['Group'] < 0
        keep_index = keep_index & (~negatives)
        print("Negative index ignored: {}".format(np.sum(negatives)))
    if not ignore_peak_profile:
            keep_index = keep_index & (pd.notnull(peak_df).all(1)) 

    print("Dropped rows: {}".format(np.sum(keep_index == False)))

    chrom_seg_df = getChromatographSegmentDf(info_df, chromatogram_df, segment_length = 600)

    # Generate data combinations
    comparisons = generateCombinationIndices(info_df[keep_index], time_cutoff = time_cutoff, return_y = False, ignore_same_sample = ignore_same_sample)
    x1 = comparisons[:,0]
    x2 = comparisons[:,1]

    data_time_1 = info_df.loc[x1]['peakMaxTime'].values
    data_time_2 = info_df.loc[x2]['peakMaxTime'].values
    data_time_diff = abs(data_time_1 - data_time_2)
    if not ignore_peak_profile:
        data_peak_1 = peak_df.loc[x1].values
        data_peak_2 = peak_df.loc[x2].values
    data_mass_spectrum_1 = mass_profile_df.loc[x1].values
    data_mass_spectrum_2 = mass_profile_df.loc[x2].values
    data_chrom_seg_1 = chrom_seg_df.loc[x1].values
    data_chrom_seg_2 = chrom_seg_df.loc[x2].values

    # Print data dimensions
    samples, segment_length = data_chrom_seg_1.shape
    _, max_mass_seq_length = data_mass_spectrum_1.shape
    print('Number of samples:', samples)
    if not ignore_peak_profile:
        _, max_peak_seq_length = data_peak_1.shape
        print('Max peak length:', max_peak_seq_length)
    print('Mass spectrum length:', max_mass_seq_length)
    print('Chromatogram segment length:', segment_length)


    prediction_data = [data_mass_spectrum_1, data_mass_spectrum_2,
                        data_chrom_seg_1.reshape((samples, segment_length, 1)),
                        data_chrom_seg_2.reshape((samples, segment_length, 1)),
                        data_time_diff]

    if not ignore_peak_profile:  # Insert peak data
        prediction_data[2:2] = [data_peak_1.reshape((samples, max_peak_seq_length, 1)),
                                data_peak_2.reshape((samples, max_peak_seq_length, 1))]


    print('Time to load and generate samples: {:.2f} sec'.format(time.time() - load_time))
    print('===============\nPredictions:')
    sys.stdout.flush()
    
    return prediction_data, comparisons, info_df, peak_df_orig, peak_intensity


def runPrediction(prediction_data, model_path, model_file, verbose = 1, predictions_save_name = None, comparisons = None):
    """
    Runs the prediction
    
    Arguments:
        prediction_data -- List of numpy arrays. Input data into the network
        model_path -- Folder containing the model, as a string
        model_file -- Name of the model, as a string
        verbose -- 
        predictions_save_name -- None or a string giving the name to save the prediction outcomes (as a csv file)
        comparisons -- Numpy array with two columns - x1 and x2 - containing the IDs of the two peaks being compared
    
    Returns:
        predictions -- list of numpy arrays. Each item in the list matches one of the output of the model (i.e. main, mass, peak, chromatogram)
                       Take the first item in the list if only the main prediction is desired
                       Gives the probability of the two peaks to being aligned, as predicted by the model
    """
    K.clear_session()
    predict_time = time.time()
    ### Load model
    loading = os.path.join(model_path, model_file) + '.h5'
    print(loading)
    model = load_model(loading)

    predictions = model.predict(prediction_data, verbose = verbose)
    
    if predictions_save_name is not None and comparisons is not None:
        predictions_df = pd.DataFrame(np.concatenate((comparisons, np.squeeze(predictions).T), axis = 1))
        if predictions_df.shape[1] == 6:
            predictions_df.columns = ['x1', 'x2', 'probability', 'prob_mass', 'prob_peak', 'prob_chrom']
        else:
            predictions_df.columns = ['x1', 'x2', 'probability', 'prob_mass', 'prob_chrom']
        predictions_df['x1'] = predictions_df['x1'].astype(int)
        predictions_df['x2'] = predictions_df['x2'].astype(int)
        predictions_df.to_csv(predictions_save_name, index = None)

    print('Time to predict: {:.2f} sec'.format(time.time() - predict_time))
    return predictions





####

if __name__ == "__main__":
    prediction_data, comparisons, info_df, peak_df_orig, peak_intensity = prepareDataForPrediction(data_path, ignore_peak_profile)
    predictions = runPrediction(prediction_data, model_path, model_file, predictions_save_name = predictions_save_name, comparisons = comparisons)
    prediction = predictions[0]

    if 'Group' in info_df:
        calculateMetrics(predictions, info_df, comparisons, calculate_for_components = calculate_metrics_for_components, calculate_f1 = calculate_f1_metric, print_metrics = True)
    
    if plot_alignment:
        plotAlignments(prediction, comparisons, info_df, peak_df_orig, peak_intensity, print_metrics = False)
        
    ##
#    x1 = comparisons[:,0]
#    x2 = comparisons[:,1]
#    g1 = info_df.loc[x1]['Group'].values
#    g2 = info_df.loc[x2]['Group'].values
#    keep = (g1 >= 0) & (g2 >= 0)  # Ignore negative indices
#    truth = (g1 == g2)
#    scores = model.evaluate(prediction_data, [truth] * 3)
#    list(zip(model.metrics_names, scores))

