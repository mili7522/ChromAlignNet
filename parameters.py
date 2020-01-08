import os
import numpy as np  # Import in case extraction options time window needs np.inf

data_options = {
    'datasets': [ 'data/training-Air103/',        #0   Folder location of available data sets (both training and test)
                  'data/training-Air115/',        #1
                  'data/training-Air143/',        #2
                  'data/training-Breath103/',     #3
                  'data/training-Breath115/',     #4
                  'data/training-Breath73/',      #5
                  'data/training-Breath88/',  	  #6
                  'data/test-Air92/',             #7
                  'data/test-Air134/',            #8
                  'data/test-Field73/',           #9
                  'data/test-Field88/',           #10
                  'data/test-Field134/'           #11
                ]
}
data_options['dataset_name'] = [ x.split('-')[-1].strip('/') for x in data_options['datasets'] ]


training_options = {
    'epochs': 50,  # Number of iterations through the training set
    'batch_size': 128,  # Number of training examples per mini-batch
    'validation_split': 0.2,  # Ratio of the data set to be used for validation
    'verbose_training': 2,  # 0 = silent, 1 = progress bar, 2 = one line per epoch 
    'adam_optimizer_options': {'lr': 0.001, 'beta_1': 0.9, 'beta_2': 0.999},  # Optimiser options. lr = learning rate
    'random_seed_type': 1,  # How the random seed is generated. Type 1 for repeatability (generates from model name), type 2 for random number using clock time, type 3 to load the last used number
    
    'save_checkpoints': True,  # If True, the model will be (uniquely) saved after every epoch. If checkpoints exist, training will resume from the last checkpoint
    'save_history': True,  # If True, the loss and accuracy of the model over epoches will be saved as a csv file
    'model_path': 'SavedModels',  # Folder to save the model (as a .h5 file). Checkpoints will be saved in a subfolder 'Checkpoints'
    'model_name': 'ChromAlignNet',  # Name prefix given to the saved model

    'datasets' : data_options['datasets'],
    'dataset_for_model': {  # Dictionary to select which preset list of data sets (indexed from 'datasets') will be used to train each model ('A', 'B', etc)
                         'A': [0, 1],
                         'B': [0, 1, 2],
                         'C': [0, 1, 3],
                         'D': [0, 1, 4],
                         'E': [0, 1, 3, 4],
                         'F': [0, 1, 3, 5],
                         'G': [0, 1, 3, 6],
                         'H': [0, 1, 2, 3, 4, 5, 6]
                         },

    'ignore_negatives': False,  # If True, groups assigned with a negative index will be ignored in training
    'info_file': 'PeakData-WithGroup.csv',  # Name of csv file containing overall information about each peak. Loaded as info_df
    'sequence_file': 'WholeSequence.csv'  # Name of csv file containing the data used to extract the chromatogram segments
}


prediction_options = {
    'ignore_negatives': False,  # If True, groups assigned with a negative index will be ignored in prediction
    'time_cutoff': 3,  # Maximum difference in retention time (in minutes) between two peaks which will be considered for alignment
    'results_path': 'results',  # Path to save results

    'model_path': 'SavedModels',  # Path where trained models are saved
    'model_file': 'ChromAlignNet-A-02-r08',  # Name of specific model to load and use for prediction
    'dataset_number': 1,  # Data set to align (indexed according to data_options['datasets'])
    'info_file': 'PeakData-WithGroup.csv',  # Name of csv file containing summary information about each peak. Loaded as infoDf
    'sequence_file': 'WholeSequence.csv',  # Name of csv file containing the data used to extract the chromatogram segments
    'plot_alignment': True,  # If True, the alignment outcome will be plotted after prediction in PredictChromAlignNet.py
    'calculate_f1_metric': True,
    'calculate_auc_metric': True,
    'calculate_average_precision_metric': True,
    'calculate_metrics_for_components': True,  # If True, the metrics for the output from each sub-network will be calculated as well as the main output
    'ignore_same_sample': False  # If True, peaks from within the same sample will not be considered for alignment (and so assumed to have a probability of 0)
}
prediction_options['dataset_name'] = data_options['dataset_name'][prediction_options['dataset_number']]
prediction_options['data_path'] = data_options['datasets'][prediction_options['dataset_number']]
# Path and filename of where prediction output will be saved:
prediction_options['predictions_save_name'] = os.path.join(prediction_options['results_path'],
                  prediction_options['model_file'] + "_" + data_options['dataset_name'][prediction_options['dataset_number']] + "_Prediction.csv")


batch_prediction_options = {
    'dataset_name': data_options['dataset_name'],
    'data_paths' : data_options['datasets'],
    'model_prefix': training_options['model_name'],
    'save_names': ["ModelTests-On{}.csv".format(x) for x in data_options['dataset_name']],  # Gives the name of the csv file to be saved with the metrics from each prediction. The particular name will be selected by indexing with the ID of the data set
    'model_repeats': range(1,11),  # The full range of repetitions of the model which were trained
    'model_names': ['H'],  # The letter name of the model that was trained (indicating the data sets that were used in the training)
    'model_variants': [1, 2, 20, 21, 23, 24],  # The variant of the model that was trained (defined in modelDefinition.py)
    'verbose_prediction' : 0,  # 0 = silent, 1 = progress bar
    'save_individual_predictions': True,  # If True, the prediction outcomes (probabilities) are saved as well as the metrics
    'individual_predictions_save_path': 'Individual',  # Name of subfolder to store individual prediction output, or 'None' to save in the main 'results_path'
    'continue_from_saved': True
}


extraction_options = {
    'data_path' : 'X:/QIMR_QTOF/PfTrial/QTOF/1.rawData_AmbientAir/sigThreshold_10/0fullResults',  # Path to the '0fullResults' output folder from peak detection (https://github.com/rosalind-wang/GCPeakDetection)
    'save_path': 'data/temp-Air103/',  # Output path, as a string
    'masses': [103],  # List of masses to extract peaks from
    'max_files_to_process': 100,  # The maximum number of files (samples) to extract from, as an Int
    'max_peaks_per_file': 1000,  # The maximum number of peaks to extract per file (sample), as an Int
    'time_window': (13.9, 15.1),  # Tuple of (start time, end time) in minutes. Peaks are only extracted within this window. Can use (0, np.inf) to get the whole chromatogram
    'chromatogram_margin': 300,  # Number of time steps to add to either side of the time window when extracting the chromatogram
    'sort_by_peak_area': False,  # If True, the peaks are extracted with IDs assigned in order of peak area. Useful when 'max_peaks_per_file' is low
    'shuffle_files': False,  # If True, the order of the samples are shuffled before extraction. If False, samples are sorted
    'peak_id_width': 3  # Number of digits in the peak id (padded by 0s). Ensure that this in enough to cover the total number of peaks generated (eg 3 for up to 999 peaks) so that the ordering of peaks is sorted the same across different operating systems
}