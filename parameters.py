training_options = {
    'epochs': 50,
    'batch_size': 128,
    'validation_split': 0.2,
    'adam_optimizer_options': {'lr': 0.001, 'beta_1': 0.9, 'beta_2': 0.999},
    'train_with_gpu': True,
    'random_seed_type': 1,  # Type 1 for repeatability within repetitions, type 2 for random
    
    'save_checkpoints': True,  # If checkpoints exist, training will resume from the last checkpoint
    'save_history': True,
    'model_path': 'SavedModels',
    'model_name': 'ChromAlignNet',


    'datasets': [ 'data/2018-04-22-ExtractedPeaks-Air103-WithMassSlice/',
                  'data/2018-04-30-ExtractedPeaks-Air115-WithMassSlice/',
                  'data/2018-04-30-ExtractedPeaks-Air143-WithMassSlice/',
                  'data/2018-05-01-ExtractedPeaks-Breath103-WithMassSlice/',
                  'data/2018-05-01-ExtractedPeaks-Breath115-WithMassSlice/',
                  'data/2018-05-14-ExtractedPeaks-Breath73-WithMassSlice-All/',
                  'data/2018-05-14-ExtractedPeaks-Breath88-WithMassSlice-All/'
                ],
    'dataset_for_model': {
                         'A': [0, 1, 2], 
                         'B': [0, 1, 2, 3],
                         'C': [0, 1, 2, 4],
                         'D': [0, 1, 2, 3, 4],
                         'E': [0, 1, 2, 3, 4, 5],
                         'F': [0, 1, 2, 3, 4, 6]
                         },

    'info_file': 'PeakData-WithGroup.csv',
    'sequence_file': 'WholeSequence.csv'
}

prediction_options = {
    'ignore_negatives': False,  # Ignore groups assigned with a negative index?
    'time_cutoff': 3, # Three minutes
    'predictions_save_name': None,

    'model_path': 'SavedModels/',
    'model_file': 'ChromAlignNet-A-20-r01',
    # 'data_path': 'data/2018-05-14-ExtractedPeaks-Breath73-WithMassSlice-All/',
    'data_path': 'data/2018-05-01-ExtractedPeaks-Breath103-WithMassSlice/',
    'info_file': 'PeakData-WithGroup.csv',
    'sequence_file': 'WholeSequence.csv'
}

prediction_options['predictions_save_name'] = 'SavedModels/{}_{}_Prediction.csv'.format(prediction_options['model_file'], prediction_options['data_path'].split('-')[4])