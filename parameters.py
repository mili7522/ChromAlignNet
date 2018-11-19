training_options = {
    'epochs': 50,
    'batch_size': 128,
    'validation_split': 0.2,
    'verbose_training': 2, # 0 = silent, 1 = progress bar, 2 = one line per epoch 
    'adam_optimizer_options': {'lr': 0.001, 'beta_1': 0.9, 'beta_2': 0.999},
    'train_with_gpu': True,
    'random_seed_type': 1,  # Type 1 for repeatability within repetitions, type 2 for random number using clock time, type 3 for previously used number
    
    'save_checkpoints': True,  # If checkpoints exist, training will resume from the last checkpoint
    'save_history': True,
    'model_path': 'SavedModels',
    'model_name': 'ChromAlignNet',


    'datasets': [ 'data/training-Air103-WithMassSlice/',        #0
                  'data/training-Air115-WithMassSlice/',        #1
                  'data/training-Air143-WithMassSlice/',        #2
                  'data/training-Breath103-WithMassSlice/',     #3
                  'data/training-Breath115-WithMassSlice/',     #4
                  'data/training-Breath73-WithMassSlice-All/',  #5
                  'data/training-Breath88-WithMassSlice-All/'   #6
                ],
    'dataset_for_model': {
                         'A': [0, 1],
                         'B': [0, 1, 2],
                         'C': [0, 1, 3],
                         'D': [0, 1, 4],
                         'E': [0, 1, 3, 4],
                         'F': [0, 1, 3, 5],
                         'G': [0, 1, 3, 6],
                         'H': [0, 1, 2, 3, 4, 5, 6]
                         },

    'ignore_negatives': False,  # Ignore groups assigned with a negative index?
    'info_file': 'PeakData-WithGroup.csv',
    'sequence_file': 'WholeSequence.csv'
}

prediction_options = {
    'ignore_negatives': False,  # Ignore groups assigned with a negative index?
    'time_cutoff': 3, # Three minutes
    'predictions_save_name': None,
    'results_path': 'results',

    'model_path': 'SavedModels/',
    'model_file': 'ChromAlignNet-D-02-r04',
    'data_path': training_options['datasets'][3],
    'info_file': 'PeakData-WithGroup.csv',
    'sequence_file': 'WholeSequence.csv',
    'real_groups_available': True
}

prediction_options['predictions_save_name'] = '{}/{}_{}_Prediction.csv'.format(prediction_options['results_path'], prediction_options['model_file'], prediction_options['data_path'].split('-')[4])


batch_prediction_options = {
    'save_names': ["ModelTests-On{}.csv".format(x.split('-')[1]) for x in training_options['datasets']],
    'model_repeats': range(1,11),
    'model_names': ['G'], # ['A', 'B', 'C', 'D', 'E', 'F', 'G'],  
    'model_variants': [1], #[20, 21, 26], # range(1, 28),
    'verbose_prediction' : 0
}
