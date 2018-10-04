training_options = {
    'epochs': 50,
    'batch_size': 128,
    'validation_split': 0.1,
    'adamOptimizerOptions': {'lr': 0.001, 'beta_1': 0.9, 'beta_2': 0.999},
    'trainWithGPU': True,
    'randomSeedType': 1,
    
    'saveCheckpoints': True,  # If checkpoints exist, training will resume from the last checkpoint
    'saveHistory': True,
    'modelPath': 'SavedModels',
    'modelName': 'ChromAlignNet',


    'dataSets': [ 'data/2018-04-22-ExtractedPeaks-Air103-WithMassSlice/',
                  'data/2018-04-30-ExtractedPeaks-Air115-WithMassSlice/',
                  'data/2018-04-30-ExtractedPeaks-Air143-WithMassSlice/',
                  'data/2018-05-01-ExtractedPeaks-Breath103-WithMassSlice/',
                  'data/2018-05-01-ExtractedPeaks-Breath115-WithMassSlice/',
                  'data/2018-05-14-ExtractedPeaks-Breath73-WithMassSlice-All/',
                  'data/2018-05-14-ExtractedPeaks-Breath88-WithMassSlice-All/'
                ],
    'datasetForModel': {
                         'A': [0, 1, 2], 
                         'B': [0, 1, 2, 3],
                         'C': [0, 1, 2, 4],
                         'D': [0, 1, 2, 3, 4],
                         'E': [0, 1, 2, 3, 4, 5],
                         'F': [0, 1, 2, 3, 4, 6]
                         },

    'infoFile': 'PeakData-WithGroup.csv',
    'sequenceFile': 'WholeSequence.csv'
}

prediction_options = {
    'ignoreNegatives': False,  # Ignore groups assigned with a negative index?
    'timeCutOff': 3, # Three minutes

    'modelPath': 'SavedModels/',
    'modelFile': 'ChromAlignNet-A-20-r01',
    'dataPath': 'data/2018-05-14-ExtractedPeaks-Breath73-WithMassSlice-All/',
    'infoFile': 'PeakData-WithGroup.csv',
    'sequenceFile': 'WholeSequence.csv'
}