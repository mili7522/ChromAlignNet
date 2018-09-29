import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import itertools
import sys
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from utils import loadData, printShapes
from model_definition import define_model


#%% Options
epochs = 12
batch_size = 128
validation_split = 0.1
adamOptimizerOptions = {'lr': 0.001, 'beta_1': 0.9, 'beta_2': 0.999}
trainWithGPU = True

saveCheckpoints = True  # If checkpoints exist, training will resume from the last checkpoint
saveHistory = True
modelPath = 'SavedModels'
modelName = 'ChromAlignNet'

# Modify the model name for different data sources and repetitions
if len(sys.argv) > 1:
    assert sys.argv[1] in ('A', 'B', 'C', 'D', 'E', 'F'), "Dataset selection needs to be a letter between A and F"
    datasetSelection = sys.argv[1]
else:
    datasetSelection = 'A'
if len(sys.argv) > 2:
    try:
        repetition = int(sys.argv[2])
    except:
        print("Repetition needs to be a number", file=sys.stderr)
        raise
else:
    repetition = 1
modelName = modelName + '-' + datasetSelection + '-01' + '-r' + str(repetition).rjust(2, '0')

print("Output model to: ", modelName)

if trainWithGPU:
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)

#%% Load and pre-process data
dataSets = [ 'data/2018-04-22-ExtractedPeaks-Air103-WithMassSlice/',
             'data/2018-04-30-ExtractedPeaks-Air115-WithMassSlice/',
             'data/2018-04-30-ExtractedPeaks-Air143-WithMassSlice/',
             'data/2018-05-01-ExtractedPeaks-Breath103-WithMassSlice/',
             'data/2018-05-01-ExtractedPeaks-Breath115-WithMassSlice/',
             'data/2018-05-14-ExtractedPeaks-Breath73-WithMassSlice-All/',
             'data/2018-05-14-ExtractedPeaks-Breath88-WithMassSlice-All/'
            ]
datasetForModel = {
                    'A': [0, 1, 2], 
                    'B': [0, 1, 2, 3],
                    'C': [0, 1, 2, 4],
                    'D': [0, 1, 2, 3, 4],
                    'E': [0, 1, 2, 3, 4, 5],
                    'F': [0, 1, 2, 3, 4, 6]
                  }

dataPaths = list( dataSets[i] for i in datasetForModel[datasetSelection] )

infoFile = 'PeakData-WithGroup.csv'
sequenceFile = 'WholeSequence.csv'


#%% Generate data combinations

dataTime1 = []
dataTime2 = []
dataPeakProfile1 = []
dataPeakProfile2 = []
dataMassProfile1 = []
dataMassProfile2 = []
sequenceProfile1 = []
sequenceProfile2 = []
dataY = []
comparisons = []


def generateCombinations(infoDf, peakDf, massProfileDf, sequenceDf):
    groups = sorted(infoDf.Group.unique())
    
    def appendSequenceProfile(time, file, margin = 300):
        seq = np.zeros((margin * 2))
        timeIdx = np.argmin(np.abs(time - sequenceDf.columns.values))
        if margin > timeIdx:
            seq[margin - timeIdx:] = sequenceDf.loc[str(file)].iloc[:timeIdx + margin].copy()
        else:
            insert = sequenceDf.loc[str(file)].iloc[timeIdx - margin: timeIdx + margin].copy()
            seq[:len(insert)] = insert
        
        idx = seq > 0
        seq[idx] = seq[idx] - np.min(seq[idx])
        
        return seq
        
    def appendData(x1, x2):
        time1 = infoDf.loc[x1]['peakMaxTime']
        time2 = infoDf.loc[x2]['peakMaxTime']
        dataTime1.append(time1)
        dataTime2.append(time2)
        dataPeakProfile1.append(peakDf.loc[x1])
        dataPeakProfile2.append(peakDf.loc[x2])
        dataMassProfile1.append(massProfileDf.loc[x1])
        dataMassProfile2.append(massProfileDf.loc[x2])
        sequenceProfile1.append(appendSequenceProfile(time1, int(infoDf.loc[x1]['File'])))
        sequenceProfile2.append(appendSequenceProfile(time2, int(infoDf.loc[x2]['File'])))
        comparisons.append((x1, x2))
    
    for group in groups:
        groupIndex = infoDf[infoDf['Group'] == group].index
        combinations = itertools.combinations(groupIndex, 2)
        count = 0
        for x1, x2 in combinations:
            appendData(x1, x2)
            dataY.append(1)
            count += 1
        
        largerGroups = infoDf[infoDf['Group'] > group].index 
        combinations = itertools.product(groupIndex, largerGroups)
        combinations = np.array(list(combinations))
        np.random.shuffle(combinations)
        
        for i, (x1, x2) in enumerate(combinations):
            if i > count: break
            appendData(x1, x2)
            dataY.append(0)

### Execute load for all folders
for dataPath in dataPaths:
    print('Loading:', dataPath)
    infoDf, peakDf, massProfileDf, sequenceDf, _, _ = loadData(dataPath, infoFile, sequenceFile)

    # Remove null rows and negative indexed groups
    keepIndex = (pd.notnull(peakDf).all(1)) & (pd.notnull(massProfileDf).all(1)) & (infoDf['Group'] >= 0)
    infoDf = infoDf[keepIndex]
    peakDf = peakDf[keepIndex]
    massProfileDf = massProfileDf[keepIndex]
    print("Dropped rows: {}".format(np.sum(keepIndex == False)))

    a = len(comparisons)
    generateCombinations(infoDf, peakDf, massProfileDf, sequenceDf)
    print(len(comparisons) - a, 'combinations generated')
    
### Shuffle data
np.random.seed(100)  # Set seed for repeatability in hyperparameter tests and continued training
shuffleIndex = np.random.permutation(len(dataTime1))
dataTime1 = np.array(dataTime1)[shuffleIndex]
dataTime2 = np.array(dataTime2)[shuffleIndex]
dataPeakProfile1 = pd.concat(dataPeakProfile1, axis = 1)  # Use pd to concat since the DataSeries might be of different lengths
dataPeakProfile1.fillna(0, inplace = True)
dataPeakProfile1 = dataPeakProfile1.transpose().values[shuffleIndex]
dataPeakProfile2 = pd.concat(dataPeakProfile2, axis = 1)
dataPeakProfile2.fillna(0, inplace = True)
dataPeakProfile2 = dataPeakProfile2.transpose().values[shuffleIndex]
dataMassProfile1 = np.array(dataMassProfile1)[shuffleIndex]
dataMassProfile2 = np.array(dataMassProfile2)[shuffleIndex]
sequenceProfile1 = np.array(sequenceProfile1)[shuffleIndex]
sequenceProfile2 = np.array(sequenceProfile2)[shuffleIndex]
dataY = np.array(dataY)[shuffleIndex]
comparisons = np.array(comparisons)[shuffleIndex]


### Split into training and test sets
training = len(dataTime1) // 5 * 4  # 80% training set, 20% testing set

trainingTime1 = dataTime1[:training]
trainingTime2 = dataTime2[:training]
trainingPeakProfile1 = dataPeakProfile1[:training]
trainingPeakProfile2 = dataPeakProfile2[:training]
trainingMassProfile1 = dataMassProfile1[:training]
trainingMassProfile2 = dataMassProfile2[:training]
trainingSequenceProfile1 = sequenceProfile1[:training]
trainingSequenceProfile2 = sequenceProfile2[:training]
trainingY = dataY[:training]

testingTime1 = dataTime1[training:]
testingTime2 = dataTime2[training:]
testingPeakProfile1 = dataPeakProfile1[training:]
testingPeakProfile2 = dataPeakProfile2[training:]
testingMassProfile1 = dataMassProfile1[training:]
testingMassProfile2 = dataMassProfile2[training:]
testingSequenceProfile1 = sequenceProfile1[training:]
testingSequenceProfile2 = sequenceProfile2[training:]
testingY = dataY[training:]
testingComparisions = comparisons[training:]


samples, max_peak_seq_length = dataPeakProfile1.shape
sequence_length = sequenceProfile1.shape[1]
testing_samples, max_mass_seq_length = testingMassProfile1.shape
training_samples = training

print('Number of samples:', samples)
print('Number of training samples:', training_samples)
print('Number of testing samples:', testing_samples)
print('Max peak length:', max_peak_seq_length)
print('Sequence length:', sequence_length)
print('Max mass sequence length:', max_mass_seq_length)




# Check if existing checkpoints are present - if so then load and resume training from last epoch
checkpointPath = os.path.join(modelPath, modelName + '-Checkpoint')
if os.path.isdir(checkpointPath):
    files = []
    for f in os.listdir(checkpointPath):
        if f.endswith('.h5'):
            files.append(os.path.join(checkpointPath, f))
    last_checkpoint = sorted(files)[-1]
    model = load_model(last_checkpoint)
    initial_epoch = int(last_checkpoint[-6:-3])
else:
    model = define_model(max_mass_seq_length, sequence_length)
    initial_epoch = 1


model.compile(optimizer = Adam(**adamOptimizerOptions),
              loss = 'binary_crossentropy',
              metrics = ['accuracy'],
              loss_weights = [1., 0.2, 0.2, 0.2])

#%% Train model
if os.path.isdir(modelPath) == False:
    os.makedirs(modelPath)

if saveCheckpoints:
    if os.path.isdir(os.path.join(modelPath, modelName + '-Checkpoint')) == False:
        os.makedirs(os.path.join(modelPath, modelName + '-Checkpoint'))
    checkpoint = ModelCheckpoint(os.path.join(modelPath, modelName + '-Checkpoint', modelName) + '-Checkpoint-{epoch:03d}.h5')
    callbacks = [checkpoint]
else:
    callbacks = None


history = model.fit([trainingMassProfile1, trainingMassProfile2,
                     trainingPeakProfile1.reshape((training_samples, max_peak_seq_length, 1)),
                     trainingPeakProfile2.reshape((training_samples, max_peak_seq_length, 1)),
                     trainingSequenceProfile1.reshape((training_samples, sequence_length, 1)),
                     trainingSequenceProfile2.reshape((training_samples, sequence_length, 1)),
                     np.abs(trainingTime2 - trainingTime1)],

                     [trainingY] * 4,
                     epochs = epochs,
                     batch_size = batch_size,
                     validation_split = validation_split,
                     initial_epoch = initial_epoch,
                     callbacks = callbacks)


model.save(os.path.join(modelPath, modelName) + '.h5')

if saveHistory:
    histDf = pd.DataFrame(history.history)
    histDf.to_csv(os.path.join(modelPath, modelName) + '-History.csv')


### Predict on test set
prediction = model.predict([testingMassProfile1, testingMassProfile2,
                            testingPeakProfile1.reshape((testing_samples, max_peak_seq_length, 1)),
                            testingPeakProfile2.reshape((testing_samples, max_peak_seq_length, 1)),
                            testingSequenceProfile1.reshape((testing_samples, sequence_length, 1)),
                            testingSequenceProfile2.reshape((testing_samples, sequence_length, 1)),
                            np.abs(testingTime2 - testingTime1)])

prediction = prediction[0]  # Only take the main outcome

wrong = abs(np.round(prediction).ravel() - testingY)
wrongIndex = np.nonzero(wrong)[0]
print('Number wrong:', np.sum(wrong))
# Print some examples of wrong predictions
for i in range(len(wrongIndex)):
    if i > 30: break
    print('Prediction:', prediction[wrongIndex[i]], 'Actual:', testingY[wrongIndex[i]])

