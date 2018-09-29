import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import itertools
import sys
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, Lambda, Concatenate, Subtract, Conv1D, Flatten, MaxPooling1D
from tensorflow.keras.layers import LSTM, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K


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


def loadData(dataPath):
    infoDf = pd.read_csv(os.path.join(dataPath, infoFile), index_col = 0)
    sequenceDf = pd.read_csv(os.path.join(dataPath, sequenceFile), index_col = 0)
    
    ### Load peak and mass slice profiles
    peakFiles = []
    massProfileFiles = []
    for f in os.listdir(dataPath):
        if f.endswith('.txt'):
            peakFiles.append(f)
            
        if f.endswith('.tsv'):
            massProfileFiles.append(f)
    
    peakFiles.sort()
    dfs = []
    for i, file in enumerate(peakFiles):
        df = pd.read_csv(os.path.join(dataPath,file), header = None)
        dfs.append(df)
    peakDf = pd.concat(dfs, axis = 1)
    
    massProfileFiles.sort()
    dfs = []
    for i, file in enumerate(massProfileFiles):
        df = pd.read_csv(os.path.join(dataPath,file), header = None)
        dfs.append(df)
    massProfileDf = pd.concat(dfs, axis = 1)
    
    del dfs
    del df
    
    ### Pre-process Data - Normalise peak height and remove abnormal somples
    peakDf = peakDf - np.min(peakDf)
    peakDf.fillna(0, inplace = True)
    
    peakDdfMax = peakDf.max(axis=0)
    peakDf = peakDf.divide(peakDdfMax, axis=1)
    peakDf = peakDf.transpose()
    peakDf.reset_index(inplace = True, drop = True)
    
    
    massProfileDf = massProfileDf - np.min(massProfileDf)
    massProfileDf.fillna(0, inplace = True)
    
    massProfileDfMax = massProfileDf.max(axis=0)
    massProfileDf = massProfileDf.divide(massProfileDfMax, axis=1)
    massProfileDf = massProfileDf.transpose()
    massProfileDf.reset_index(inplace = True, drop = True)
    
    
    idx = sequenceDf > 0
    sequenceDf[idx] = np.log2(sequenceDf[idx])
    sequenceDf = sequenceDf.transpose()
    
    
    keepIndex = (pd.notnull(peakDf).all(1)) & (pd.notnull(massProfileDf).all(1)) & (infoDf['Group'] >= 0)
    infoDf = infoDf[keepIndex]
    peakDf = peakDf[keepIndex]
    massProfileDf = massProfileDf[keepIndex]
    
    
    print("Dropped rows: {}".format(np.sum(keepIndex == False)))
    
    return infoDf, peakDf, massProfileDf, sequenceDf


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
    infoDf, peakDf, massProfileDf, sequenceDf = loadData(dataPath)
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


def printShapes():
    print('trainingTime1:', trainingTime1.shape)
    print('trainingTime2:', trainingTime2.shape)
    print('trainingPeakProfile1:', trainingPeakProfile1.shape)
    print('trainingPeakProfile2:', trainingPeakProfile2.shape)
    print('trainingMassProfile1:', trainingMassProfile1.shape)
    print('trainingMassProfile2:', trainingMassProfile2.shape)
    print('trainingSequenceProfile1:', trainingSequenceProfile1.shape)
    print('trainingSequenceProfile2:', trainingSequenceProfile2.shape)
    print('trainingY:', trainingY.shape)
    print('---')
    print('testingTime1:', testingTime1.shape)
    print('testingTime2:', testingTime2.shape)
    print('testingPeakProfile1:', testingPeakProfile1.shape)
    print('testingPeakProfile2:', testingPeakProfile2.shape)
    print('testingMassProfile1:', testingMassProfile1.shape)
    print('testingMassProfile2:', testingMassProfile2.shape)
    print('testingSequenceProfile1:', testingSequenceProfile1.shape)
    print('testingSequenceProfile2:', testingSequenceProfile2.shape)
    print('testingY:', testingY.shape)
    print('testingComparisions:', testingComparisions.shape)


def define_model():
    ### Mass profile model
    mass_input_shape = (max_mass_seq_length,)
    mass_left_input = Input(mass_input_shape)
    mass_right_input = Input(mass_input_shape)
    
    mass_encoder = Sequential()
    mass_encoder.add(Dense(64, input_shape = mass_input_shape, activation = 'relu'))
    mass_encoder.add(Dropout(0.2))
    mass_encoder.add(Dense(64, activation = 'relu'))
    mass_encoder.add(Dropout(0.2))
    mass_encoder.add(Dense(10, activation = 'relu'))
    
    mass_encoded_l = mass_encoder(mass_left_input)
    mass_encoded_r = mass_encoder(mass_right_input)
    
    
    # Merge and compute L1 distance
    mass_both = Subtract()([mass_encoded_l, mass_encoded_r])
    mass_both = Lambda(lambda x: K.abs(x))(mass_both)
    mass_prediction = Dense(1, activation='sigmoid', name = 'mass_prediction')(mass_both)
    
    
    ### Peak profile model
    peak_input_shape = (None, 1)  # Variable sequence length
    P_in = Input(peak_input_shape)
    peak_left_input = Input(peak_input_shape)
    peak_right_input = Input(peak_input_shape)
    
    
    P = Bidirectional(LSTM(64, return_sequences = True))(P_in)
    P = Dropout(0.2)(P)
    P = Bidirectional(LSTM(64, return_sequences = True))(P)
    P = Dropout(0.2)(P)
    _, state_h, state_c = LSTM(10, return_sequences = False, return_state = True)(P)
    peak_output = Concatenate(axis = -1)([state_h, state_c])
    peak_output = Dropout(0.2)(peak_output)
    peak_output = Dense(10)(peak_output)
    
    peak_encoder = Model(inputs = P_in, outputs = peak_output)
    
    peak_encoded_l = peak_encoder(peak_left_input)
    peak_encoded_r = peak_encoder(peak_right_input)
    
    
    peak_both = Subtract()([peak_encoded_l, peak_encoded_r])
    peak_both = Lambda(lambda x: K.abs(x))(peak_both)
    peak_prediction = Dense(1, activation='sigmoid', name = 'peak_prediction')(peak_both)
    
    
    ### Surrounding profile model
    surround_input_shape = (sequence_length, 1)  # One channel
    S_in = Input(surround_input_shape)
    surround_left_input = Input(surround_input_shape)
    surround_right_input = Input(surround_input_shape)
    
    # sequence_length of 600
    F1 = Conv1D(filters = 3, kernel_size = 3, strides = 1, padding = 'same', activation = 'relu')(S_in)
    F1 = Conv1D(filters = 3, kernel_size = 3, strides = 1, padding = 'same', activation = 'relu')(F1)
    F1 = MaxPooling1D(3)(F1)
    # sequence_length of 200
    F1 = Conv1D(filters = 6, kernel_size = 3, strides = 1, padding = 'same', activation = 'relu')(F1)
    F1 = Conv1D(filters = 6, kernel_size = 3, strides = 1, padding = 'same', activation = 'relu')(F1)
    F1 = MaxPooling1D(2)(F1)
    # sequence_length of 100
    F1 = Conv1D(filters = 12, kernel_size = 3, strides = 1, padding = 'same', activation = 'relu')(F1)
    F1 = Conv1D(filters = 12, kernel_size = 3, strides = 1, padding = 'same', activation = 'relu')(F1)
    F1 = MaxPooling1D(2)(F1)
    # sequence_length of 50
    F1 = Conv1D(filters = 24, kernel_size = 3, strides = 1, padding = 'same', activation = 'relu')(F1)
    F1 = Conv1D(filters = 24, kernel_size = 3, strides = 1, padding = 'same', activation = 'relu')(F1)
    F1 = MaxPooling1D(2)(F1)
    # sequence_length of 25
    
    F2 = Conv1D(filters = 3, kernel_size = 3, strides = 1, padding = 'same', activation = 'relu')(S_in)
    F2 = MaxPooling1D(3)(F2)  # Sequence length of 200
    F2 = Conv1D(filters = 6, kernel_size = 3, strides = 1, padding = 'same', activation = 'relu')(F2)
    F2 = MaxPooling1D(2)(F2)  # Sequence length of 100
    F2 = Conv1D(filters = 12, kernel_size = 3, strides = 1, padding = 'same', activation = 'relu')(F2)
    F2 = MaxPooling1D(2)(F2)  # Sequence length of 50

    
    F1 = Flatten()(F1)
    F2 = Flatten()(F2)
    surround_output = Concatenate(axis = -1)([F1, F2])
    surround_output = Dropout(0.2)(surround_output)
    surround_output = Dense(10)(surround_output)
    
    surround_encoder = Model(inputs = S_in, outputs = surround_output)
    
    surround_encoded_l = surround_encoder(surround_left_input)
    surround_encoded_r = surround_encoder(surround_right_input)
    
    surround_both = Subtract()([surround_encoded_l, surround_encoded_r])
    surround_both = Lambda(lambda x: K.abs(x))(surround_both)
    surround_prediction = Dense(1, activation='sigmoid', name = 'surround_prediction')(surround_both)    
    
    ### Time model
    
    time_diff = Input((1,))
    
    
    ### Combined model
    combined_outputs = Lambda(lambda x: K.concatenate(x, axis = -1))([mass_both, peak_both, surround_both, time_diff])
    
    combined_outputs = Dropout(0.2)(combined_outputs)  # The time data may be dropped directly
    combined_model = Dense(64, activation = 'relu')(combined_outputs)
    combined_prediction = Dense(1, activation = 'sigmoid', name = 'main_prediction')(combined_model)
    
    
    ### Build and compile
    
    siamese_net = Model(inputs = [mass_left_input, mass_right_input, peak_left_input, peak_right_input,
                                  surround_left_input, surround_right_input, time_diff],
                        outputs = [combined_prediction, mass_prediction, peak_prediction, surround_prediction])
    
    return siamese_net


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
    model = define_model()
    initial_epoch = 1


model.compile(optimizer = Adam(**adamOptimizerOptions),
              loss = 'binary_crossentropy',
              metrics = ['accuracy'],
              loss_weights = [1., 0.2, 0.2, 0.2])

#%% Fit model
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

