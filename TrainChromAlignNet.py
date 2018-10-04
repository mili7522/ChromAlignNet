import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# import importlib
import time
import sys
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.optimizers import Adam
from utils import loadData, printShapes, getChromatographSegmentDf, generateCombinationIndices
from model_definition import getModelVariant
from parameters import training_options

# Load parameters
epochs = training_options.get('epochs')
batch_size = training_options.get('batch_size')
test_split = training_options.get('test_split')
validation_split = training_options.get('validation_split')
adam_optimizer_options = training_options.get('adam_optimizer_options')
train_with_gpu = training_options.get('train_with_gpu')
random_seed_type = training_options.get('random_seed_type')

save_checkpoints = training_options.get('save_checkpoints')  # If checkpoints exist, training will resume from the last checkpoint
save_history = training_options.get('save_history')
model_path = training_options.get('model_path')
model_name = training_options.get('model_name')

datasets = training_options.get('datasets')
dataset_for_model = training_options.get('dataset_for_model')
info_file = training_options.get('info_file')
sequence_file = training_options.get('sequence_file')


# Modify the model name for different data sources, model variants and repetitions
if len(sys.argv) > 1:
    assert sys.argv[1] in ('A', 'B', 'C', 'D', 'E', 'F'), "Dataset selection needs to be a letter between A and F"
    dataset_selection = sys.argv[1]
else:
    dataset_selection = 'A'
if len(sys.argv) > 2:
    try:
        repetition = int(sys.argv[2])
    except:
        print("Repetition needs to be a number", file=sys.stderr)
        raise
else:
    repetition = 1
if len(sys.argv) > 3:
    try:
        model_variant = int(sys.argv[3])
    except:
        print("Model variant needs to be a number (1 to 27)", file=sys.stderr)
        raise
else:
    model_variant = 1


chrom_align_model = getModelVariant(model_variant)
buildModel = getattr(chrom_align_model, 'buildModel')
ignore_peak_profile = getattr(chrom_align_model, 'ignore_peak_profile')

model_name = model_name + '-' + dataset_selection + '-{:02d}'.format(model_variant) + '-r' + str(repetition).rjust(2, '0')

print("Output model to: ", model_name)

data_paths = list( datasets[i] for i in dataset_for_model[dataset_selection] )

random_seed = int(ord(dataset_selection) * 1E4 + model_variant * 1E2 + repetition)
if random_seed_type == 2:
    random_seed = random_seed + int(time.time())
with open(os.path.join(model_path, model_name) + '-RandomSeed.txt', 'w') as f:
  f.write('%d' % random_seed)


if train_with_gpu:
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)


### Execute load for all folders
data_time_1 = []
data_time_2 = []
if not ignore_peak_profile:
    data_peak_1 = []
    data_peak_2 = []
data_mass_spectrum_1 = []
data_mass_spectrum_2 = []
data_chrom_seg_1 = []
data_chrom_seg_2 = []
data_y = []


for data_path in data_paths:
    print('Loading:', data_path)
    info_df, peak_df, mass_profile_df, chromatogram_df, _, _ = loadData(data_path, info_file, sequence_file)

    # Remove null rows and negative indexed groups
    keep_index = (pd.notnull(mass_profile_df).all(1)) & (info_df['Group'] >= 0)
    if not ignore_peak_profile:
        keep_index = keep_index & (pd.notnull(peak_df).all(1))
    info_df = info_df[keep_index]
    peak_df = peak_df[keep_index]
    mass_profile_df = mass_profile_df[keep_index]
    chrom_seg_df = getChromatographSegmentDf(info_df, chromatogram_df, sequence_length = 600)
    print("Dropped rows: {}".format(np.sum(keep_index == False)))

    a = len(data_y)
    x1, x2, y = generateCombinationIndices(info_df, time_cutoff = None, return_y = True, random_seed = random_seed)
    data_time_1.extend(info_df.loc[x1]['peakMaxTime'])
    data_time_2.extend(info_df.loc[x2]['peakMaxTime'])
    if not ignore_peak_profile:
        data_peak_1.append(peak_df.loc[x1])
        data_peak_2.append(peak_df.loc[x2])
    data_mass_spectrum_1.append(mass_profile_df.loc[x1])
    data_mass_spectrum_2.append(mass_profile_df.loc[x2])
    data_chrom_seg_1.append(chrom_seg_df.loc[x1])
    data_chrom_seg_2.append(chrom_seg_df.loc[x2])
    data_y.extend(y)
    print(len(data_y) - a, 'combinations generated')
    
### Shuffle data
np.random.seed(random_seed)
shuffle_index = np.random.permutation(len(data_time_1))
data_time_1 = np.array(data_time_1)[shuffle_index]
data_time_2 = np.array(data_time_2)[shuffle_index]
if not ignore_peak_profile:
    data_peak_1 = pd.concat(data_peak_1, axis = 0)  # Use pd to concat since the DataSeries might be of different lengths
    data_peak_1.fillna(0, inplace = True)
    data_peak_1 = data_peak_1.values[shuffle_index]
    data_peak_2 = pd.concat(data_peak_2, axis = 0)
    data_peak_2.fillna(0, inplace = True)
    data_peak_2 = data_peak_2.values[shuffle_index]
data_mass_spectrum_1 = np.array(pd.concat(data_mass_spectrum_1))[shuffle_index]
data_mass_spectrum_2 = np.array(pd.concat(data_mass_spectrum_2))[shuffle_index]
data_chrom_seg_1 = np.array(pd.concat(data_chrom_seg_1))[shuffle_index]
data_chrom_seg_2 = np.array(pd.concat(data_chrom_seg_2))[shuffle_index]
data_y = np.array(data_y)[shuffle_index]


### Split into training and test sets
training = int(len(data_time_1) * (1-test_split))

train_time_1 = data_time_1[:training]
train_time_2 = data_time_2[:training]
if not ignore_peak_profile:
    train_peak_1 = data_peak_1[:training]
    train_peak_2 = data_peak_2[:training]
train_mass_spectrum_1 = data_mass_spectrum_1[:training]
train_mass_spectrum_2 = data_mass_spectrum_2[:training]
train_chrom_seg_1 = data_chrom_seg_1[:training]
train_chrom_seg_2 = data_chrom_seg_2[:training]
train_y = data_y[:training]

test_time_1 = data_time_1[training:]
test_time_2 = data_time_2[training:]
if not ignore_peak_profile:
    test_peak_1 = data_peak_1[training:]
    test_peak_2 = data_peak_2[training:]
test_mass_spectrum_1 = data_mass_spectrum_1[training:]
test_mass_spectrum_2 = data_mass_spectrum_2[training:]
test_chrom_seg_1 = data_chrom_seg_1[training:]
test_chrom_seg_2 = data_chrom_seg_2[training:]
test_y = data_y[training:]

if not ignore_peak_profile:
    samples, max_peak_seq_length = data_peak_1.shape
samples, sequence_length = data_chrom_seg_1.shape
testing_samples, max_mass_seq_length = test_mass_spectrum_1.shape
training_samples = training

print('Number of samples:', samples)
print('Number of training samples:', training_samples)
print('Number of testing samples:', testing_samples)
if not ignore_peak_profile:
    print('Max peak length:', max_peak_seq_length)
print('Sequence length:', sequence_length)
print('Max mass sequence length:', max_mass_seq_length)



# Check if existing checkpoints are present - if so then load and resume training from last epoch
checkpoint_path = os.path.join(model_path, model_name + '-Checkpoint')
if os.path.isdir(checkpoint_path) and os.listdir(checkpoint_path):
    files = []
    for f in os.listdir(checkpoint_path):
        if f.endswith('.h5'):
            files.append(os.path.join(checkpoint_path, f))
    last_checkpoint = sorted(files)[-1]
    model = load_model(last_checkpoint)
    initial_epoch = int(last_checkpoint[-6:-3])
else:
    model = buildModel(max_mass_seq_length, sequence_length)
    initial_epoch = 0



model.compile(optimizer = Adam(**adam_optimizer_options),
              loss = 'binary_crossentropy',
              metrics = ['accuracy'],
              loss_weights = [1] + [0.2] * (2 if ignore_peak_profile else 3))

#%% Train model
if os.path.isdir(model_path) == False:
    os.makedirs(model_path)

if save_history:
    logger = CSVLogger(os.path.join(model_path, model_name) + '-History.csv', separator = ',', append = True)
else:
    logger = None
if save_checkpoints:
    if os.path.isdir(os.path.join(model_path, model_name + '-Checkpoint')) == False:
        os.makedirs(os.path.join(model_path, model_name + '-Checkpoint'))
    checkpoint = ModelCheckpoint(os.path.join(model_path, model_name + '-Checkpoint', model_name) + '-Checkpoint-{epoch:03d}.h5')
    
    callbacks = [checkpoint] + ([logger] if logger is not None else [])
else:
    callbacks = [logger] if logger is not None else None

if ignore_peak_profile:
    training_data = [train_mass_spectrum_1, train_mass_spectrum_2,
                    train_chrom_seg_1.reshape((training_samples, sequence_length, 1)),
                    train_chrom_seg_2.reshape((training_samples, sequence_length, 1)),
                    np.abs(train_time_2 - train_time_1)]
else:
    training_data = [train_mass_spectrum_1, train_mass_spectrum_2,
                    train_peak_1.reshape((training_samples, max_peak_seq_length, 1)),
                    train_peak_2.reshape((training_samples, max_peak_seq_length, 1)),
                    train_chrom_seg_1.reshape((training_samples, sequence_length, 1)),
                    train_chrom_seg_2.reshape((training_samples, sequence_length, 1)),
                    np.abs(train_time_2 - train_time_1)]

history = model.fit(training_data,
                    [train_y] * (3 if ignore_peak_profile else 4),
                    epochs = epochs,
                    batch_size = batch_size,
                    validation_split = validation_split,
                    initial_epoch = initial_epoch,
                    callbacks = callbacks)


model.save(os.path.join(model_path, model_name) + '.h5')


if ignore_peak_profile:
    prediction_data = [test_mass_spectrum_1, test_mass_spectrum_2,
                        test_chrom_seg_1.reshape((testing_samples, sequence_length, 1)),
                        test_chrom_seg_2.reshape((testing_samples, sequence_length, 1)),
                        np.abs(test_time_2 - test_time_1)]
else:
    prediction_data = [test_mass_spectrum_1, test_mass_spectrum_2,
                        test_peak_1.reshape((testing_samples, max_peak_seq_length, 1)),
                        test_peak_2.reshape((testing_samples, max_peak_seq_length, 1)),
                        test_chrom_seg_1.reshape((testing_samples, sequence_length, 1)),
                        test_chrom_seg_2.reshape((testing_samples, sequence_length, 1)),
                        np.abs(test_time_2 - test_time_1)]

### Predict on test set
prediction = model.predict(prediction_data)

prediction = prediction[0]  # Only take the main outcome

wrong = abs(np.round(prediction).ravel() - test_y)
wrongIndex = np.nonzero(wrong)[0]
print('Number wrong:', np.sum(wrong))
# Print some examples of wrong predictions
for i in range(len(wrongIndex)):
    if i > 30: break
    print('Prediction:', prediction[wrongIndex[i]], 'Actual:', test_y[wrongIndex[i]])

