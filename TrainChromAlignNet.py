import pandas as pd
import numpy as np
import time
import sys
import os
import tensorflow as tf
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.optimizers import Adam
from utilsData import loadData, getChromatographSegmentDf, generateCombinationIndices
from modelDefinition import getModelVariant
from parameters import training_options

### Load parameters
epochs = training_options['epochs']  # Number of iterations through the training set
batch_size = training_options['batch_size']  # Number of training examples per mini-batch
validation_split = training_options['validation_split']  # Ratio of the data set to be used for validation
verbose_training = training_options['verbose_training']  # 0 = silent, 1 = progress bar, 2 = one line per epoch 
adam_optimizer_options = training_options['adam_optimizer_options']  # Optimiser options
random_seed_type = training_options['random_seed_type']  # How the random seed is generated. 1 = from model name, 2 = clock time, 3 = load from file

save_checkpoints = training_options['save_checkpoints']  # If True, checkpoints are saved. If checkpoints exist, training will resume from the last checkpoint
save_history = training_options['save_history']  # If True, the loss and accuracy of the model over epoches will be saved as a csv file
model_path = training_options['model_path']  # Folder to save the model
model_name = training_options['model_name']  # Name prefix given to the saved model. ie 'ChromAlignNet'

datasets = training_options['datasets']  # List of paths to data sets
dataset_for_model = training_options['dataset_for_model']  # Dictionary to select which preset list of data sets are used in training
info_file = training_options['info_file']  # Name of csv file containing overall information about each peak
sequence_file = training_options['sequence_file']  # Name of csv file containing the data used to extract the chromatogram segments
ignore_negatives = training_options['ignore_negatives']  # If True, groups assigned with a negative index will be ignored in training


# Three parameters are taken from the system input
# Argument 1: selects the data sets the model will be trained on from the options in 'dataset_for_model'
# Argument 2: specifies the repetition number the model will be saved as
# Argument 3: selects the model variant (as defined in modelDefinition.py)
if len(sys.argv) > 1:
    assert sys.argv[1] in dataset_for_model, "Data set selection needs to be a letter between A and H"
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
        print("Model variant needs to be a number (1 to 31)", file=sys.stderr)
        raise
else:
    model_variant = 1

# Modify the save name for the model based on the data sets, model variant and repetition
model_name = model_name + '-' + dataset_selection + '-{:02d}'.format(model_variant) + '-r' + str(repetition).rjust(2, '0')
print("Output model to: ", model_name)

os.makedirs(model_path, exist_ok = True)  # Make the model_path if it doesn't exist


# Get the correct model variant and extract the buildModel method and the ignore_peak_profile attribute
chrom_align_model = getModelVariant(model_variant)
buildModel = getattr(chrom_align_model, 'buildModel')
ignore_peak_profile = getattr(chrom_align_model, 'ignore_peak_profile')


# Get the list of data sets for training based on what is given in 'dataset_for_model'
data_paths = list( datasets[i] for i in dataset_for_model[dataset_selection] )

# Set and save the random seed
random_seed = int(ord(dataset_selection) * 1E4 + model_variant * 1E2 + repetition)  # Produces a number based on the data set selection, model variant and repetition
if random_seed_type == 2:
    random_seed = random_seed + int(time.time())
if random_seed_type == 3:
    with open(os.path.join(model_path, model_name) + '-RandomSeed.txt', 'r') as f:
        lines = f.readlines()
        random_seed = lines[-1][:-1]  # Ignore last \n character

with open(os.path.join(model_path, model_name) + '-RandomSeed.txt', 'a') as f:
    f.write('%d\n' % random_seed)

# Configure the GPU if it is used for training
if tf.__version__.startswith('1') and tf.test.is_gpu_available():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)



### Load the data from all data set folders
# Initialise
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

# load the data sets one folder at a time and combines them together
for data_path in data_paths:
    print('Loading:', data_path)
    info_df, peak_df, mass_profile_df, chromatogram_df, _, _ = loadData(data_path, info_file, sequence_file)

    # Remove null rows and peaks from negative indexed groups (if 'ignore_negatives' is True)
    keep_index = (pd.notnull(mass_profile_df).all(1))
    if ignore_negatives:
        keep_index = keep_index & (info_df['Group'] >= 0)
    if not ignore_peak_profile:
        keep_index = keep_index & (pd.notnull(peak_df).all(1))
    print("Dropped rows: {}".format(np.sum(keep_index == False)))

    # Extract segments of the chromatogram centred at each peak
    chrom_seg_df = getChromatographSegmentDf(info_df, chromatogram_df, segment_length = 600)
    
    # Generate indices corresponding to positive and negative combinations of peaks
    x1, x2, y = generateCombinationIndices(info_df[keep_index], time_cutoff = None, return_y = True, random_seed = random_seed)
    
    
    prev_len = len(data_y)
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
    print(len(data_y) - prev_len, 'combinations generated')
    
# Shuffle the data and create numpy arrays out of lists of pandas DataFrames
np.random.seed(random_seed)
shuffle_index = np.random.permutation(len(data_time_1))
data_time_1 = np.array(data_time_1)[shuffle_index]
data_time_2 = np.array(data_time_2)[shuffle_index]
if not ignore_peak_profile:
    data_peak_1 = pd.concat(data_peak_1, axis = 0)  # The peak profile DataFrames from different data sets might be of different lengths, so need to concatonate carefully
    data_peak_1.fillna(0, inplace = True)  # Turn any nan values (from concatonating together DataFrames of different lengths) into 0s
    data_peak_1 = data_peak_1.values[shuffle_index]
    data_peak_2 = pd.concat(data_peak_2, axis = 0)
    data_peak_2.fillna(0, inplace = True)
    data_peak_2 = data_peak_2.values[shuffle_index]
data_mass_spectrum_1 = np.array(pd.concat(data_mass_spectrum_1))[shuffle_index]
data_mass_spectrum_2 = np.array(pd.concat(data_mass_spectrum_2))[shuffle_index]
data_chrom_seg_1 = np.array(pd.concat(data_chrom_seg_1))[shuffle_index]
data_chrom_seg_2 = np.array(pd.concat(data_chrom_seg_2))[shuffle_index]
data_y = np.array(data_y)[shuffle_index]
data_time_diff = np.abs(data_time_2 - data_time_1)

# Print data dimensions
samples, segment_length = data_chrom_seg_1.shape
_, max_mass_seq_length = data_mass_spectrum_1.shape
print('Number of samples:', samples)
if not ignore_peak_profile:
    _, max_peak_seq_length = data_peak_1.shape
    print('Max peak length:', max_peak_seq_length)
print('Mass spectrum length:', max_mass_seq_length)
print('Chromatogram segment length:', segment_length)
sys.stdout.flush()

# Put all the training data into a list to pass into the model
training_data = [data_mass_spectrum_1, data_mass_spectrum_2,
                 data_chrom_seg_1.reshape((samples, segment_length, 1)),
                 data_chrom_seg_2.reshape((samples, segment_length, 1)),
                 data_time_diff]

if not ignore_peak_profile:  # Insert the peak data if it is used
    training_data[2:2] = [data_peak_1.reshape((samples, max_peak_seq_length, 1)),
                          data_peak_2.reshape((samples, max_peak_seq_length, 1))]




### Create and compile the model
# Check if existing checkpoints are present - if so then load and resume training from last epoch
checkpoint_path = os.path.join(model_path, model_name + '-Checkpoint')
if os.path.isdir(checkpoint_path) and os.listdir(checkpoint_path):
    files = []
    for f in os.listdir(checkpoint_path):
        if f.endswith('.h5'):
            files.append(os.path.join(checkpoint_path, f))
    last_checkpoint = sorted(files)[-1]  # Find the checkpoint with the largest epoch number
    model = load_model(last_checkpoint)  # Load the model
    initial_epoch = int(last_checkpoint[-6:-3])  # Set the initial epoch value
else:
    model = buildModel(max_mass_seq_length, segment_length)  # Call the buildModel method from the obtained model variant
    initial_epoch = 0

# Compile the model
model.compile(optimizer = Adam(**adam_optimizer_options),
              loss = 'binary_crossentropy',
              metrics = ['accuracy'],
              loss_weights = [1] + [0.2] * (2 if ignore_peak_profile else 3))


### Train the model
# Create the history csv and checkpoint callbacks
if save_history:
    logger = CSVLogger(os.path.join(model_path, model_name) + '-History.csv', separator = ',', append = True)
else:
    logger = None
if save_checkpoints:
    os.makedirs(os.path.join(model_path, model_name + '-Checkpoint'), exist_ok = True)  # Create folder if not present
    checkpoint = ModelCheckpoint(os.path.join(model_path, model_name + '-Checkpoint', model_name) + '-Checkpoint-{epoch:03d}.h5')
    
    callbacks = [checkpoint] + ([logger] if logger is not None else [])
else:
    callbacks = [logger] if logger is not None else None

# Training
history = model.fit(training_data,
                    [data_y] * (3 if ignore_peak_profile else 4),
                    epochs = epochs,
                    batch_size = batch_size,
                    validation_split = validation_split,
                    verbose = verbose_training,   
                    initial_epoch = initial_epoch,
                    callbacks = callbacks)

### Save the model
model.save(os.path.join(model_path, model_name) + '.h5')