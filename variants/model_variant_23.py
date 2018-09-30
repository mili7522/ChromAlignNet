import tensorflow.keras.backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, Lambda, Concatenate, Subtract, Conv1D, Flatten, MaxPooling1D, GRU

## Variant 3 + 13
## Peak encoder simplified
## Dropout on CNN encoder (end only) increased to 50%. Encoder neurons increased to 30

ignorePeakProfile = False

def define_model(max_mass_seq_length, sequence_length):
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

    P = GRU(32, return_sequences = True)(P_in)
    P = Dropout(0.2)(P)
    P = GRU(32, return_sequences = True)(P)
    P = Dropout(0.2)(P)
    _, state_h = GRU(10, return_sequences = False, return_state = True)(P)
    peak_output = Dropout(0.2)(state_h)
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
    surround_output = Dropout(0.5)(surround_output)
    surround_output = Dense(30)(surround_output)
    
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