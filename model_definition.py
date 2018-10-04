import tensorflow.keras.backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, Lambda, Concatenate, Subtract, Conv1D, Flatten, MaxPooling1D, LSTM, Bidirectional, GRU


class ChromAlignModel:
    def __init__(self, mass_network_neurons = 64, peak_network_neurons = 64, chromatogram_network_neurons = 64,
                 mass_encoder_neurons = 10, peak_encoder_neurons = 10, chromatogram_encoder_neurons = 10,
                 mass_dropout_percentage = 0.2, peak_dropout_percentage = 0.2, chromatogram_dropout_percentage = 0.2,
                 number_of_left_convolution_stacks = 4, number_of_right_convolution_stacks = 3,
                 chromatogram_convolution_dropout_percentage = 0, combined_output_neurons = 64,
                 combined_output_dropout_percentage = 0.2, ignore_peak_profile = False):
        self.mass_network_neurons = mass_network_neurons
        self.peak_network_neurons = peak_network_neurons
        self.chromatogram_network_neurons = chromatogram_network_neurons
        self.mass_encoder_neurons = mass_encoder_neurons
        self.peak_encoder_neurons = peak_encoder_neurons
        self.chromatogram_encoder_neurons = chromatogram_encoder_neurons
        self.mass_dropout_percentage = mass_dropout_percentage
        self.peak_dropout_percentage = peak_dropout_percentage
        self.chromatogram_dropout_percentage = chromatogram_dropout_percentage
        self.number_of_left_convolution_stacks = number_of_left_convolution_stacks
        self.number_of_right_convolution_stacks = number_of_right_convolution_stacks
        self.chromatogram_convolution_dropout_percentage = chromatogram_convolution_dropout_percentage
        self.combined_output_neurons = combined_output_neurons
        self.combined_output_dropout_percentage = combined_output_dropout_percentage
        self.ignore_peak_profile = ignore_peak_profile

    def makeSiameseComponent(self, encoder_model, left_input, right_input):
        left_branch = encoder_model(left_input)
        right_branch = encoder_model(right_input)
        
        # Merge and compute L1 distance
        comparison = Subtract()([left_branch, right_branch])
        comparison = Lambda(lambda x: K.abs(x))(comparison)

        return comparison

    def buildMassEncoder(self, max_mass_seq_length):
        mass_input_shape = (max_mass_seq_length,)
        mass_left_input = Input(mass_input_shape)
        mass_right_input = Input(mass_input_shape)

        mass_encoder = Sequential()
        mass_encoder.add(Dense(self.mass_network_neurons, input_shape = mass_input_shape, activation = 'relu'))
        mass_encoder.add(Dropout(self.mass_dropout_percentage))
        mass_encoder.add(Dense(self.mass_network_neurons, activation = 'relu'))
        mass_encoder.add(Dropout(self.mass_dropout_percentage))
        mass_encoder.add(Dense(self.mass_encoder_neurons, activation = 'relu'))

        mass_comparison = self.makeSiameseComponent(mass_encoder, mass_left_input, mass_right_input)
        mass_prediction = Dense(1, activation = 'sigmoid', name = 'mass_prediction')(mass_comparison)

        return mass_left_input, mass_right_input, mass_comparison, mass_prediction

    def buildPeakEncoder(self):
        peak_input_shape = (None, 1)  # Variable sequence length
        P_in = Input(peak_input_shape)
        peak_left_input = Input(peak_input_shape)
        peak_right_input = Input(peak_input_shape)
        
        P = Bidirectional(LSTM(self.peak_network_neurons, return_sequences = True))(P_in)
        P = Dropout(self.peak_dropout_percentage)(P)
        P = Bidirectional(LSTM(self.peak_network_neurons, return_sequences = True))(P)
        P = Dropout(self.peak_dropout_percentage)(P)
        _, state_h, state_c = LSTM(self.peak_encoder_neurons, return_sequences = False, return_state = True)(P)
        peak_output = Concatenate(axis = -1)([state_h, state_c])
        peak_output = Dropout(self.peak_dropout_percentage)(peak_output)
        peak_output = Dense(self.peak_encoder_neurons)(peak_output)
        
        peak_encoder = Model(inputs = P_in, outputs = peak_output)

        peak_comparison = self.makeSiameseComponent(peak_encoder, peak_left_input, peak_right_input)
        peak_prediction = Dense(1, activation = 'sigmoid', name = 'peak_prediction')(peak_comparison)

        return peak_left_input, peak_right_input, peak_comparison, peak_prediction

    def buildChromatogramEncoder(self, sequence_length):
        chromatogram_input_shape = (sequence_length, 1)  # One channel input
        C_in = Input(chromatogram_input_shape)
        chromatogram_left_input = Input(chromatogram_input_shape)
        chromatogram_right_input = Input(chromatogram_input_shape)
        
        # Sequence_length of 600, 200, 100, 50, etc
        initial_filter_number = 3
        for i in range(self.number_of_left_convolution_stacks):
            F1 = Conv1D(filters = initial_filter_number * (2**i), kernel_size = 3, strides = 1, padding = 'same', activation = 'relu')(C_in if i == 0 else F1)
            F1 = Conv1D(filters = initial_filter_number * (2**i), kernel_size = 3, strides = 1, padding = 'same', activation = 'relu')(F1)
            F1 = MaxPooling1D(3 if i == 0 else 2)(F1)
            if (self.chromatogram_convolution_dropout_percentage != 0) and (i != (self.number_of_left_convolution_stacks - 1)):
                F1 = Dropout(self.chromatogram_convolution_dropout_percentage)(F1)
        
        # Sequence length of 600, 200, 100, 50, etc
        for i in range(self.number_of_right_convolution_stacks):
            F2 = Conv1D(filters = initial_filter_number * (2**i), kernel_size = 3, strides = 1, padding = 'same', activation = 'relu')(C_in if i == 0 else F2)
            F2 = MaxPooling1D(3 if i == 0 else 2)(F2)
            if (self.chromatogram_convolution_dropout_percentage != 0) and (i != (self.number_of_right_convolution_stacks - 1)):
                F2 = Dropout(self.chromatogram_convolution_dropout_percentage)(F2)

        F1 = Flatten()(F1)
        F2 = Flatten()(F2)
        chromatogram_output = Concatenate(axis = -1)([F1, F2])
        chromatogram_output = Dropout(self.chromatogram_dropout_percentage)(chromatogram_output)
        chromatogram_output = Dense(self.chromatogram_encoder_neurons)(chromatogram_output)
        
        chromatogram_encoder = Model(inputs = C_in, outputs = chromatogram_output)
        
        chromatogram_comparison = self.makeSiameseComponent(chromatogram_encoder, chromatogram_left_input, chromatogram_right_input)
        chromatogram_prediction = Dense(1, activation = 'sigmoid', name = 'chromatogram_prediction')(chromatogram_comparison)    
        
        return chromatogram_left_input, chromatogram_right_input, chromatogram_comparison, chromatogram_prediction


    def buildModel(self, max_mass_seq_length, sequence_length):

        mass_left_input, mass_right_input, mass_comparison, mass_prediction = self.buildMassEncoder(max_mass_seq_length)
        if not self.ignore_peak_profile:
            peak_left_input, peak_right_input, peak_comparison, peak_prediction = self.buildPeakEncoder()
        chromatogram_left_input, chromatogram_right_input, chromatogram_comparison, chromatogram_prediction = self.buildChromatogramEncoder(sequence_length)
        time_diff = Input((1,))

        if self.ignore_peak_profile:
            components = [mass_comparison, chromatogram_comparison, time_diff]
        else:
            components = [mass_comparison, peak_comparison, chromatogram_comparison, time_diff]

        combined_outputs = Lambda(lambda x: K.concatenate(x, axis = -1))(components)
        combined_outputs = Dropout(self.combined_output_dropout_percentage)(combined_outputs)
        combined_model = Dense(self.combined_output_neurons, activation = 'relu')(combined_outputs)
        combined_prediction = Dense(1, activation = 'sigmoid', name = 'main_prediction')(combined_model)

        if self.ignore_peak_profile:
            inputs = [mass_left_input, mass_right_input, chromatogram_left_input, chromatogram_right_input, time_diff]
            outputs = [combined_prediction, mass_prediction, chromatogram_prediction]
        else:
            inputs = [mass_left_input, mass_right_input, peak_left_input, peak_right_input, chromatogram_left_input, chromatogram_right_input, time_diff]
            outputs = [combined_prediction, mass_prediction, peak_prediction, chromatogram_prediction]

        siamese_net = Model(inputs = inputs, outputs = outputs)

        return siamese_net


class SimplifiedPeakEncoderVariant(ChromAlignModel):
    def buildPeakEncoder(self):
        peak_input_shape = (None, 1)  # Variable sequence length
        P_in = Input(peak_input_shape)
        peak_left_input = Input(peak_input_shape)
        peak_right_input = Input(peak_input_shape)
        
        P = GRU(self.peak_network_neurons, return_sequences = True)(P_in)
        P = Dropout(self.peak_dropout_percentage)(P)
        P = GRU(self.peak_network_neurons, return_sequences = True)(P)
        P = Dropout(self.peak_dropout_percentage)(P)
        _, state_h = GRU(self.peak_encoder_neurons, return_sequences = False, return_state = True)(P)

        peak_output = Dropout(self.peak_dropout_percentage)(state_h)
        peak_output = Dense(self.peak_encoder_neurons)(peak_output)

        peak_encoder = Model(inputs = P_in, outputs = peak_output)

        peak_comparison = self.makeSiameseComponent(peak_encoder, peak_left_input, peak_right_input)
        peak_prediction = Dense(1, activation = 'sigmoid', name = 'peak_prediction')(peak_comparison)

        return peak_left_input, peak_right_input, peak_comparison, peak_prediction


def getModelVariant(variant):
    switcher = {
        1: ChromAlignModel(),
        2: ChromAlignModel(ignore_peak_profile = True),
        3: SimplifiedPeakEncoderVariant(peak_network_neurons = 32),
        4: ChromAlignModel(peak_dropout_percentage = 0.5),
        5: ChromAlignModel(peak_encoder_neurons = 5),
        6: ChromAlignModel(peak_dropout_percentage = 0.5, peak_encoder_neurons = 20),
        7: ChromAlignModel(peak_dropout_percentage = 0.5, peak_encoder_neurons = 30),
        8: ChromAlignModel(mass_encoder_neurons = 5),
        9: ChromAlignModel(mass_dropout_percentage = 0.5, mass_encoder_neurons = 20),
        10: ChromAlignModel(mass_dropout_percentage = 0.5, mass_encoder_neurons = 30),
        11: ChromAlignModel(chromatogram_encoder_neurons = 5),
        12: ChromAlignModel(chromatogram_dropout_percentage = 0.5, chromatogram_encoder_neurons = 20),
        13: ChromAlignModel(chromatogram_dropout_percentage = 0.5, chromatogram_encoder_neurons = 30),
        14: ChromAlignModel(chromatogram_convolution_dropout_percentage = 0.2),
        15: ChromAlignModel(chromatogram_convolution_dropout_percentage = 0.5),
        16: ChromAlignModel(number_of_left_convolution_stacks = 5),
        17: ChromAlignModel(number_of_left_convolution_stacks = 3),
        18: ChromAlignModel(number_of_right_convolution_stacks = 4),
        19: ChromAlignModel(number_of_right_convolution_stacks = 2),
        20: ChromAlignModel(chromatogram_encoder_neurons = 5, ignore_peak_profile = True),
        21: SimplifiedPeakEncoderVariant(peak_network_neurons = 32, chromatogram_encoder_neurons = 5),
        22: ChromAlignModel(chromatogram_dropout_percentage = 0.5, chromatogram_encoder_neurons = 30, ignore_peak_profile = True),
        23: SimplifiedPeakEncoderVariant(peak_network_neurons = 32, chromatogram_dropout_percentage = 0.5, chromatogram_encoder_neurons = 30),
        24: ChromAlignModel(chromatogram_encoder_neurons = 5, ignore_peak_profile = True, number_of_left_convolution_stacks = 5),
        25: SimplifiedPeakEncoderVariant(peak_network_neurons = 32, chromatogram_encoder_neurons = 5, number_of_left_convolution_stacks = 5),
        26: ChromAlignModel(chromatogram_dropout_percentage = 0.5, chromatogram_encoder_neurons = 30, ignore_peak_profile = True, number_of_left_convolution_stacks = 5),
        27: SimplifiedPeakEncoderVariant(peak_network_neurons = 32, chromatogram_dropout_percentage = 0.5, chromatogram_encoder_neurons = 30, number_of_left_convolution_stacks = 5)
    }
    assert variant in switcher, "Model variant does not exist. Check the integer given"
    return switcher[variant]