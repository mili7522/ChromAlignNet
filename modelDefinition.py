import keras.backend as K
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Dropout, Lambda, Concatenate, Subtract, Conv1D, Flatten, MaxPooling1D, LSTM, Bidirectional, GRU


class ChromAlignModel:
    """
    Model definitions for ChromAlignNet. Specified and built using Keras
    """
    def __init__(self, mass_network_neurons = 64, peak_network_neurons = 64, initial_convolution_filter_number = 3,
                 mass_encoder_neurons = 10, peak_encoder_neurons = 10, chromatogram_encoder_neurons = 10,
                 mass_dropout_percentage = 0.2, peak_dropout_percentage = 0.2, chromatogram_dropout_percentage = 0.2,
                 number_of_left_convolution_stacks = 4, number_of_right_convolution_stacks = 3,
                 chromatogram_convolution_dropout_percentage = 0, combined_output_neurons = 64,
                 combined_output_dropout_percentage = 0.2, ignore_peak_profile = False):
        self.mass_network_neurons = mass_network_neurons  # Number of neurons in the first two layers of the mass encoder
        self.peak_network_neurons = peak_network_neurons  # Number of units in the LSTM of the peak encoder. Gives the dimensionality of the output space
        self.initial_convolution_filter_number = initial_convolution_filter_number  # Number of output neurons in the first convolutional layer of the chromatogram encoder. This number doubles per convolutional layer
        self.mass_encoder_neurons = mass_encoder_neurons  # Number of neurons in the final layer of the mass encoder
        self.peak_encoder_neurons = peak_encoder_neurons  # Number of neurons in the final layer of the peak encoder
        self.chromatogram_encoder_neurons = chromatogram_encoder_neurons  # Number of neurons in the final layer of the chromatogram encoder
        self.mass_dropout_percentage = mass_dropout_percentage  # The value of dropout applied between each layer of the mass encoder
        self.peak_dropout_percentage = peak_dropout_percentage  # The value of dropout applied between each LSTM layer of the peak encoder
        self.chromatogram_dropout_percentage = chromatogram_dropout_percentage  # The value of dropout applied after the concatonation of the two convolution stacks
        self.number_of_left_convolution_stacks = number_of_left_convolution_stacks  # The number of CONV - CONV - MAXPOOL sequences in the left convolution stack
        self.number_of_right_convolution_stacks = number_of_right_convolution_stacks  # The number of CONV - MAXPOOL layers sequences in the right convolution stack
        self.chromatogram_convolution_dropout_percentage = chromatogram_convolution_dropout_percentage  # The value of dropout applied after each max pooling operation within each convolution stack
        self.combined_output_neurons = combined_output_neurons  # The number of neurons in the fully connected layer after combining each encoder and the time-based peak information
        self.combined_output_dropout_percentage = combined_output_dropout_percentage  # The value of dropout applied after combining each encoder and the time-based peak information
        self.ignore_peak_profile = ignore_peak_profile  # If True, the peak encoder is not included as part of the model

    def makeSiameseComponent(self, encoder_model, left_input, right_input):
        """
        Creates a Siamese sub-network given an encoder model and an input for each of the two branches
        """
        left_branch = encoder_model(left_input)
        right_branch = encoder_model(right_input)
        
        # Merge and compute the element-wise absolute distance
        comparison = Subtract()([left_branch, right_branch])
        comparison = Lambda(lambda x: K.abs(x))(comparison)

        return comparison

    def buildMassEncoder(self, mass_input_shape):
        """
        Creates a model for a mass encoder with three fully connected layers and dropout between each layer
        """
        mass_encoder = Sequential()
        mass_encoder.add(Dense(self.mass_network_neurons, input_shape = mass_input_shape, activation = 'relu'))
        mass_encoder.add(Dropout(self.mass_dropout_percentage))
        mass_encoder.add(Dense(self.mass_network_neurons, activation = 'relu'))
        mass_encoder.add(Dropout(self.mass_dropout_percentage))
        mass_encoder.add(Dense(self.mass_encoder_neurons, activation = 'relu'))

        return mass_encoder
        
    
    def buildMassSiamese(self, max_mass_seq_length):
        """
        Creates a siamese (sub)-network built using the mass encoder.
        Returns the two input tensors and the output tensor of the (sub)-network
        as well as the predictive output using only this (sub)-network
        """
        # Create input tensors
        mass_input_shape = (max_mass_seq_length,)
        mass_left_input = Input(mass_input_shape)
        mass_right_input = Input(mass_input_shape)

        mass_encoder = self.buildMassEncoder(mass_input_shape)

        # Make the siamese (sub)-network and get the output tensor
        mass_comparison = self.makeSiameseComponent(mass_encoder, mass_left_input, mass_right_input)
        
        # The predictive output is made from a single neuron with a sigmoid activation function
        mass_prediction = Dense(1, activation = 'sigmoid', name = 'mass_prediction')(mass_comparison)

        return mass_left_input, mass_right_input, mass_comparison, mass_prediction


    def buildPeakEncoder(self):
        """
        Creates a model for a peak encoder as a bi-directional LSTM.
        Three LSTM layers are used, with the first two being bi-directional.
        The output from the LSTM layers are passed through a fully connected layer to get the encoding
        """
        peak_input_shape = (None, 1)  # Variable sequence length
        P_in = Input(peak_input_shape)
        
        P = Bidirectional(LSTM(self.peak_network_neurons, return_sequences = True))(P_in)
        P = Dropout(self.peak_dropout_percentage)(P)
        P = Bidirectional(LSTM(self.peak_network_neurons, return_sequences = True))(P)
        P = Dropout(self.peak_dropout_percentage)(P)
        _, state_h, state_c = LSTM(self.peak_encoder_neurons, return_sequences = False, return_state = True)(P)
        
        peak_output = Concatenate(axis = -1)([state_h, state_c])  # Use both the LSTM state and memory to get the encoding
        peak_output = Dropout(self.peak_dropout_percentage)(peak_output)
        peak_output = Dense(self.peak_encoder_neurons)(peak_output)
        
        peak_encoder = Model(inputs = P_in, outputs = peak_output)

        return peak_encoder
    
    def buildPeakSiamese(self):
        """
        Creates a siamese (sub)-network using the peak encoder. See comments in buildMassSiamese
        """
        peak_input_shape = (None, 1)  # Variable sequence length
        peak_left_input = Input(peak_input_shape)
        peak_right_input = Input(peak_input_shape)

        peak_encoder = self.buildPeakEncoder()

        peak_comparison = self.makeSiameseComponent(peak_encoder, peak_left_input, peak_right_input)
        peak_prediction = Dense(1, activation = 'sigmoid', name = 'peak_prediction')(peak_comparison)

        return peak_left_input, peak_right_input, peak_comparison, peak_prediction

    def buildChromatogramEncoder(self, chromatogram_input_shape):
        """
        Creates a model for a chromatogram encoder as a CNN with two separate stacks of convolutions
        The output from the two stacks are combined and passed through a fully connected layer to get the encoding
        """
        C_in = Input(chromatogram_input_shape)
        
        initial_filter_number = self.initial_convolution_filter_number
        # Build up the left stack of convolutional layers
        for i in range(self.number_of_left_convolution_stacks):
            # The number of convolutional filters doubles per group of layers
            F1 = Conv1D(filters = initial_filter_number * (2**i), kernel_size = 3, strides = 1, padding = 'same', activation = 'relu')(C_in if i == 0 else F1)
            F1 = Conv1D(filters = initial_filter_number * (2**i), kernel_size = 3, strides = 1, padding = 'same', activation = 'relu')(F1)
            F1 = MaxPooling1D(3 if i == 0 else 2)(F1)  # Segment length of 600, 200, 100, 50, etc (reduces to a third after the first group of layers, then halves after each following group of layers)
            if (self.chromatogram_convolution_dropout_percentage != 0) and (i != (self.number_of_left_convolution_stacks - 1)):  # Don't apply dropout after the last group of layers
                F1 = Dropout(self.chromatogram_convolution_dropout_percentage)(F1)
        # Build up the right stack of convolutional layers 
        for i in range(self.number_of_right_convolution_stacks):
            F2 = Conv1D(filters = initial_filter_number * (2**i), kernel_size = 3, strides = 1, padding = 'same', activation = 'relu')(C_in if i == 0 else F2)
            F2 = MaxPooling1D(3 if i == 0 else 2)(F2)  # Segment length of 600, 200, 100, 50, etc
            if (self.chromatogram_convolution_dropout_percentage != 0) and (i != (self.number_of_right_convolution_stacks - 1)):
                F2 = Dropout(self.chromatogram_convolution_dropout_percentage)(F2)

        F1 = Flatten()(F1)
        F2 = Flatten()(F2)
        chromatogram_output = Concatenate(axis = -1)([F1, F2])  # Combine the output from the two stacks of convolutions to get the encoding
        chromatogram_output = Dropout(self.chromatogram_dropout_percentage)(chromatogram_output)
        chromatogram_output = Dense(self.chromatogram_encoder_neurons)(chromatogram_output)
        
        chromatogram_encoder = Model(inputs = C_in, outputs = chromatogram_output)
        
        return chromatogram_encoder

    def buildChromatogramSiamese(self, segment_length):
        """
        Creates a siamese (sub)-network using the chromatogram encoder. See comments in buildMassSiamese
        """
        chromatogram_input_shape = (segment_length, 1)  # One channel input
        chromatogram_left_input = Input(chromatogram_input_shape)
        chromatogram_right_input = Input(chromatogram_input_shape)

        chromatogram_encoder = self.buildChromatogramEncoder(chromatogram_input_shape)

        chromatogram_comparison = self.makeSiameseComponent(chromatogram_encoder, chromatogram_left_input, chromatogram_right_input)
        chromatogram_prediction = Dense(1, activation = 'sigmoid', name = 'chromatogram_prediction')(chromatogram_comparison)    
        
        return chromatogram_left_input, chromatogram_right_input, chromatogram_comparison, chromatogram_prediction


    def buildModel(self, max_mass_seq_length, segment_length):
        """
        Creates the ChromAlignNet model using the Keras functional API
        
        Arguments:
            max_mass_seq_length -- Number of data points in each mass spectra, as an Int
                                   Will be set as the input size of the mass encoder
            segment_length -- Number of time steps in the chromatogram segment, as an Int
                              Will be set as the input size of the chromatogram encoder
        
        Returns:
            model -- Keras model containing several siamese sub-networks
                     Needs to be compiled before training
        """
        # Create each siamese (sub)-network
        mass_left_input, mass_right_input, mass_comparison, mass_prediction = self.buildMassSiamese(max_mass_seq_length)
        if not self.ignore_peak_profile:
            peak_left_input, peak_right_input, peak_comparison, peak_prediction = self.buildPeakSiamese()
        chromatogram_left_input, chromatogram_right_input, chromatogram_comparison, chromatogram_prediction = self.buildChromatogramSiamese(segment_length)
        # Also include the difference in retention times as another input
        time_diff = Input((1,))

        if self.ignore_peak_profile:
            components = [mass_comparison, chromatogram_comparison, time_diff]
        else:
            components = [mass_comparison, peak_comparison, chromatogram_comparison, time_diff]

        # Concatonate all the outputs from each siamese (sub)-network together to do the main prediction
        combined_outputs = Lambda(lambda x: K.concatenate(x, axis = -1))(components)
        combined_outputs = Dropout(self.combined_output_dropout_percentage)(combined_outputs)
        combined_model = Dense(self.combined_output_neurons, activation = 'relu')(combined_outputs)
        combined_prediction = Dense(1, activation = 'sigmoid', name = 'main_prediction')(combined_model)

        # Select the relevant inputs and outputs, based on if the peak encoder was used or not
        if self.ignore_peak_profile:
            inputs = [mass_left_input, mass_right_input, chromatogram_left_input, chromatogram_right_input, time_diff]
            outputs = [combined_prediction, mass_prediction, chromatogram_prediction]
        else:
            inputs = [mass_left_input, mass_right_input, peak_left_input, peak_right_input, chromatogram_left_input, chromatogram_right_input, time_diff]
            outputs = [combined_prediction, mass_prediction, peak_prediction, chromatogram_prediction]

        # Create the overall network with all of the inputs and outputs
        model = Model(inputs = inputs, outputs = outputs)

        return model


class SimplifiedPeakEncoderVariant(ChromAlignModel):
    """
    Variant of ChromAlignModel with a simplified peak encoder
    Uni-directional GRU layers replaces the bi-directional LSTM layers
    """
    def buildPeakEncoder(self):
        peak_input_shape = (None, 1)  # Variable sequence length
        P_in = Input(peak_input_shape)
        
        P = GRU(self.peak_network_neurons, return_sequences = True)(P_in)
        P = Dropout(self.peak_dropout_percentage)(P)
        P = GRU(self.peak_network_neurons, return_sequences = True)(P)
        P = Dropout(self.peak_dropout_percentage)(P)
        _, state_h = GRU(self.peak_encoder_neurons, return_sequences = False, return_state = True)(P)

        peak_output = Dropout(self.peak_dropout_percentage)(state_h)
        peak_output = Dense(self.peak_encoder_neurons)(peak_output)

        peak_encoder = Model(inputs = P_in, outputs = peak_output)

        return peak_encoder


def getModelVariant(variant):
    """
    Initialises a ChromAlignModel (or similar) object with specific parameters matching a pre-specified variant
    
    Arguments:
        variant -- the model variant to select, given as an Int
    
    Returns:
        model -- ChromAlignModel (or similar) object constructed with specific parameters matching the requested variant
    """
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

        20: ChromAlignModel(mass_encoder_neurons = 5, ignore_peak_profile = True),  # 2 + 8
        21: ChromAlignModel(mass_dropout_percentage = 0.5, mass_encoder_neurons = 20, ignore_peak_profile = True),  # 2 + 9
        22: ChromAlignModel(chromatogram_encoder_neurons = 5, ignore_peak_profile = True),  # 2 + 11
        23: ChromAlignModel(chromatogram_dropout_percentage = 0.5, chromatogram_encoder_neurons = 20, ignore_peak_profile = True),  # 2 + 12
        24: ChromAlignModel(number_of_left_convolution_stacks = 3, ignore_peak_profile = True),  # 2 + 17
        25: ChromAlignModel(number_of_right_convolution_stacks = 2, ignore_peak_profile = True),  # 2 + 19
        
        26: SimplifiedPeakEncoderVariant(mass_encoder_neurons = 5),    # 3 + 8
        27: SimplifiedPeakEncoderVariant(mass_dropout_percentage = 0.5, mass_encoder_neurons = 20),    # 3 + 9
        28: SimplifiedPeakEncoderVariant(chromatogram_encoder_neurons = 5),    # 3 + 11
        29: SimplifiedPeakEncoderVariant(chromatogram_dropout_percentage = 0.5, chromatogram_encoder_neurons = 20),    # 3 + 12
        30: SimplifiedPeakEncoderVariant(number_of_left_convolution_stacks = 3),    # 3 + 17
        31: SimplifiedPeakEncoderVariant(number_of_right_convolution_stacks = 2)    # 3 + 19
    }
    assert variant in switcher, "Model variant does not exist. Check the integer input"
    model = switcher[variant]
    return model