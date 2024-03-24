import numpy as np
from scipy import signal
from keras.models import Model
from keras.layers import Dense, Conv1D, Flatten, Dropout, BatchNormalization, Input, Activation, MaxPooling1D

''' Adapted from:
        Sajjad Abdoli, Patrick Cardinal, Alessandro Lameiras Koerich,
        End-to-end environmental sound classification using a 1D convolutional neural network,
        Expert Systems with Applications,
        Volume 136,
        2019,
        Pages 252-263,
        ISSN 0957-4174,
        https://doi.org/10.1016/j.eswa.2019.06.040.
    Model architecture: Modified 16,000 and 16,000G
'''


def create_classification_model_gammatone(input_size, sr=22050):
    '''
    Create a 1D CNN model with gammatone filters as the first layer.
    input_size: int, the size of the input tensor.
    return: keras model.
    '''
    
    input_tensor = Input(shape=(input_size, 1))

    # CL1
    x = Conv1D(filters=64, kernel_size=512, strides=1, padding='valid')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # PL1
    x = MaxPooling1D(pool_size=8, strides=8)(x)
    x = Activation('relu')(x)
    # CL2
    x = Conv1D(filters=32, kernel_size=32, strides=2, padding='valid')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # PL2
    x = MaxPooling1D(pool_size=8, strides=8)(x)
    x = Activation('relu')(x)
    # CL3
    x = Conv1D(filters=64, kernel_size=16, strides=8, padding='valid')(x)

    x = Flatten()(x)
    x = Dense(32)(x)
    x = Activation('relu')(x)
    x = Dropout(0.25)(x)
    x = Dense(16)(x)
    x = Activation('relu')(x)
    x = Dropout(0.25)(x)
    x = Dense(1)(x)
    x = Activation('sigmoid')(x)

    model = Model(inputs=input_tensor, outputs=x)

    n_filters = 64
    gammatone_filters = np.zeros((n_filters, 512))
    gammatone_filters_center_freqs = np.geomspace(0.1e3, 8e3, n_filters)
    for i in range(n_filters):
        gammatone_filters[i], _ = signal.gammatone(gammatone_filters_center_freqs[i], 'fir', 4, 512, sr)
    
    model.layers[1].set_weights([gammatone_filters[:, np.newaxis, :].T, np.zeros(n_filters)])

    return model


def create_classification_model(input_size):
    '''
    Create a 1D CNN model with random initialization.
    input_size: int, the size of the input tensor.
    return: keras model.
    '''
    
    input_tensor = Input(shape=(input_size, 1))

    # CL1
    x = Conv1D(filters=16, kernel_size=64, strides=2, padding='valid')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # PL1
    x = MaxPooling1D(pool_size=8, strides=8)(x)
    x = Activation('relu')(x)
    # CL2
    x = Conv1D(filters=32, kernel_size=32, strides=2, padding='valid')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # PL2
    x = MaxPooling1D(pool_size=8, strides=8)(x)
    x = Activation('relu')(x)
    # CL3
    x = Conv1D(filters=64, kernel_size=16, strides=8, padding='valid')(x)

    x = Flatten()(x)
    x = Dense(32)(x)
    x = Activation('relu')(x)
    x = Dropout(0.25)(x)
    x = Dense(16)(x)
    x = Activation('relu')(x)
    x = Dropout(0.25)(x)
    x = Dense(1)(x)
    x = Activation('sigmoid')(x)

    return Model(inputs=input_tensor, outputs=x)

if __name__ == "__main__":
    pass
