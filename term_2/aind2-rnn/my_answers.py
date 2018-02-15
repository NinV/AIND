import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []

    start = 0
    stop = window_size
    while stop < len(series):
        X.append(series[start:stop])
        y.append(series[stop])
        start +=1
        stop +=1

    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)
    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    """
    - layer 1 uses an LSTM module with 5 hidden units (note here the input_shape = (window_size,1))
    - layer 2 uses a fully connected module with one unit
    - the 'mean_squared_error' loss should be used (remember: we are performing regression here)
    """
    model = Sequential()
    model.add(LSTM(5, input_shape=(window_size,1)))
    model.add(Dense(1))
    model.summary()
    return model


### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    import string
    punctuation = ['!', ',', '.', ':', ';', '?']
    ascii_characters = [c for c in string.ascii_lowercase]
    legal_characters = set(punctuation + ascii_characters + [' '])
    text = [c for c in text if c in legal_characters]
    return ''.join(text)

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []

    start = 0
    stop = window_size
    while stop < len(text):
        inputs.append(text[start:stop])
        outputs.append(text[stop])
        start += step_size
        stop += step_size

    return inputs, outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    """
    Time to get to work: build a 3 layer RNN model of the following specification
        - layer 1 should be an LSTM module with 200 hidden units --> note this should have input_shape = (window_size,len(chars)) where len(chars) = number of unique characters in your cleaned text
        - layer 2 should be a linear module, fully connected, with len(chars) hidden units --> where len(chars) = number of unique characters in your cleaned text
        - layer 3 should be a softmax activation ( since we are solving a *multiclass classification*)
        - Use the **categorical_crossentropy** loss
    """
    model = Sequential()
    model.add(LSTM(200, input_shape=(window_size, num_chars)))
    model.add(Dense(num_chars, activation='softmax'))
    model.summary()
    return model