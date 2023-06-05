# The code in this module is adapted from the ElliotVilhelm/RNN-Trading
# with modifications made to suit the specific needs of this project.
# https://github.com/ElliotVilhelm/RNN-Trading

from keras import Sequential
from keras.layers import SimpleRNN, Dense, Dropout, LSTM, BatchNormalization, Input, Reshape


def build_model(dim) -> Sequential:
    model = Sequential()

    model.add(Input(shape=(dim, )))
    model.add(Reshape((dim, 1)))
    model.add(BatchNormalization())

    model.add(LSTM(256, return_sequences=True))
    model.add(Dropout(0.8))
    model.add(BatchNormalization())

    model.add(SimpleRNN(128, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())

    model.add(LSTM(256))
    model.add(Dropout(0.8))
    model.add(BatchNormalization())

    model.add(Dense(64, activation='tanh'))
    model.add(Dropout(0.1))

    model.add(Dense(2, activation='softmax'))

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    return model
