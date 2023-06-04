from keras import Sequential
from keras.layers import SimpleRNN, Dense, Dropout, LSTM, BatchNormalization, Input, Reshape


def build_model() -> Sequential:
    model = Sequential()

    model.add(Input(shape=(4, )))
    model.add(Reshape((4, 1)))
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

    # model.add(Dense(2, activation=keras.activations.relu(alpha=10e-4)))
    model.add(Dense(2, activation='softmax'))

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    return model
