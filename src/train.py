import numpy as np
import joblib
import sys
from model import build_model
from sklearn.metrics import f1_score
from config import Config


def train(X_train, X_test, y_train, y_test):
    model = build_model(X_train.shape[1])

    # Train the model
    model.fit(X_train, y_train, epochs=50, batch_size=32)

    # Predict the labels for the test set
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)

    f1 = f1_score(y_test, y_pred)
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f'F1 score: {f1:.3f}')
    print(f'Accuracy: {test_acc:.3f}')
    print(f'Loss:     {test_loss:.3f}')

    return model


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <csv_file>")
        sys.exit(1)

    csv_file = sys.argv[1].split('.')[0]

    config = Config(csv_file)

    x_train, x_test, y_train, y_test = joblib.load(
        config.features_path)

    # Train and save the model
    train(x_train, x_test, y_train, y_test)\
        .save(config.model_path)
