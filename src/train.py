import numpy as np
import joblib
import sys
from model import build_model
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from config import Config
from datetime import datetime


def save_history(hist_file, f1, accuracy, precision, recall, test_loss, test_acc):
    with open(hist_file, 'a') as f:
        f.write(
            "\n%s"
            "\nF1 Score:        %s"
            "\nAccuracy:        %s"
            "\nPrecision:       %s"
            "\nRecall:          %s"
            "\nTest Loss:       %s"
            "\nTest Accuracy:   %s"
            "\n---------------------" %
            (datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
             f1, accuracy, precision, recall, test_loss, test_acc)
        )


def train(save_path, X_train, X_test, y_train, y_test):
    model = build_model(X_train.shape[1])

    # Train the model
    model.fit(X_train, y_train, epochs=50, batch_size=32)

    # Predict the labels for the test set
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)

    # Evaluation metrics
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print(f'F1 score:       {f1:.3f}')
    print(f'Accuracy:       {accuracy:.3f}')
    print(f'Precision:      {precision:.3f}')
    print(f'Recall:         {recall:.3f}')

    # Test set loss and accuracy
    test_loss, test_acc = model.evaluate(X_test, y_test)

    print(f'Test Accuracy:  {test_acc:.3f}')
    print(f'Test Loss:      {test_loss:.3f}')

    model.save(save_path)

    return (f1, accuracy, precision, recall, test_loss, test_acc)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <csv_file>")
        sys.exit(1)

    csv_file = sys.argv[1].split('.')[0]

    config = Config(csv_file)

    x_train, x_test, y_train, y_test = joblib.load(
        config.features_path)

    # Train and save the model
    eval = train(config.model_path, x_train, x_test, y_train, y_test)
    
    save_history(config.history_path, *eval)
