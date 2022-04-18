import pandas as pd
import numpy as np
import sys
from logreg_train import sigmoid


# Get weights from a saved file
def get_weights():
    weights = []

    try:
        file = open('../../saves/weights.txt', 'r')

        for line in file.readlines():
            values = line.split(', ')
            values[:3] = [float(i) for i in values[:3]]
            values[3] = values[3][:len(values[3]) - 1]
            weights.append((values[:3], values[3]))
    except IOError:
        print('Error during weights file manipulation')
    return weights


# Select only two columns useful to predict the model
def clean_dataset(path):
    # Check if path is valid
    i

    data = pd.read_csv(path)

    data = data[data['Herbology'].notna()]
    data = data[data['Defense Against the Dark Arts'].notna()]

    data = data.iloc[:, 8:10]
    print(data.head())

    return data.values


# Predict each student by his marks
def predict(weights, X):
    X = np.insert(X, 0, 1, axis=1)
    sigmoid_v = np.vectorize(sigmoid)
    X_predicted = [max((sigmoid_v(i.dot(weight)), c) for weight, c in weights)[1] for i in X]
    return X_predicted


# Write predictions in saves/houses.csv file
def write_predictions(predictions):
    try:
        file = open('../../saves/houses.csv', 'w+')

        # Write header
        file.write('Index, Hogwarts House\n')

        # Write prediction line by line
        index = 0
        for prediction in predictions:
            file.write(str(index) + ',' + str(prediction) + '\n')
            index += 1
    except IOError:
        print('Error during weights file manipulation')
    file.close()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage : python3 logreg_predict.py "PATH_DATASET"')
        sys.exit(1)

    # Clean dataset
    data = clean_dataset(sys.argv[1])

    # Get predictions
    predictions = predict(get_weights(), data)

    # Write predicitions in correct file
    write_predictions(predictions)
