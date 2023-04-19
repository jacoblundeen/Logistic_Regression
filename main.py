import numpy as np
import random
from typing import List, Tuple, Dict, Callable

random.seed(49)


def blur(data):
    def apply_noise(value):
        if value < 0.5:
            v = random.gauss(0.10, 0.05)
            if v < 0.0:
                return 0.0
            if v > 0.75:
                return 0.75
            return v
        else:
            v = random.gauss(0.90, 0.10)
            if v < 0.25:
                return 0.25
            if v > 1.00:
                return 1.00
            return v

    noisy_readings = [apply_noise(v) for v in data[0:-1]]
    return noisy_readings + [data[-1]]


def generate_data(data, n, key_label):
    labels = list(data.keys())
    labels.remove(key_label)

    total_labels = len(labels)
    result = []
    # create n "not label" and code as y=0
    count = 1
    while count <= n:
        label = labels[count % total_labels]
        datum = blur(random.choice(data[label]))
        xs = datum[0:-1]
        result.append((xs, 0))
        count += 1

    # create n "label" and code as y=1
    for _ in range(n):
        datum = blur(random.choice(data[key_label]))
        xs = datum[0:-1]
        result.append((xs, 1))
    random.shuffle(result)
    return result


def calculate_error(thetas: List, x_data: List, y_data: List) -> float:
    n = len(x_data)
    y_hat = calculate_yhat(thetas, x_data)
    error = (-1 / n) * np.sum(y_data * np.log(y_hat) + np.subtract(1, y_data) * np.log(np.subtract(1, y_hat)))
    return error


def calculate_yhat(thetas: List, x_data: List) -> List:
    y_hat = []
    for obs in x_data:
        y_hat.append(1 / (1 + np.exp(np.sum(-1 * thetas * obs))))
    return y_hat


def derivative(j: int, thetas: List, x_data: List, y_data: List) -> int:
    y_hat = calculate_yhat(thetas, x_data)
    deriv = (1/len(x_data)) * np.sum(np.subtract(y_hat, y_data) * x_data[:, j])
    return deriv


def learn_model(data: List[List], verbose=False) -> List:
    x_data = np.delete(data, -1, axis=1)
    y_data = data[:, [-1]]
    thetas = np.random.uniform(-1, 1, len(x_data[0]))
    epsilon = 1E-07
    alpha = 0.1
    previous_error = 0.0
    current_error = calculate_error(thetas, x_data, y_data)
    count = 0
    while abs(current_error - previous_error) > epsilon:
        new_thetas = []
        for j in range(len(thetas)):
            new_thetas.append(thetas[j] - alpha * derivative(j, thetas, x_data, y_data))
        thetas = np.array(new_thetas)
        previous_error = current_error
        current_error = calculate_error(thetas, x_data, y_data)
        if verbose and count % 1000 == 0:
            print("The current error is: " + str(current_error))
        if current_error > previous_error:
            alpha = alpha / 10
        count += 1
    return thetas


def apply_model(model: List, test_data: List, labeled=False) -> List[Tuple]:
    predictions = []
    x_test = np.delete(test_data, -1, axis=1)
    y_test = test_data[:, [-1]]
    count = 0
    for obs in x_test:
        pred = np.sum(model * obs)
        if labeled:
            if pred >= 0.5:
                predictions.append((y_test[count][0], 1))
            else:
                predictions.append((y_test[count][0], 0))
        else:
            if pred >= 0.5:
                predictions.append((1, pred))
            else:
                predictions.append((0, pred))
        count += 1
    return predictions


def transform_data(data: List[Tuple[List]]) -> List:
    temp = np.empty(len(data[0][0]) + 1)
    x_0 = np.ones([len(data), 1])
    count = 0
    for result in data:
        temp1 = result[0]
        temp1.append(result[1])
        if count == 0:
            temp = temp1
        else:
            temp = np.vstack((temp, temp1))
        count += 1
    temp = np.append(x_0, temp, axis=1)
    return temp


def evaluate(results: List[Tuple]) -> float:
    true, pred = map(list, zip(*results))
    confusion_matrix = np.zeros((2,2))
    error_rate = (sum(i != j for i, j in zip(true, pred)) / len(true)) * 100
    for i in range(len(pred)):
        if int(pred[i]) == 1 and int(true[i]) == 0:
            confusion_matrix[0, 0] += 1  # True Positives
        elif int(pred[i]) == -1 and int(true[i]) == 1:
            confusion_matrix[0, 1] += 1  # False Positives
        elif int(pred[i]) == 0 and int(true[i]) == 1:
            confusion_matrix[1, 0] += 1  # False Negatives
        elif int(pred[i]) == 0 and int(true[i]) == 0:
            confusion_matrix[1, 1] += 1  # True Negatives
    return error_rate, confusion_matrix


if __name__ == "__main__":
    clean_data = {
        "plains": [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, "plains"]
        ],
        "forest": [
            [0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, "forest"],
            [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, "forest"],
            [1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, "forest"],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, "forest"]
        ],
        "hills": [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, "hills"],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, "hills"],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, "hills"],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, "hills"]
        ],
        "swamp": [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, "swamp"],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, "swamp"]
        ]
    }

    train_data = transform_data(generate_data(clean_data, 10, "hills"))
    test_data = transform_data(generate_data(clean_data, 10, "hills"))

    model = learn_model(train_data, True)
    results = apply_model(model, test_data, True)
    error_rate, confusion_matrix = evaluate(results)
    print(model)
    print(results)
    print(str(error_rate) + "%")
    print(confusion_matrix)
