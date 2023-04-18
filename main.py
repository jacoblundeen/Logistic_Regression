import numpy as np
import random

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


def calculate_error(thetas, x_data, y_data):
    n = len(x_data)
    y_hat = calculate_yhat(thetas, x_data)
    error = (-1 / n) * np.sum(y_data * np.log(y_hat) + np.subtract(1, y_data) * np.log(np.subtract(1, y_hat)))
    return error


def calculate_yhat(thetas, x_data):
    y_hat = []
    for index in range(len(x_data)):
        y_hat.append(1 / (1 + sum(np.exp(-1 * thetas * x_data[index]))))
    return y_hat


def derivative(j, thetas, x_data, y_data):
    y_hat = calculate_yhat(thetas, x_data)
    deriv = (1/len(x_data)) * np.sum(np.subtract(y_hat, y_data) * x_data[:,j])
    return deriv


def learn_model(data, verbose=False):
    x_data = np.delete(data, -1, axis=1)
    y_data = data[:, [-1]]
    thetas = np.random.uniform(-1, 1, len(x_data[0]))
    epsilon = 1E-07
    alpha = 0.1
    previous_error = 0.0
    current_error = calculate_error(thetas, x_data, y_data)
    while abs(current_error - previous_error) > epsilon:
        new_thetas = []
        for j in range(len(thetas)):
            new_thetas.append(thetas[j] - alpha * derivative(j, thetas, x_data, y_data))
        thetas = new_thetas
        previous_error = current_error
        current_error = calculate_error(thetas, x_data, y_data)
    return thetas


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

    results = generate_data(clean_data, 10, "hills")
    print(results[0][0])
    temp = np.empty(len(results[0][0]) + 1)
    count = 0
    for result in results:
        temp1 = result[0]
        temp1.append(result[1])
        if count == 0:
            temp = temp1
        else:
            temp = np.vstack((temp, temp1))
        count += 1

    model = learn_model(temp)
    print(model)
