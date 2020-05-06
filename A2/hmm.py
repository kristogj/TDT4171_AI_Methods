import numpy as np

T = np.array([[0.7, 0.3], [0.3, 0.7]])  # Transition model
O = np.array([[0.9, 0], [0, 0.2]])  # Sensor model


def forward_backward(p_rain0, umbrella_t, b):
    fv = [p_rain0]  # Forwarded values - prior added for Rain_0
    sv = []  # Smoothed values

    print("Forwarded values:")
    for i in range(1, len(umbrella_t)):
        fv.append(forward(fv[i - 1], umbrella_t[i]))
        print("Probability Rain day {}: {}".format(i, fv[i].flatten()))

    print("\nSmoothed values:")
    for i in range(len(umbrella_t) - 1, 0, -1):
        smoothed = np.multiply(fv[i], b)
        smoothed /= sum(smoothed)  # Normalize
        sv.append(smoothed)
        b = backward(b, umbrella_t[i])
        mapped_index = len(umbrella_t) - 1 - i
        print("Probability Rain day {}: {}".format(i, sv[mapped_index].flatten()))


def forward(f_prev, u_i):
    O_i = O
    if not u_i:
        O_i = 1 - O_i
    f = O_i.dot(T.T).dot(f_prev)  # Forward message
    f /= sum(f)  # Normalize so it sum to 1
    return f


def backward(b, u_i):
    O_i = O
    if not u_i:
        O_i = 1 - O_i
    b = T.dot(O_i).dot(b)
    return b


if __name__ == '__main__':
    rain_0 = np.array([[0.5], [0.5]])
    umbrella_t = [None, True, True, False, True, True]
    b = np.ones((2, 1))
    forward_backward(rain_0, umbrella_t, b)
