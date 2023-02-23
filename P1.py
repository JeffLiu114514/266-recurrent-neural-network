import numpy as np


class RNN():
    def __init__(self, W, U, bh, V, by, h0):
        self.W = W
        self.U = U
        self.bh = bh
        self.V = V
        self.by = by
        self.ht_1 = h0

    def sigmoid(self, input_vector):
        out = np.zeros(np.size(input_vector))
        if (np.size(input_vector) > 1):
            for i, value in enumerate(input_vector):
                if value > 0:
                    out[i] = 1
                else:
                    out[i] = 0
        else:
            if input_vector > 0:
                out = 1
            else:
                out = 0
        return out

    def forward(self, x):
        h = self.sigmoid(np.dot(self.U, x) +
                         np.dot(self.W, self.ht_1) + self.bh)
        y = self.sigmoid(self.V.dot(h) + self.by)
        print(
            f"Ux: {np.dot(self.U, x)}\n Wht-1: {np.dot(self.W, self.ht_1)}\n h: {h}\n Vh: {self.V.dot(h)}\n y: {y}")
        self.ht_1 = h
        return y


def main():
    W = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]])
    U = np.array([[1, 1], [1, 1], [1, 1]])
    bh = np.array([0, -1, -2])
    by = 0
    V = np.array([1, -1, 1])
    h0 = np.array([0, 0, 0])

    rnn = RNN(W, U, bh, V, by, h0)
    input1 = np.array([1, 0, 0, 1, 1, 1])
    input2 = np.array([1, 1, 0, 0, 1, 0])
    output = np.zeros(len(input1)+1)

    for i in reversed(range(np.size(input1))):
        output[i+1] = rnn.forward(np.array([input1[i], input2[i]]))
    if np.array_equal(np.dot(W, rnn.ht_1), np.array([1, 1, 1])):
        output[0] = 1

    print(output)


if __name__ == '__main__':
    main()
