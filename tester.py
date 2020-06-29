import os

import matplotlib.pyplot as plt
import numpy as np


class Tester:

    def __init__(self, dir_path):
        self.dir = dir_path
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

    def test_svm(self, model, features_batch, test_y, batch_size, svm, transformers):

        for transformer in transformers:
            features_batch = transformer.transform(features_batch)
        y_pred = svm.predict(features_batch)
        print('actual:     ', list(np.int_(test_y[test_y.shape[0] - 1:, 0, ].reshape((24)))))
        print('prediction: ', list(np.int_(y_pred[y_pred.shape[0] - 24:, ])))
        plt.ion()
        plt.plot(list(np.int_(test_y[test_y.shape[0] - 1:, 0, ].reshape((24)))), color='red')
        plt.plot(list(np.int_(y_pred[y_pred.shape[0] - 24:, ])), color='blue')
        plt.savefig(self.dir + '/svm_9_tir.png')
        plt.close()

    def test_neuron(self, model, test_x, test_y, batch_size):

        y_pred = model.predict(test_x, batch_size=batch_size)

        print(y_pred[y_pred.shape[0] - 1, 0, ])
        plt.ion()
        plt.plot(list(np.int_([
            417.51,
            394.82,
            359.13,
            329.58,
            303.47,
            275.44,
            244.91,
            238.28,
            252.34,
            278.81,
            309.94,
            326.57,
            344.83,
            364.66,
            358.95,
            313.13,
            274.05,
            245.75,
            237.56,
            236.47,
            246.38,
            263.63,
            266.43,
            262.14
        ])), color='red')
        plt.plot(list(np.int_((y_pred[y_pred.shape[0] - 1, 0, ]))), color='blue')
        plt.savefig(self.dir + '/neural_network_9_tir.png')
        plt.close()
