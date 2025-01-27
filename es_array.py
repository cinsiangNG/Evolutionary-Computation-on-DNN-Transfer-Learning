from copy import deepcopy

import numpy as np
from tensorflow.keras.initializers import GlorotUniform


class EsArray:
    def __init__(self, shape: tuple, step_size: float, k_prime: float=1, k: float=1, eps: float=1e-3):
        self.array: np.ndarray

        if len(shape) == 1:
            n = shape[0]
            self.array = np.random.normal(0, 0.1, size=shape)
        elif len(shape) == 2:
            n = shape[0] * shape[1]
            initializer = GlorotUniform()
            self.array = np.array(initializer(shape=[*shape]))
        else:
            raise AssertionError

        self.sigmas: np.ndarray = np.ones(shape) * step_size
        self.tau_prime: float = k_prime / (2 * n)**0.5
        self.tau: float = k / (2 * n**0.5)**0.5
        self.eps: float = eps

    @property
    def shape(self) -> tuple:
        return self.array.shape

    def mutate(self):
        child = deepcopy(self)

        overall_lr = self.tau_prime * np.random.normal(0, 1)
        coord_wise_lr = self.tau * np.random.normal(0, 1, self.shape)

        child.sigmas *= np.exp(coord_wise_lr + overall_lr)
        child.sigmas = np.clip(child.sigmas, self.eps, None)

        child.array += child.sigmas * np.random.normal(0, 1, self.shape)

        return child
