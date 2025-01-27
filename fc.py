from typing import Callable, Union

import numpy as np
from tensorflow.keras.initializers import GlorotUniform

from .es_array import EsArray


class BaseFc:
    ACT_RELU = "relu"
    ACT_SIGMOID = "sigmoid"
    ACT_SOFTMAX = "softmax"

    def __init__(
        self,
        wgts: Union[np.ndarray, EsArray],
        bias: Union[np.ndarray, EsArray],
        activation: str,
    ):
        assert len(wgts.shape) == 2
        assert len(bias.shape) == 1
        assert wgts.shape[1] == bias.shape[0]
        assert activation in (self.ACT_RELU, self.ACT_SIGMOID, self.ACT_SOFTMAX)

        self.wgts: Union[np.ndarray, EsArray] = wgts  # in_dim X out_dim
        self.bias: Union[np.ndarray, EsArray] = bias  # out_dim
        self.activation: str = activation

    @classmethod
    def build_from_dim(cls, in_dim: int, out_dim: int, activation: str, **kwargs):
        raise NotImplementedError

    @property
    def in_dim(self) -> int:
        return self.wgts.shape[0]

    @property
    def out_dim(self) -> int:
        return self.wgts.shape[1]

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def mutate(self):
        raise NotImplementedError

    def recombine(self, other) -> tuple:
        raise NotImplementedError

    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        assert len(x.shape) == 1
        return x * (x > 0)

    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        assert len(x.shape) == 1
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        assert len(x.shape) == 1
        return np.exp(x) / sum(np.exp(x))


class GaFc(BaseFc):
    BIAS_MU = 0
    BIAS_SIGMA = 0.1

    def __init__(
        self,
        wgts: np.ndarray,
        bias: np.ndarray,
        activation: str,
        **config,  # callbacks of GA operators
    ):
        super().__init__(wgts, bias, activation)
        self.wgts: np.ndarray  # in_dim X out_dim
        self.bias: np.ndarray  # out_dim

        self.wgts_crossover_callback: Callable = config["wgts_crossover_callback"]
        self.wgts_mutate_callback: Callable = config["wgts_mutate_callback"]
        self.bias_crossover_callback: Callable = config["bias_crossover_callback"]
        self.bias_mutate_callback: Callable = config["bias_mutate_callback"]

    @classmethod
    def build_from_dim(cls, in_dim: int, out_dim: int, activation: str, **config):
        initializer = GlorotUniform()
        return GaFc(
            wgts=np.array(initializer(shape=[in_dim, out_dim])),
            bias=np.random.normal(cls.BIAS_MU, cls.BIAS_SIGMA, size=out_dim),
            activation=activation,
            **config,
        )

    @property
    def call_back_config(self) -> dict:
        return {
            "wgts_crossover_callback": self.wgts_crossover_callback,
            "wgts_mutate_callback": self.wgts_mutate_callback,
            "bias_crossover_callback": self.bias_crossover_callback,
            "bias_mutate_callback": self.bias_mutate_callback,
        }

    def forward(self, x: np.ndarray) -> np.ndarray:
        res = np.matmul(x, self.wgts)
        res += self.bias

        if self.activation == self.ACT_RELU:
            res = self.relu(res)
        elif self.activation == self.ACT_SIGMOID:
            res = self.sigmoid(res)
        else:
            res = self.softmax(res)

        return res

    def mutate(self):
        """update in place"""
        self.wgts_mutate_callback(self.wgts)
        self.bias_mutate_callback(self.bias)

    def recombine(self, other) -> tuple:
        assert isinstance(other, GaFc)

        wgts1, wgts2 = self.wgts_crossover_callback(self.wgts, other.wgts)
        bias1, bias2 = self.bias_crossover_callback(self.bias, other.bias)

        child1 = GaFc(wgts1, bias1, self.activation, **self.call_back_config)
        child2 = GaFc(wgts2, bias2, self.activation, **self.call_back_config)

        return child1, child2


class EsFc(BaseFc):
    def __init__(self, wgts: EsArray, bias: EsArray, activation: str):
        super().__init__(wgts, bias, activation)
        self.wgts: EsArray
        self.bias: EsArray

    @classmethod
    def build_from_dim(cls, in_dim: int, out_dim: int, activation: str, **arr_kwargs):
        """
        kawrgs:
            - sigma
            - k
            - k_prime
            - eps
        """

        return EsFc(
            wgts=EsArray(shape=(in_dim, out_dim), **arr_kwargs),
            bias=EsArray(shape=(out_dim,), **arr_kwargs),
            activation=activation,
        )

    def forward(self, x: np.ndarray) -> np.ndarray:
        res = np.matmul(x, self.wgts.array)
        res += self.bias.array

        if self.activation == self.ACT_RELU:
            res = self.relu(res)
        elif self.activation == self.ACT_SIGMOID:
            res = self.sigmoid(res)
        else:
            res = self.softmax(res)

        return res

    def mutate(self):
        return EsFc(
            wgts=self.wgts.mutate(),
            bias=self.bias.mutate(),
            activation=self.activation,
        )
