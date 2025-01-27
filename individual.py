from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from .fc import BaseFc, GaFc, EsFc


@dataclass
class Performance:
    train_loss: float
    test_loss: float
    train_acc: float
    test_acc: float


class BaseIndividual:
    EVAL_BY_ACC = "acc"
    EVAL_BY_LOSS = "loss"

    def __init__(self, eval_by: str, dim_list: list[int] = [], **fc_kwargs):
        """
        dim_list: [ in_dim, hidden_dim1, hidden_dim2, ..., out_dim ]
        """
        assert eval_by in (self.EVAL_BY_ACC, self.EVAL_BY_LOSS)

        self.eval_by: str = eval_by
        self.layers: list[BaseFc] = []
        self.performance: Performance = None
        self.age: int = 0

        for i in range(len(dim_list) - 1):
            in_dim = dim_list[i]
            out_dim = dim_list[i + 1]
            assert in_dim > 0

            if i < len(dim_list) - 2:
                # hidden layer
                activation = BaseFc.ACT_RELU
            elif out_dim == 1:
                # output layer
                activation = BaseFc.ACT_SIGMOID
            else:
                # output layer
                activation = BaseFc.ACT_SOFTMAX

            fc = self._build_fc(
                in_dim=in_dim,
                out_dim=out_dim,
                activation=activation,
                **fc_kwargs,
            )
            self.layers.append(fc)

    @property
    def fitness(self) -> float:
        return (
            self.performance.train_acc
            if self.eval_by == self.EVAL_BY_ACC
            else 1 / self.performance.train_loss
        )

    @property
    def in_dim(self) -> int:
        return self.layers[0].in_dim

    @property
    def out_dim(self) -> int:
        return self.layers[-1].out_dim

    def copy(self):
        ind = deepcopy(self)
        ind.age = 0
        return ind

    def predict(self, x: np.ndarray) -> float:
        res = np.copy(x)
        for layer in self.layers:
            res = layer.forward(res)

        return res[0]

    def increment_age(self):
        self.age += 1

    def set_performance(self, p: Performance):
        assert self.performance is None  # performance should be calculated only once
        self.performance = p

    def add_layer(self, layer: BaseFc):
        assert (len(self.layers) == 0) or (self.out_dim == layer.in_dim)
        self.layers.append(layer)

    def mutate(self):
        raise NotImplementedError

    def recombine(self, other) -> tuple:
        raise NotImplementedError

    def _build_fc(self, in_dim: int, out_dim: int, activation: str, **kwargs) -> BaseFc:
        raise NotImplementedError


class GaIndividual(BaseIndividual):
    def __init__(self, eval_by: str, dim_list: list[int] = [], **ga_operator_config):
        super().__init__(eval_by, dim_list, **ga_operator_config)
        self.layers: list[GaFc]

    def mutate(self):
        """update in place"""
        for fc in self.layers:
            fc.mutate()

        # need reevaluation
        self.performance = None

    def recombine(self, other) -> tuple:
        assert isinstance(other, GaIndividual)
        assert len(self.layers) == len(other.layers)

        child_a, child_b = GaIndividual(self.eval_by), GaIndividual(self.eval_by)

        for self_layer, other_layer in zip(self.layers, other.layers):
            fc_a, fc_b = self_layer.recombine(other_layer)
            child_a.add_layer(fc_a)
            child_b.add_layer(fc_b)

        return child_a, child_b

    def _build_fc(self, in_dim: int, out_dim: int, activation: str, **operator_callbacks) -> GaFc:
        return GaFc.build_from_dim(
            in_dim=in_dim,
            out_dim=out_dim,
            activation=activation,
            **operator_callbacks,
        )


class EsIndividual(BaseIndividual):
    def __init__(self, eval_by: str, dim_list: list[int] = [], **mutate_kwargs):
        super().__init__(eval_by, dim_list, **mutate_kwargs)
        self.layers: list[EsFc]

    def mutate(self):
        child = EsIndividual(self.eval_by)
        for layer in self.layers:
            child.add_layer(layer.mutate())

        return child

    def _build_fc(self, in_dim: int, out_dim: int, activation: str, **mutate_kwargs) -> EsFc:
        return EsFc.build_from_dim(
            in_dim=in_dim,
            out_dim=out_dim,
            activation=activation,
            **mutate_kwargs,
        )
