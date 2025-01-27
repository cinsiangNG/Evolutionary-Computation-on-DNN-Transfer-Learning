from typing import Callable
import random
import gc
from copy import deepcopy
from dataclasses import dataclass
import time
import os

import numpy as np
from sklearn.metrics import accuracy_score, log_loss

from .individual import BaseIndividual, GaIndividual, EsIndividual, Performance


@dataclass
class Dataset:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


class EC:
    def __init__(self, pop_sz: int, offspr_sz: int, dataset: Dataset):
        """
        offspr_sz: number of offsprings to be created per _evolve
        """

        self.pop_sz: int = pop_sz
        self.offspr_sz: int = offspr_sz
        self.dataset: Dataset = dataset

        self.population: list[BaseIndividual]
        self.best_individual: BaseIndividual = None
        self._update_best_individual()

    @property
    def best_fitness(self) -> float:
        return self._evaluate(self.best_individual)

    def run(self, max_iter: int, target_acc: float, save_history_to: str, report_period: int = 10) -> BaseIndividual:
        assert not os.path.exists(save_history_to)

        start_time = time.time()

        for i in range(max_iter):
            self._evolve()
            assert len(self.population) == self.pop_sz
            self._update_best_individual()

            with open(save_history_to, "a") as f:
                time_elapsed = time.time() - start_time
                perf = self.best_individual.performance
                f.write(f"{time_elapsed}, {perf}\n")

            if (i + 1) % report_period == 0:
                print(f"===== Iteration {i+1} ===== Best Individual: {self.best_individual.performance}")

            if self.best_individual.performance.train_acc >= target_acc:
                break

        return self.best_individual

    def _evolve(self):
        raise NotImplementedError

    def _update_best_individual(self):
        ind = max(self.population, key=lambda x: self._evaluate(x))

        if (self.best_individual is None) or (ind.fitness > self.best_fitness):
            self.best_individual = deepcopy(ind)  # escape from _eliminate_oldest

    def _eliminate_oldest(self, n: int):
        key = lambda x: x.age
        self.population.sort(key=key, reverse=True)

        del self.population[:n]
        gc.collect()

    def _eliminate_worst(self, n: int):
        key = lambda x: self._evaluate(x)
        self.population.sort(key=key)

        del self.population[:n]
        gc.collect()

    def _evaluate(self, ind: BaseIndividual) -> float:
        if ind.performance is None:
            train_pred = np.array([ind.predict(x) for x in self.dataset.X_train])
            test_pred = np.array([ind.predict(x) for x in self.dataset.X_test])

            p = Performance(
                train_loss=self._bce(self.dataset.y_train, train_pred),
                test_loss=self._bce(self.dataset.y_test, test_pred),
                train_acc=self._acc(self.dataset.y_train, train_pred),
                test_acc=self._acc(self.dataset.y_test, test_pred),
            )
            ind.set_performance(p)

        return ind.fitness

    @staticmethod
    def _acc(y_true: np.ndarray, y_hat: np.ndarray) -> float:
        assert len(y_true.shape) == len(y_hat.shape) == 1
        y_hat = y_hat > 0.5
        return accuracy_score(y_true, y_hat)

    @staticmethod
    def _bce(y_true: np.ndarray, y_hat: np.ndarray) -> float:
        assert len(y_true.shape) == len(y_hat.shape) == 1
        return log_loss(y_true, y_hat)


class GeneticAlgorithm(EC):
    ELIMINATE_BY_AGE = "age"
    ELIMINATE_BY_FITNESS = "fitness"

    def __init__(
        self,
        pop_sz: int,
        offspr_sz: int,
        recomb_prob: float,
        mutate_prob: float,
        eliminate_by: str,
        select_callback: Callable,
        select_kwargs: dict,
        dataset: Dataset,
        eval_by: str,
        dim_list: list[int],
        **operator_callbacks,
    ):
        assert pop_sz > offspr_sz
        assert 0 <= recomb_prob <= 1
        assert 0 <= mutate_prob <= 1
        assert eliminate_by in (self.ELIMINATE_BY_AGE, self.ELIMINATE_BY_FITNESS)

        self.recomb_prob: float = recomb_prob
        self.mutate_prob: float = mutate_prob
        self.eliminate_by = eliminate_by
        self.select_callback: Callable = select_callback
        self.select_kwargs: dict = select_kwargs
        self.population: list[GaIndividual] = [
            GaIndividual(
                eval_by=eval_by,
                dim_list=dim_list,
                **operator_callbacks,
            ) for _ in range(pop_sz)
        ]

        super().__init__(
            pop_sz=pop_sz,
            offspr_sz=offspr_sz,
            dataset=dataset,
        )

    def _evolve(self):
        offspr_lst = []
        while len(offspr_lst) < self.offspr_sz:
            p1 = self.select_callback(self.population, **self.select_kwargs)
            p2 = self.select_callback(self.population, **self.select_kwargs)

            if random.random() > self.recomb_prob:
                offspr_lst += [GaIndividual.copy(p1), GaIndividual.copy(p2)]
            else:
                c1, c2 = p1.recombine(p2)
                offspr_lst += [c1, c2]

        if self.eliminate_by == self.ELIMINATE_BY_AGE:
            self._eliminate_oldest(self.offspr_sz)
        else:
            self._eliminate_worst(self.offspr_sz)

        for ind in self.population:
            ind.increment_age()
            if random.random() <= self.mutate_prob:
                ind.mutate()

        self.population += offspr_lst


class EvolutionaryStrategy(EC):
    ES_COMMA = "comma"
    ES_PLUS = "plus"

    def __init__(
        self,
        pop_sz: int,
        offspr_sz: int,
        select_by: str,
        dataset: Dataset,
        eval_by: str,
        dim_list: list[int],
        **mutate_kwargs,
    ):
        """
        mutate_kwargs:
            - step_size
            - tau: coordinate-wise learning rate
            - tau_prime: general learning rate
            - eps: min step size
        """
        assert pop_sz <= offspr_sz
        assert select_by in (self.ES_COMMA, self.ES_PLUS)

        self.select_by: str = select_by
        self.population: list[EsIndividual] = [
            EsIndividual(
                eval_by=eval_by,
                dim_list=dim_list,
                **mutate_kwargs,
            ) for _ in range(pop_sz)
        ]

        super().__init__(
            pop_sz=pop_sz,
            offspr_sz=offspr_sz,
            dataset=dataset,
        )

    def _evolve(self):
        parents = random.choices(self.population, k=self.offspr_sz)
        offspr_lst = [ind.mutate() for ind in parents]

        if self.select_by == self.ES_COMMA:
            del self.population
            self.population = offspr_lst
            self._eliminate_worst(len(self.population) - self.pop_sz)
        else:
            self.population += offspr_lst
            self._eliminate_worst(len(self.population) - self.pop_sz)
