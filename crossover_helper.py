import random

import numpy as np


class CrossoverHelper:

    # ==================================== Any Dimension =====================================

    @classmethod
    def uniform(cls, arr1: np.ndarray, arr2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert arr1.shape == arr2.shape

        mask = np.random.randint(low=0, high=2, size=arr1.shape) # [low, high)

        child1 = arr1 * mask + arr2 * (1 - mask)
        child2 = arr1 * (1 - mask) + arr2 * mask

        return child1, child2

    @classmethod
    def one_point(cls, arr1: np.ndarray, arr2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert arr1.shape == arr2.shape

        l = 1
        for d in arr1.shape: l *= d

        p = random.randint(0, l-1)
        mask = np.concatenate([np.ones(p), np.zeros(l-p)])
        mask = np.reshape(mask, arr1.shape)

        child1 = arr1 * mask + arr2 * (1 - mask)
        child2 = arr1 * (1 - mask) + arr2 * mask

        return child1, child2

    @classmethod
    def two_point(cls, arr1: np.ndarray, arr2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert arr1.shape == arr2.shape

        l = 1
        for d in arr1.shape: l *= d

        p1, p2 = sorted(random.sample(range(l), k=2))
        mask = np.concatenate([np.ones(p1), np.zeros(p2-p1), np.ones(l-p2)])
        mask = np.reshape(mask, arr1.shape)

        child1 = arr1 * mask + arr2 * (1 - mask)
        child2 = arr1 * (1 - mask) + arr2 * mask

        return child1, child2

    # ========================================== 2D ==========================================

    @classmethod
    def submat_2d(cls, mat1: np.ndarray, mat2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert mat1.shape == mat2.shape
        assert len(mat1.shape) == 2

        """         
                      A1 | B1             A2 | B2
              mat1 =  ---+---  ,   mat2 = ---+---
                      C1 | D1             C2 | D2

                      A1 | B2             A2 | B1
            child1 =  ---+---  , child2 = ---+---
                      C2 | D1             C1 | D2
        """

        r = random.randint(0, mat1.shape[0]-1)
        c = random.randint(0, mat1.shape[1]-1)

        child1 = np.copy(mat2)
        child1[:r, :c] = np.copy(mat1[:r, :c]) # A
        child1[r:, c:] = np.copy(mat1[r:, c:]) # D

        child2 = np.copy(mat1)
        child2[:r, :c] = np.copy(mat2[:r, :c]) # A
        child2[r:, c:] = np.copy(mat2[r:, c:]) # D

        return child1, child2

    @classmethod
    def row_2d(cls, mat1: np.ndarray, mat2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert mat1.shape == mat2.shape
        assert len(mat1.shape) == 2

        """
            child1 = [ mat1[:r, :], mat2[r:, :] ]
            child2 = [ mat2[:r, :], mat1[r:, :] ]
        """

        r = random.randint(0, mat1.shape[0]-1)

        child1 = np.copy(mat1)
        child1[r:, :] = np.copy(mat2[r:, :])

        child2 = np.copy(mat2)
        child2[r:, :] = np.copy(mat1[r:, :])

        return child1, child2

    @classmethod
    def col_2d(cls, mat1: np.ndarray, mat2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert mat1.shape == mat2.shape
        assert len(mat1.shape) == 2

        """         
            child1 = [ mat1[:, :c], mat2[:, c:] ]
            child2 = [ mat2[:, :c], mat1[:, c:] ]
        """

        c = random.randint(0, mat1.shape[1]-1)

        child1 = np.copy(mat1)
        child1[:, c:] = np.copy(mat2[:, c:])

        child2 = np.copy(mat2)
        child2[:, c:] = np.copy(mat1[:, c:])

        return child1, child2
