import random

import numpy as np


class MutateHelper:
    """ ALL METHODS UPDATE INPLACE """

    @classmethod
    def add_noise(cls, x: np.ndarray, sigma: float=0.1):
        mu = 0
        x += np.random.normal(mu, sigma, size=x.shape)

    # ========================================== 1D ==========================================

    @classmethod
    def swap_1d(cls, x: np.ndarray):
        assert len(x.shape) == 1

        i, j = random.sample(range(len(x)), k=2)
        x[i], x[j] = x[j], x[i]

    # ========================================== 2D ==========================================

    @classmethod
    def swap_rows_2d(cls, x: np.ndarray):
        assert len(x.shape) == 2

        idx_1, idx_2 = random.sample(range(x.shape[0]), 2)
        x[[idx_1, idx_2]] = x[[idx_2, idx_1]]

    @classmethod
    def swap_cols_2d(cls, x: np.ndarray):
        assert len(x.shape) == 2

        idx_1, idx_2 = random.sample(range(x.shape[1]), 2)
        x[:, [idx_1, idx_2]] = x[:, [idx_2, idx_1]]

    @classmethod
    def swap_submat_2d(cls, x: np.ndarray):
        assert len(x.shape) == 2

        def get_boundary_0(rows: int, cols: int) -> tuple:
            """
                M1 ∈ x[:r, :]
                M2 ∈ x[r:, :]
            """
            r = random.randint(1, rows-1)
            height = random.randint(1, min(r, rows-r))
            width = random.randint(1, cols)

            m1_left = random.randint(0, cols-width)
            m1_top = random.randint(0, r-height)

            m2_left = random.randint(0, cols-width)
            m2_top = random.randint(r, rows-height)

            return m1_left, m1_top, m2_left, m2_top, height, width

        def get_boundary_1(rows: int, cols: int) -> tuple:
            """
                M1 ∈ x[:, :c]
                M2 ∈ x[:, c:]
            """
            c = random.randint(1, cols-1)
            width = random.randint(1, min(c, cols-c))
            height = random.randint(1, rows)

            m1_left = random.randint(0, c-width)
            m1_top = random.randint(0, rows-height)

            m2_left = random.randint(c, cols-width)
            m2_top = random.randint(0, rows-height)

            return m1_left, m1_top, m2_left, m2_top, height, width

        if random.randint(0, 1) == 0:
            m1_left, m1_top, m2_left, m2_top, height, width = get_boundary_0(*x.shape)
        else:
            m1_left, m1_top, m2_left, m2_top, height, width = get_boundary_1(*x.shape)

        sub_mat_1 = np.copy(x[m1_top : m1_top+height, m1_left : m1_left+width])
        sub_mat_2 = np.copy(x[m2_top : m2_top+height, m2_left : m2_left+width])

        x[m1_top : m1_top+height, m1_left : m1_left+width] = sub_mat_2
        x[m2_top : m2_top+height, m2_left : m2_left+width] = sub_mat_1
