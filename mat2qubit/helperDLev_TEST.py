# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Test helperDLev.py

import functools
import unittest

import numpy as np
from openfermion import QubitOperator

import helperDLev

# from mat2qubit import qopmats
import qopmats


class helperDLev_tests(unittest.TestCase):
    def setup(self):
        pass

    def test_pauli_op_to_matrix(s):

        I = qopmats.qub["I2"]  # noqa: E741
        X = qopmats.qub["X"]
        Y = qopmats.qub["Y"]
        Z = qopmats.qub["Z"]

        # # Zero
        # pop = QubitOperator('')

        # # Identity
        # pop = QubitOperator('[]')

        # # X0
        # pop = QubitOperator('X')

        # # X0 with two qubits
        # pop = QubitOperator('X')

        # # X0 Z1
        # pop = QubitOperator('0.25 [X0 Z1]')

        # # X0 Z1 with three qubits
        # pop = QubitOperator('0.25 [X0 Z1]')
        # nq = 3

        # Multi
        pop = QubitOperator("0.25 [X0 Z1] + 0.5 [Y1 Z2] + 0.125 []")
        nq = 4
        gold = (
            0.25 * functools.reduce(np.kron, (I, I, Z, X))
            + 0.5 * functools.reduce(np.kron, (I, Z, Y, I))
            + 0.125 * functools.reduce(np.kron, (I, I, I, I))
        )
        res = helperDLev.pauli_op_to_matrix(pop, nq).todense()
        print(gold.shape)
        print(res.shape)
        # np.testing.assert_array_equal(gold,res)
        s.assertEqual(gold.tolist(), res.tolist())


if __name__ == "__main__":
    unittest.main()
