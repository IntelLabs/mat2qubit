# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Tests for qsymbop2dlev.py

2019-10
"""

import unittest

import numpy as np

from dLevelSystemEncodings import compositeDLevels, compositeOperator

# from mat2qubit import compositeDLevels,compositeOperator
import qSymbOp

from qsymbop2dlev import symbop_to_dlevcompositeop, symbop_pauli_to_mat

import qopmats

# from mat2qubit import qopmats

from functools import reduce


class symb2qubit_test(unittest.TestCase):
    def setUp(s):
        s.globalop1 = qSymbOp.qSymbOp("1 [q_1 p_v2]")
        s.globalop2 = qSymbOp.qSymbOp("-kk [q_1 p_v2]")
        s.globalop3 = qSymbOp.qSymbOp("[q_1 p_v2] ++ 1.4 [q_1 n_v3]")

        # Include identity too

        s.X = qopmats.qub["X"]
        s.Y = qopmats.qub["Y"]
        s.Z = qopmats.qub["Z"]
        s.I = qopmats.qub["I2"]

    def test_conversion(s):

        ssid_order = [
            "1",
            "v2",
            "v3",
        ]  # List or tuple. Will define ssid --> numeric ssid
        # ssid_to_d = ['1':2,'v2':4,'v3':4]
        ssid_to_d = dict(zip(ssid_order, [2, 4, 4]))
        ssid_to_enc = dict(zip(ssid_order, ["stdbinary", "unary", "unary"]))

        opchars_to_op = {}  # Should let it be other string, op, or np.array. There are
        # built-in functions, so this one will be optional.

        # symb_scalars = {} # The scalars to convert. This should be done *beforehand*.
        # # Hence shouldn't be testing this in this module.

        # *** Test globalop1 ***
        dlevCompOp = symbop_to_dlevcompositeop(
            s.globalop1, ssid_order, ssid_to_d, ssid_to_enc
        )

        s.assertIsInstance(dlevCompOp, compositeOperator)

        s.assertEqual(str(dlevCompOp), "[(1.0, ((0, 'q'), (1, 'qhoMom')))]")

        # *** Test globalop2 ***
        # Symbolic (includes 'kk'), so should raise exception
        with s.assertRaises(TypeError):
            symbop_to_dlevcompositeop(s.globalop2, ssid_order, ssid_to_d, ssid_to_enc)

        # *** Test globalop3 ***
        dlevCompOp = symbop_to_dlevcompositeop(
            s.globalop3, ssid_order, ssid_to_d, ssid_to_enc
        )

        # print(dlevCompOp)
        s.assertEqual(
            str(dlevCompOp),
            "[(1.0, ((0, 'q'), (1, 'qhoMom'))), (1.4, ((0, 'q'), (2, 'numop')))]",
        )

        # *** Test identity - alone ***
        dlevCompOp_ident = symbop_to_dlevcompositeop(
            qSymbOp.qSymbOp("7 [ ]"), [], {}, {}
        )
        s.assertEqual(str(dlevCompOp_ident), "[(7.0, 'ident')]")

        # *** Test identity - with other ops ***
        dlevCompOp = symbop_to_dlevcompositeop(
            qSymbOp.qSymbOp("7 [ ] ++ [q_1 p_v2]"), ssid_order, ssid_to_d, ssid_to_enc
        )
        s.assertEqual(
            str(dlevCompOp), "[(7.0, 'ident'), (1.0, ((0, 'q'), (1, 'qhoMom')))]"
        )

        # If you wanted to push to Pauli, you'd do this:
        # print( dlevCompOp.opToPauli() )

        # *** Test all-the-same-value inputs for d and enc
        op = qSymbOp.qSymbOp("[q_a p_b] ++ 1.4 [q_b n_c]")
        dlevCompOp = symbop_to_dlevcompositeop(op, ["a", "b", "c"], 4, "gray")
        s.assertEqual(
            str(dlevCompOp),
            "[(1.0, ((0, 'q'), (1, 'qhoMom'))), (1.4, ((1, 'q'), (2, 'numop')))]",
        )
        s.assertEqual(dlevCompOp.subsystems[1].d, 4)
        s.assertEqual(dlevCompOp.subsystems[1].enc, "gray")

        # *** Test inputting lists for d and enc
        op = qSymbOp.qSymbOp("[q_a p_b] ++ 1.4 [q_b n_c]")
        dlevCompOp = symbop_to_dlevcompositeop(
            op, ["a", "b", "c"], [4, 5, 6], ["gray", "unary", "stdbinary"]
        )
        s.assertEqual(
            str(dlevCompOp),
            "[(1.0, ((0, 'q'), (1, 'qhoMom'))), (1.4, ((1, 'q'), (2, 'numop')))]",
        )
        s.assertEqual(dlevCompOp.subsystems[1].d, 5)
        s.assertEqual(dlevCompOp.subsystems[1].enc, "unary")
        s.assertEqual(dlevCompOp.subsystems[2].d, 6)
        s.assertEqual(dlevCompOp.subsystems[2].enc, "stdbinary")

    # def test_identity(s):

    #   ident = qSymbOp.qSymbOp('-7 []')
    #   op_with_ident = qSymbOp.qSymbOp('-kk [q_1 p_v2] ++ 7 []')
    #   ssid_order = ['1','v2']
    #   ssid_to_d = dict(zip(ssid_order,[2,4,4]))
    #   ssid_to_enc = dict(zip(ssid_order,['stdbinary','unary','unary']))

    #   dlevCompOp = symbop_to_dlevcompositeop(s.globalop1, ssid_order,ssid_to_d,ssid_to_enc)

    def test_symbop_pauli_to_mat(s):

        # op1 = qSymbOp.qSymbOp('2.1 [X_0 Y_2 Z_3]')
        # op2 = qSymbOp.qSymbOp('7 [ ] ++ [X_1 Y_1]')
        # op3 = qSymbOp.qSymbOp('7 [ ] ++ [X_1 Y_1]')
        # op4 = qSymbOp.qSymbOp('7 [ ] ++ [X_1 Y_1]')

        res1 = symbop_pauli_to_mat(
            "2.1 [X_0 Y_2 Z_3]", [str(i) for i in range(4)]
        ).toarray()
        res2 = symbop_pauli_to_mat(
            "7 [ ] ++ [X_1 Y_1]", [str(i) for i in range(2)]
        ).toarray()
        # res1 = symbop_pauli_to_mat(op1, list(range(4)) )
        # res2 = symbop_pauli_to_mat(op2, list(range(2)) )

        # gold1 = 2.1*reduce(np.kron, (s.X,s.I,s.Y,s.Z) )
        # gold2 = 7*reduce(np.kron, (s.I,s.I) ) + 1*np.kron( s.I , np.dot(s.X,s.Y)  )

        gold1 = 2.1 * reduce(np.kron, (s.Z, s.Y, s.I, s.X))
        gold2 = 7 * reduce(np.kron, (s.I, s.I)) + 1 * np.kron(np.dot(s.X, s.Y), s.I)

        np.testing.assert_almost_equal(res1, gold1, 12)
        np.testing.assert_almost_equal(res2, gold2, 12)


if __name__ == "__main__":
    unittest.main()
