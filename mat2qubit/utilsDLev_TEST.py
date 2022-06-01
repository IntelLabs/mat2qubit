# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# dLevelUtils_TEST.py

import functools
import unittest
from io import StringIO

import numpy as np
import scipy.linalg as la
import scipy.sparse as spr
from ns_primitives import dotx, krx
from openfermion import (
    BosonOperator,
    QuadOperator,
    QubitOperator,
    qubit_operator_sparse,
)

import qopmats
import utilsDLev
from dLevelSystemEncodings import compositeDLevels, compositeOperator, dLevelSubsystem
from helperDLev import sglQubOp

# from mat2qubit.dLevelSystemEncodings import dLevelSubsystem,compositeDLevels,compositeOperator#,compositeQasmBuilder
# import mat2qubit.utilsDLev as utilsDLev
# from mat2qubit.helperDLev import sglQubOp#, trotterQubop2qasm
# import mat2qubit.qopmats as qopmats




# import basic_qcircuit as bqc




class dlevelutils_tests(unittest.TestCase):

    # static variables declared here
    # encodings = ["","","",""]

    def setUp(self):
        pass

    def test_int_and_bin_strings(s):

        # *** bitstring_to_state_id(bitstring) ***

        res = utilsDLev.bitstring_to_state_id([0, 1])  # least-sig fig listed *first*
        res2 = utilsDLev.bitstring_to_state_id([0, 1, 0])  # should be same
        res3 = utilsDLev.bitstring_to_state_id([0, 1, 0, 0])  # should be same
        gold = 2
        s.assertEqual(gold, res)
        s.assertEqual(gold, res2)
        s.assertEqual(gold, res3)
        res = utilsDLev.bitstring_to_state_id([1, 1, 1, 0])
        gold = 7
        s.assertEqual(gold, res)

        # *** integerstring_to_bitstring(intstring,sysdlev) ***
        encs = ("gray", "unary", "stdbinary")
        ds = (3, 4, 4)
        compos_op = compositeOperator()
        compos_op.appendSubsystem(dLevelSubsystem(d=ds[0], enc=encs[0]))
        compos_op.appendSubsystem(dLevelSubsystem(d=ds[1], enc=encs[1]))
        compos_op.appendSubsystem(dLevelSubsystem(d=ds[2], enc=encs[2]))
        int_string = (2, 2, 2)
        res = utilsDLev.integerstring_to_bitstring(int_string, compos_op)
        gold_bitstring = (1, 1, 0, 0, 1, 0, 0, 1)
        s.assertEqual(gold_bitstring, res)

        # Raise error if the number is bigger than the limit for that subsystem
        ints_outofbounds = (3, 2, 2)  # Value of 3 is out of bounds for a 3-level system
        s.assertRaises(
            IndexError,
            utilsDLev.integerstring_to_bitstring,
            ints_outofbounds,
            compos_op,
        )

        # I guess this IndexError should come from the composite operator itself?
        # (NO. MAYBE NOT ACTUALLY. )

        # *** integerstring_to_state_id() ***
        gold_val = int("10" + "0100" + "11", 2)
        res_stateid = utilsDLev.integerstring_to_state_id(int_string, compos_op)
        s.assertEqual(gold_val, res_stateid)

        # ***
        # Second level of verification using projector ops,
        # just to make sure we're implicitly matching m2q code.
        # Only works with compact codes though.
        # Creating projector, comparing to value.
        encs = ("stdbinary", "gray")  # ,'stdbinary')
        ds = (3, 4)
        compos_op = compositeOperator()
        compos_op.appendSubsystem(dLevelSubsystem(d=ds[0], enc=encs[0]))
        compos_op.appendSubsystem(dLevelSubsystem(d=ds[1], enc=encs[1]))
        int_string = (0, 2)
        # gold_bitstring = ( 1,1,  0,0,1,0,  0,1 )

        # opstr = ((0,"Pr{}".format(int_string[0])),)

        gold_val = utilsDLev.bitstring_to_state_id((0, 0, 1, 1))
        print("gold_val: {}".format(gold_val))
        opstr = ((0, "Pr{}".format(int_string[0])), (1, "Pr{}".format(int_string[1])))
        print(opstr)

        # projpop = compos_op.opStringToPauli( 2. , opstr )
        # projector = qubit_operator_sparse(projpop) ******
        projector = compos_op.opStringToMatRep(1, opstr)

        print(projector)
        print(type(projector))
        print(gold_val)
        assert (
            len(projector.indices) == 1
        ), (
            projector.indices
        )  # Just making sure it's really a one-val projector (only true in dense case)
        m2q_stateid = projector.indices[0]

        s.assertEqual(gold_val, m2q_stateid)

        # Much simpler projector test
        print("******")
        encs = ("stdbinary", "stdbinary", "stdbinary")
        ds = (2, 2, 2)
        compos_op = compositeOperator()
        compos_op.appendSubsystem(dLevelSubsystem(d=ds[0], enc=encs[0]))
        compos_op.appendSubsystem(dLevelSubsystem(d=ds[1], enc=encs[1]))
        compos_op.appendSubsystem(dLevelSubsystem(d=ds[2], enc=encs[2]))
        int_string = (0, 0, 1)
        gold_val = utilsDLev.bitstring_to_state_id(int_string)
        print("gold_val: {}".format(gold_val))
        opstr = (
            (0, "Pr{}".format(int_string[0])),
            (1, "Pr{}".format(int_string[1])),
            (2, "Pr{}".format(int_string[2])),
        )
        print(opstr)

        projector = compos_op.opStringToMatRep(1, opstr)
        print(projector)
        print(type(projector))
        print(gold_val)
        assert (
            len(projector.indices) == 1
        ), (
            projector.indices
        )  # Just making sure it's really a one-val projector (only true in dense case)
        m2q_stateid = projector.indices[0]

        s.assertEqual(gold_val, m2q_stateid)

    def test_intstrings_to_state(s):

        # Test simple of |0>+|1>+|2>. More complex cases are tested in test_permutations anyway.

        int_strings = (
            (0,),
            (1,),
            (2,),
        )
        d = 3

        # Unary
        psi_notnrml = np.array([0, 1, 1, 0, 1, 0, 0, 0.0])  # 001,010,100
        psi = psi_notnrml / np.sqrt(3)
        enc = "unary"
        res_nn = utilsDLev.int_strings_to_psi_notnormalized(int_strings, d, enc)
        res = utilsDLev.int_strings_to_psi(int_strings, d, enc)
        np.testing.assert_array_almost_equal(psi_notnrml, res_nn)
        np.testing.assert_array_almost_equal(psi, res)

        # SB
        psi_notnrml = np.array([1, 1, 1, 0.0])  # 00,01,10 (reverse indexing)
        psi = psi_notnrml / np.sqrt(3)
        enc = "stdbinary"
        res_nn = utilsDLev.int_strings_to_psi_notnormalized(int_strings, d, enc)
        res = utilsDLev.int_strings_to_psi(int_strings, d, enc)
        np.testing.assert_array_almost_equal(psi_notnrml, res_nn)
        np.testing.assert_array_almost_equal(psi, res)
        # (quick test of projector)
        proj_gold = np.diag([1, 1, 1, 0])
        proj_res = utilsDLev.int_strings_to_projector(int_strings, d, enc).toarray()
        np.testing.assert_array_almost_equal(proj_gold, proj_res)

        # Gray
        psi_notnrml = np.array([1, 1, 0, 1.0])  # 00,01,11 (reverse indexing)
        psi = psi_notnrml / np.sqrt(3)
        enc = "gray"
        res_nn = utilsDLev.int_strings_to_psi_notnormalized(int_strings, d, enc)
        res = utilsDLev.int_strings_to_psi(int_strings, d, enc)
        np.testing.assert_array_almost_equal(psi_notnrml, res_nn)
        np.testing.assert_array_almost_equal(psi, res)

        # *****
        # Slightly more complex example: two particles.
        # Gray:
        # 0 -->   [0,0]
        # 1 -->   [1,0]
        # 2 -->   [1,1]
        # 0,2 --> [0,0, 1,1] is 12
        # 1,2 --> [1,0, 1,1] is 13
        int_strings = ((0, 2), (1, 2))
        d = 3
        psi_notnrml = np.zeros(16, dtype=float)
        psi_notnrml[12] = 1
        psi_notnrml[13] = 1
        psi = psi_notnrml / np.sqrt(2)
        enc = "gray"
        res_nn = utilsDLev.int_strings_to_psi_notnormalized(int_strings, d, enc)
        res = utilsDLev.int_strings_to_psi(int_strings, d, enc)
        np.testing.assert_array_almost_equal(psi_notnrml, res_nn)
        np.testing.assert_array_almost_equal(psi, res)

    def test_permutations(s):

        # *** permutation_superposition( sites , d , enc ) ***
        # *** permutation_projector( sites , d , enc ) ***

        # ***** Test 1 - simplest
        sites = 2
        d = 2
        gold_state_nn = np.array([0, 1, 1, 0.0])
        gold_state = gold_state_nn / np.sqrt(2)
        res_state = utilsDLev.permutation_superposition(sites, d, "stdbinary")
        np.testing.assert_array_almost_equal(gold_state, res_state)
        gold_projector = np.diag(gold_state_nn)
        res_projector = utilsDLev.permutation_projector(sites, d, "stdbinary")
        np.testing.assert_array_equal(gold_projector, res_projector.todense())

        # s.assertAlmostEqual(gold_projector,res_projector)

        # *** 3 sites, d=2 ***
        sites = 2
        d = 3
        # ***** Test 2 - Standard binary
        ids = []
        ids.append(utilsDLev.bitstring_to_state_id([0, 0, 1, 0]))  # 0,1
        ids.append(utilsDLev.bitstring_to_state_id([1, 0, 0, 0]))  # 1,0
        ids.append(utilsDLev.bitstring_to_state_id([0, 0, 0, 1]))  # 0,2
        ids.append(utilsDLev.bitstring_to_state_id([0, 1, 0, 0]))  # 2,0
        ids.append(utilsDLev.bitstring_to_state_id([1, 0, 0, 1]))  # 1,2
        ids.append(utilsDLev.bitstring_to_state_id([0, 1, 1, 0]))  # 2,1
        gold_state_nn = np.zeros(2**4)
        for i in ids:
            gold_state_nn[i] = 1
        gold_state = gold_state_nn / la.norm(gold_state_nn)
        res_state = utilsDLev.permutation_superposition(sites, d, "stdbinary")
        np.testing.assert_array_almost_equal(gold_state, res_state)
        gold_projector = np.diag(gold_state_nn)
        res_projector = utilsDLev.permutation_projector(sites, d, "stdbinary")
        np.testing.assert_array_equal(gold_projector, res_projector.todense())

        # ***** Test 3 - gray
        ids = []
        ids.append(utilsDLev.bitstring_to_state_id([0, 0, 1, 0]))  # 0,1
        ids.append(utilsDLev.bitstring_to_state_id([1, 0, 0, 0]))  # 1,0
        ids.append(utilsDLev.bitstring_to_state_id([0, 0, 1, 1]))  # 0,2
        ids.append(utilsDLev.bitstring_to_state_id([1, 1, 0, 0]))  # 2,0
        ids.append(utilsDLev.bitstring_to_state_id([1, 0, 1, 1]))  # 1,2
        ids.append(utilsDLev.bitstring_to_state_id([1, 1, 1, 0]))  # 2,1
        gold_state_nn = np.zeros(2**4)
        for i in ids:
            gold_state_nn[i] = 1
        gold_state = gold_state_nn / la.norm(gold_state_nn)
        res_state = utilsDLev.permutation_superposition(sites, d, "gray")
        np.testing.assert_array_almost_equal(gold_state, res_state)
        gold_projector = np.diag(gold_state_nn)
        res_projector = utilsDLev.permutation_projector(sites, d, "gray")
        np.testing.assert_array_equal(gold_projector, res_projector.todense())

        # ***** Test 4 - unary
        ids = []
        ids.append(utilsDLev.bitstring_to_state_id([1, 0, 0, 0, 1, 0]))  # 0,1
        ids.append(utilsDLev.bitstring_to_state_id([0, 1, 0, 1, 0, 0]))  # 1,0
        ids.append(utilsDLev.bitstring_to_state_id([1, 0, 0, 0, 0, 1]))  # 0,2
        ids.append(utilsDLev.bitstring_to_state_id([0, 0, 1, 1, 0, 0]))  # 2,0
        ids.append(utilsDLev.bitstring_to_state_id([0, 1, 0, 0, 0, 1]))  # 1,2
        ids.append(utilsDLev.bitstring_to_state_id([0, 0, 1, 0, 1, 0]))  # 2,1
        gold_state_nn = np.zeros(2**6)
        for i in ids:
            gold_state_nn[i] = 1
        gold_state = gold_state_nn / la.norm(gold_state_nn)
        res_state = utilsDLev.permutation_superposition(sites, d, "unary")
        np.testing.assert_array_almost_equal(gold_state, res_state)
        gold_projector = np.diag(gold_state_nn)
        res_projector = utilsDLev.permutation_projector(sites, d, "unary")
        np.testing.assert_array_equal(gold_projector, res_projector.todense())

        """
        3x3:
        1,2,3
        2,1,3
        3,1,2
        1,3,2
        2,3,1
        3,2,1
        """

    def test_braket(s):

        filecontents1 = """NUMREGISTERS 2
REGSIZES 2,2
5.6 |0:0><0:0|
5.6 |0:1,1:0><0:1,1:0|
BREAK
1.1 |0:1,1:0><0:0,1:1|
1.1 |0:0,1:1><0:1,1:0|
"""

        d = 2

        stringStream1 = StringIO(filecontents1)
        # stringStream1.write(filecontents1)

        # Parse "file" to get composite operator object
        composDOp = utilsDLev.parseBraket(stringStream1)

        manualComposDOp = compositeOperator()
        manualComposDOp.appendSubsystem(dLevelSubsystem(d))
        manualComposDOp.appendSubsystem(dLevelSubsystem(d))

        op_00 = qopmats.singleElement(d, 0, 0)
        op_01 = qopmats.singleElement(d, 0, 1)
        op_10 = qopmats.singleElement(d, 1, 0)
        op_11 = qopmats.singleElement(d, 1, 1)

        manualComposDOp.addHamTerm(5.6, [(0, op_00)])
        manualComposDOp.addHamTerm(5.6, [(0, op_11), (1, op_00)])
        manualComposDOp.addHamTerm(1.1, [(0, op_10), (1, op_01)])
        manualComposDOp.addHamTerm(1.1, [(0, op_01), (1, op_10)])

        # Gold matrix op
        goldMatOp = 5.6 * krx(np.array([[1, 0], [0, 0]], dtype=complex), np.eye(d))
        goldMatOp += 5.6 * krx(np.array([[0, 0], [0, 1]]), np.array([[1, 0], [0, 0]]))
        goldMatOp += 1.1 * krx(np.array([[0, 0], [1, 0]]), np.array([[0, 1], [0, 0]]))
        goldMatOp += 1.1 * krx(np.array([[0, 1], [0, 0]]), np.array([[0, 0], [1, 0]]))

        # *** Now do with the QASM-builder ***
        stringStream1 = StringIO(filecontents1)
        qasmBuilder = utilsDLev.parseBraketToQasmBuilder(stringStream1)
        for action in qasmBuilder.circCommands:
            print(action)

        filecontents2 = """NUMREGISTERS 2
REGSIZES 2,2
5.60216 |0:0><0:0|
5.60216 |0:0,1:0><0:0,1:0|
11.2043 |0:0,1:0><0:0,1:0|
BREAK
1.19186 |1:1><1:1|
1.19186 |0:1,1:1><0:1,1:1|
2.38371 |0:1,1:1><0:1,1:1|
BREAK
1.11072 |0:1,1:1><0:0,1:1|
1.11072 |0:0,1:1><0:0,1:0|
1.11072 |0:0,1:0><0:0,1:1|
1.11072 |0:0,1:1><0:1,1:1|
BREAK
2.50076 |0:0><0:0|
2.50076 |0:0,1:0><0:0,1:0|
5.00152 |0:0,1:0><0:0,1:0|
BREAK
4.67163 |1:1><1:1|
4.67163 |0:1,1:1><0:1,1:1|
9.34325 |0:1,1:1><0:1,1:1|
"""


if __name__ == "__main__":
    unittest.main()
