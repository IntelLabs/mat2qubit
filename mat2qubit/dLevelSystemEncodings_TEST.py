# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# dLevelSystemEncodings_TEST.py

import unittest

from mat2qubit import dLevelSubsystem,compositeDLevels,compositeOperator #,compositeQasmBuilder
#from dLevelUtils import ofOpToCompositeOp

from openfermion import QubitOperator, QuadOperator, BosonOperator

#from of_util import trotterQubop2qasm
from mat2qubit.helperDLev import sglQubOp,pauli_op_to_matrix
import qopmats

# import basic_qcircuit as bqc
import functools
import scipy.sparse as spr

import numpy as np

# NOTE: OpenFermion's __eq__ between operators already does almost-equal

class dlevel_tests(unittest.TestCase):
    
    # static variables declared here
    # encodings = ["","","",""]

    def setup(self):
        pass

    def test_simpleops(s):
        
        ii = QubitOperator('')
        x0 = QubitOperator('X0')
        y0 = QubitOperator('Y0')
        z0 = QubitOperator('Z0')
        x1 = QubitOperator('X1')
        y1 = QubitOperator('Y1')
        z1 = QubitOperator('Z1')
        x2 = QubitOperator('X2')
        y2 = QubitOperator('Y2')
        z2 = QubitOperator('Z2')
        
        miscOp = np.array( [[7,0,1+2j], [0,0,0], [3+4j,0,9]] , dtype=complex )
        
        testprec = 12

        # For the first set, enter gold pauli ops manually. then use it to test of_util. then later you can 
        # just use of_util directly.


        ss1 = dLevelSubsystem( d=3 , enc="stdbinary" )
        gold = .25*(x0-1j*y0)*(ii+z1) + 0.25*np.sqrt(2.)* (x0+1j*y0)*(x1-1j*y1)
        res = ss1.opToPauli("qhoCr")
        # print(gold)
        # print()
        # print(res)
        s.assertEqual( res , gold )
        s.assertEqual( ss1.nqubits , 2 )

        # ** Test inputting matrep instead of character string
        res = ss1.opToPauli(np.diag([0.,1,2]))
        gold = 0.75*ii + 0.25*z0 - 0.75*z0*z1 - 0.25*z1
        s.assertEqual( res , gold )

        
        ss1.setEncoding("gray")
        # not completed

        ss1.setEncoding("unary")
        gold = .25*(x1-1j*y1)*(x0+1j*y0)  + .25*np.sqrt(2.)* (x2-1j*y2)*(x1+1j*y1)
        res = ss1.opToPauli("qhoCr")
        s.assertEqual( res , gold )
        s.assertEqual( ss1.nqubits, 3 )

        # ** Test inputting matrep instead of character string
        # res = ss1.opToPauli(np.diag([0.,1,2]))
        res = ss1.opToPauli( 'numop' )
        gold = ( 3*ii - 1*z1 - 2*z2 ) / 2
        s.assertEqual( res , gold )


        # Test locopProductToPauli
        # The code first multiplies the matrix reps of q*q=q^2
        # Takes resulting matrix rep, and maps that to Pauli
        resQsq  = ss1.locopProductToPauli( ("qhoPos","qhoPos") )
        goldQsq = 1.*sglQubOp(1,1,  0) + 3.*sglQubOp(1,1,  1) + 5.*sglQubOp(1,1,  2) + \
            np.sqrt(2.) * ( sglQubOp(1,0, 2)*sglQubOp(0,1, 0) + sglQubOp(1,0, 0)*sglQubOp(0,1, 2)  ) 
        s.assertEqual( res, gold )



        # *********
        # Now test locopProductToPauli() for non-commuting cases
        # a*ad
        ss2 = dLevelSubsystem(d=2,enc="stdbinary")
        ss2.setQubitShift(7)
        opstr_a_ad = ("qhoAn","qhoCr")
        res_a_ad = ss2.locopProductToPauli(opstr_a_ad)
        gold_a_ad = QubitOperator('0.5 [] + 0.5 [Z7]')
        s.assertEqual( res_a_ad, gold_a_ad )

        # ad*a
        opstr_ad_a = ["qhoCr","qhoAn"]
        res_ad_a = ss2.locopProductToPauli(opstr_ad_a)
        gold_ad_a = QubitOperator('0.5 [] - 0.5[Z7]')
        s.assertEqual( res_ad_a, gold_ad_a)








    
    def test_ss_qub_count(s):
        
        ss = dLevelSubsystem(2)
        ss.setEncoding("blockunary",{'g':3,'localEncodingFunc':"gray"})
        ss.set_d(2)
        s.assertEqual( ss.nqubits,2 )
        ss.set_d(3)
        s.assertEqual( ss.nqubits,2 )
        ss.set_d(4)
        s.assertEqual( ss.nqubits,4 )
        ss.set_d(5)
        s.assertEqual( ss.nqubits,4 )
        ss.set_d(6)
        s.assertEqual( ss.nqubits,4 )
        ss.set_d(7)
        s.assertEqual( ss.nqubits,6 )
        
    # def test_projectors(s):

    #     encs = ('gray','unary','stdbinary')
    #     ds   = (3,4,4)
    #     compos_op = compositeOperator()
    #     compos_op.appendSubsystem( dLevelSubsystem(d=ds[0],enc=


    
    def test_compositesystem(s):
        
        # ** Two-site, two-level systems, a_0^ a_1, unary
        # Create a subsystem
        ss1 = dLevelSubsystem(2, "unary")

        # Create composite system
        twoSite = compositeDLevels()
        twoSite.appendSubsystem(ss1) # Site 1, deepcopied
        twoSite.appendSubsystem(ss1) # Site 2, deepcopied

        # Get a Hamiltonian from a single term (list of tuples)
        opString = [(0,"qhoCr"),(1,"qhoAn")]
        res = twoSite.opStringToPauli(2.0,opString)

        # Gold is |1><0|(x)|2><3| -->  sigP_1 sigN_0 sigP_2 sigN_3
        gold = (2.0) * sglQubOp(1,0,  1)*sglQubOp(0,1,  0) * sglQubOp(1,0,  2)*sglQubOp(0,1,  3)
        
        # print(res)
        # print()
        # print(gold)
        
        s.assertEqual( res,gold )
        
        '''
        # ************************************************
        # Use same system to test gate counts, in qasmBuilder
        # ************************************************
        # Initialize qasm builder
        qasmBuilder = compositeQasmBuilder()
        # Add the qho-based operator from before, and add its conjugate to make hermitian
        qasmBuilder.addHamTerm( 2.0, opString )
        qasmBuilder.addHamTerm( 2.0, [(0,"qhoAn"),(1,"qhoCr")] )
        # Calculate gate count
        ncnot = qasmBuilder.countCnotUBound_IncludeBreaks(twoSite)
        
        # Non-herm had 16 terms, but herm op has 8 terms
        s.assertEqual( ncnot , 6*8 ) # 8 terms, 6 cnots each
        
        '''


        # ************************************************
        # Test compositeOperator(), the child class of compositeDLevels()
        # ************************************************
        # Do it all from scratch. 2-site hubbard model
        hubb2 = compositeOperator()
        id1 = hubb2.appendSubsystem(ss1)
        id2 = hubb2.appendSubsystem(ss1)
        k = 0.3
        hubb2.addHamTerm( k , [(0,"qhoCr"),(1,"qhoAn")] )
        
        resPauli = hubb2.opToPauli()
        gold = k * sglQubOp(1,0,  1)*sglQubOp(0,1,  0) * sglQubOp(1,0,  2)*sglQubOp(0,1,  3)
        s.assertEqual( resPauli, gold )

    def test_getNumQub(s):

        '''
        unary 3  - 3
        stdbin 3 - 2
        gray 4   - 2
        stdbin 5 - 3
        TOTAL    - 10
        '''

        compop = compositeDLevels()
        compop.appendSubsystem( dLevelSubsystem(3,'unary') )
        compop.appendSubsystem( dLevelSubsystem(3,'stdbinary') )
        compop.appendSubsystem( dLevelSubsystem(4,'gray') )
        compop.appendSubsystem( dLevelSubsystem(5,'stdbinary') )

        gold = 10
        res = compop.getNumQub()
        s.assertEqual(gold,res)


    def test_ketbra(s):

        d = 15
        dlev = dLevelSubsystem(d=d)

        resmat = dlev.opToMatrixRep("k2b3")
        gold = np.zeros((d,d))
        gold[2,3] = 1.
        np.testing.assert_array_equal(gold,resmat)

        resmat = dlev.opToMatrixRep("k12b13")
        gold = np.zeros((d,d))
        gold[12,13] = 1.
        np.testing.assert_array_equal(gold,resmat)



    def test_matreps(s):
        # Identity is tested in here as well

        # *** First test ***
        # Two subsystems
        compOp = compositeOperator()
        compOp.appendSubsystem( dLevelSubsystem(3) )
        compOp.appendSubsystem( dLevelSubsystem(3) )

        # Operator = 3*a_0^*a_1 + 2.*I
        opString1 = [(0,"qhoCr"),(1,"qhoAn")]
        compOp.addHamTerm(3.,opString1)
        compOp.addHamTerm(2.,"ident") 

        # Expected result
        ad  = np.zeros((3,3),dtype=complex)
        ad[1,0] = 1.;   ad[2,1] = np.sqrt(2.);
        a = np.zeros((3,3))
        a[0,1] = 1.;   a[1,2] = np.sqrt(2.);
        gold = 3.*np.kron(a,ad) + 2.*np.eye(9)

        # Previous ordering:        
        # # Expected result
        # ad  = np.zeros((3,3),dtype=complex)
        # ad[1,0] = 1.;   ad[2,1] = np.sqrt(2.);
        # a = np.zeros((3,3))
        # a[0,1] = 1.;   a[1,2] = np.sqrt(2.);
        # gold = 3.*np.kron(ad,a) + 2.*np.eye(9)

        # Compare
        res = compOp.toFullMatRep(ignore_encoding=True).todense()
        s.assertEqual( res.tolist() , gold.tolist() )

        # Make sure older ordering *doesn't* work
        falsegold = 3.*np.kron(ad,a) + 2.*np.eye(9)
        s.assertNotEqual( res.tolist() , falsegold.tolist() )



        # *** 2nd test ***
        # Testing multiple ops on same subsystem
        # 'q0 p2 p0 q1 p0'
        compOp = compositeOperator()
        compOp.appendSubsystem( dLevelSubsystem(4) )
        compOp.appendSubsystem( dLevelSubsystem(2) )
        compOp.appendSubsystem( dLevelSubsystem(2) )
        k = 0.1
        opstring = ( (0,'q'), (2,'p'), (0,'p'), (1,'q'), (0,'p') )


        s2 = np.sqrt(2)
        s3 = np.sqrt(3)
        q0 = np.array([ [0, 1, 0, 0],
                        [1, 0,s2, 0],
                        [0,s2, 0,s3],
                        [0, 0,s3, 0]
                        ])/np.sqrt(2)
        p0 = np.array([ [ 0, -1j,  0,  0],
                        [ 1j,  0, -1j*s2,  0],
                        [ 0, 1j*s2,  0, -1j*s3],
                        [ 0,  0, 1j*s3,  0]
                        ])/np.sqrt(2)
        q1 = np.array([[0,1],[1,0.]])/np.sqrt(2)
        p2 = np.array([[0,-1j],[1j,0]])/np.sqrt(2)
        op_ss0 = functools.reduce(np.dot, (q0,p0,p0))
        # op_ss0 = functools.reduce(np.dot, (p0,p0,q0)) # Incorrect order gives different (incorrect) result.

        gold = k * functools.reduce(np.kron, (p2,q1,op_ss0))

        res = compOp.opStringToMatRep(k,opstring, ignore_encoding=True).todense()

        np.testing.assert_array_almost_equal( gold, res )



        # *** 3rd test ***
        # Test with ignore_encoding=False (the default)
        # 0.5 [n_0 n_1] + 0.25 []

        # q0 = q0[:3,:3]
        # n1 = np.diag([0.,1,2]) # 1.5 [] -0.5 [Z3] -1.0 [Z4]
        compOp = compositeOperator()
        compOp.appendSubsystem( dLevelSubsystem(3,'stdbinary') )
        compOp.appendSubsystem( dLevelSubsystem(3,'unary') )
        compOp.addHamTerm( 0.5, ((0,'numop'),(1,'numop'))  )
        compOp.addHamTerm( 0.25, ("ident")  )

        gold =  0.5*QubitOperator("1.5 [] -0.5 [Z3] -1.0 [Z4]")*QubitOperator("0.75 [] + 0.25 [Z0] - 0.75 [Z0 Z1] - 0.25 [Z1]")
        # print('gold:')
        # print(gold)
        # print()
        gold += QubitOperator("0.25 []")
        gold = pauli_op_to_matrix(gold).todense()

        res = compOp.toFullMatRep(ignore_encoding=False).todense()

        np.testing.assert_array_almost_equal( gold, res )




    def test_multiply(s):

        # Two subsystems
        compOp = compositeOperator()
        compOp.appendSubsystem( dLevelSubsystem(3) )
        compOp.appendSubsystem( dLevelSubsystem(3) )
        # Two subsystems in other one as well
        compOp_b = compositeOperator()
        compOp.appendSubsystem( dLevelSubsystem(3) )
        compOp.appendSubsystem( dLevelSubsystem(3) )


        # Operator a = 3*a_0^*a_1 - 2.*I
        opString1 = [(0,"qhoCr"),(1,"qhoAn")]
        compOp.addHamTerm(3.,opString1)
        compOp.addHamTerm(-2.,"ident") 

        # Operator b
        compOp_b.addHamTerm(1., [(0,"numop")])
        compOp_b.addHamTerm(2., [(1,"qhoPos")])

        # Result
        res = compOp*compOp_b

        # Gold
        goldOp = compositeOperator()
        goldOp.addHamTerm( -2, [(0,"numop"),] )
        goldOp.addHamTerm( -4, [(1,"qhoPos"),] )
        goldOp.addHamTerm( 3 , [(0,"qhoCr"),(1,"qhoAn"),(0,"numop")] )
        goldOp.addHamTerm( 6 , [(0,"qhoCr"),(1,"qhoAn"),(1,"qhoPos")] )

        # print(goldOp.hamTerms)

        # Assert equal
        s.assertEqual( goldOp, res )

        # Reverse ordering of multiplication
        reverseGoldOp = compositeOperator()
        reverseGoldOp.addHamTerm( -2, [(0,"numop"),] )
        reverseGoldOp.addHamTerm( -4, [(1,"qhoPos"),] )
        reverseGoldOp.addHamTerm( 3 , [(0,"numop"),(0,"qhoCr"),(1,"qhoAn"),] )
        reverseGoldOp.addHamTerm( 6 , [(1,"qhoPos"),(0,"qhoCr"),(1,"qhoAn"),] )


        # Assert not equal for the reverse-multiplied case.
        s.assertNotEqual( reverseGoldOp, res )
        # s.assertNotEqual( goldOp, res)  <-- this one fails. Great.

        # Assert equal though if you reverse the multiplication
        s.assertEqual( reverseGoldOp, compOp_b*compOp)



if __name__ == '__main__':
    unittest.main()




cmt='''
    def test_conversion_from_of(s):


        # composopA0 = ofOpToCompositeOp(s,inpBosonOp,dVals,encodings)


        # ***************
        # Cases for one-ss operator
        # ***************
        # (A) One subsystem, one operator
        bosA0 = BosonOperator('0')  # a0

        composopA0 = ofOpToCompositeOp(bosA0,3,'unary')
        # First make sure obj's system props are correct
        s.assertEqual(composopA0.subsystems[0].enc,'unary')
        # Make sure obj's hamTerms are correct
        s.assertEqual(composopA0.subsystems[0].d,3)

        gold = np.sqrt(1)*sglQubOp(1,0,  0)*sglQubOp(0,1,  1) + np.sqrt(2.)*sglQubOp(1,0,  1)*sglQubOp(0,1,  2)
        res = composopA0.opToPauli()
        s.assertEqual(res,gold)

        # YOU DONT NEED TO TEST THE PAULI OPS. OK, MAYBE JUST ONCE
        # WELL, THE SHIFT TOO


        # (B) 3 subsys, operator only one 3rd
        bosA2 = BosonOperator('2')  # a2
        composopA2 = ofOpToCompositeOp(bosA2,3,'unary')
        s.assertEqual(composopA2.subsystems[0].enc,'unary')
        s.assertEqual(composopA2.subsystems[0].enc,'unary')
        s.assertEqual(composopA2.subsystems[0].enc,'unary')
        # Make sure obj's hamTerms are correct
        s.assertEqual(composopA2.subsystems[0].d,3)
        s.assertEqual(composopA2.subsystems[1].d,3)
        s.assertEqual(composopA2.subsystems[2].d,3)

        shift = 2*3
        gold = np.sqrt(1)*sglQubOp(1,0,  0+shift)*sglQubOp(0,1,  1+shift) + \
                np.sqrt(2.)*sglQubOp(1,0,  1+shift)*sglQubOp(0,1,  2+shift)
        res = composopA2.opToPauli()
        s.assertEqual(res,gold)

        # (C) 3 subsys, operator only on 3rd, diff encodings for each
        bosA2 = BosonOperator('2')  # a2
        composopA2 = ofOpToCompositeOp(bosA2,[2,3,3],['stdbinary','gray','unary']) #[1 bit,2 bits, 3 bits]
        s.assertEqual(composopA2.subsystems[0].enc,'stdbinary')
        s.assertEqual(composopA2.subsystems[1].enc,'gray')
        s.assertEqual(composopA2.subsystems[2].enc,'unary')
        # Make sure obj's hamTerms are correct
        s.assertEqual(composopA2.subsystems[0].d,2)
        s.assertEqual(composopA2.subsystems[1].d,3)
        s.assertEqual(composopA2.subsystems[2].d,3)

        shift = 1 + 2
        gold = np.sqrt(1)*sglQubOp(1,0,  0+shift)*sglQubOp(0,1,  1+shift) + \
                np.sqrt(2.)*sglQubOp(1,0,  1+shift)*sglQubOp(0,1,  2+shift)
        res = composopA2.opToPauli()
        s.assertEqual(res,gold)



        # **************************
        # Now for a composite system with multiple operators,
        # and this time using QuadOperator()
        # **************************
        # Quad Examples: q0q1 + 0.5*q1p2. All stdbin.
        # -->   X0 X1 + 0.5 * X1 Y2
        quadEx = QuadOperator('1[q0 q1] + 0.5[q1 p2]')
        qubEx  = (1/np.sqrt(2))**2 * QubitOperator('1[X0 X1] + 0.5[X1 Y2]') # Same operator, assuming 2-level systems
        # Above is 1/sqrt(2)^2=2, because q=1/sqrt(2)*X

        composOpQuad = ofOpToCompositeOp(quadEx,[2,2,2],['gray','gray','gray'])
        res = composOpQuad.opToPauli()
        s.assertEqual(qubEx,res)

'''
































