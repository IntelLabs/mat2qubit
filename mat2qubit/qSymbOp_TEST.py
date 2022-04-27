


import unittest

import numpy as np

import qSymbOp




'''
PURPOSE of this code is to be able to have *both* coefficients and subspaces
be symbolic. For now, putting this on the backburner though.
'''


class qSymbOp_test(unittest.TestCase):


    def setUp(s):
        s.globalop1 = qSymbOp.qSymbOp('1 [X_1 Z_v2]')
        s.globalop2 = qSymbOp.qSymbOp('-kk [X_1 Z_v2]')

    #   How should I *access* the individual terms. Should not allow indexed access.
    #   Maker iterator (generator), than can also be used to pull it all out at once
    #   Should maybe actually have each string be like a 'struct' of terms, and each op
    #   in each term would be accessed by (thelocalop.op) and (thelocalop.ss) OR by simply 0 and 1.

    # NO ACTUALLY. DONT DO THIS. KEEP AS DICT OF TUPLE OF 2-TUPLES.
    # What you COULD do though is have functions that wrap around the dict?



    def test_str_read_0(s):

        symop1 = qSymbOp.qSymbOp('3 [Z_v2 ]')

        opstringTuple = (('v2','Z'),)

        s.assertEqual( symop1.getSsids(), {'v2'} )
        s.assertEqual( str(symop1) , '((3+0j)) [Z_v2]' )



    def test_str_read_1(s):
        # Testing single term
        symop1 = qSymbOp.qSymbOp('1 [X_1 Z_v2]')
        # A coefficient shouldn't be required

        opstringTuple = (('1','X'),('v2','Z'))

        # Subsystems stored correctly
        # s.assertEqual(  symop1.getCurrentlyUsedSsids() , ('1','v2') )
        # s.assertEqual(  symop1.getCurrentlyAllowedSsids() , ('1','v2') )
        s.assertEqual(  symop1.getSsids() , {'1','v2'} )

        # Operator is correct
        s.assertEqual( str(symop1) , '((1+0j)) [X_1 Z_v2]' )
        s.assertEqual( symop1.getCoeff( opstringTuple ) , 1. )  # Otherwise should be (numeric,'string')

        # Raises errors
        with s.assertRaises( IndexError ):
            symop1.getCoeff( (('2','X'),) )   # Since subsys '2' doesn't exist
        with s.assertRaises( IndexError ):
            symop1.getCoeff( (('1','Y'),) )   # Since operator 'Y' isn't one of allowed ops

        # # getOpstringsContainingLocalOp(): Get opstrings containing this one
        # # Will return its own qSymbOp object, holding a subset of the full operator
        # s.assertEqual( str(symop.getOpstringsContainingLocalOp(('v2','Z'))) , '[X_1 Z_v2]' )
        # This funcion seemed pointless

    def test_str_read_1_symbCoeff(s):
        symop1 = qSymbOp.qSymbOp('-kk [X_1 Z_v2]')
        stringTuple = (('1','X'),('v2','Z'),)

        # s.assertEqual( symop1.getCoeff(stringTuple) , (-1.,'kk') )
        s.assertEqual( symop1.getCoeff(stringTuple) , qSymbOp.symbScalarFromStr('-kk') )


        # Test the __str__ func
        s.assertEqual( str(symop1) , '(-kk) [X_1 Z_v2]' )



    def test_str_read_2a(s):
        symop2 = qSymbOp.qSymbOp('[X_1 Z_v2] ++ 2*k [Y_1 Z_v3]')

        opstringTuple1 = ( ('1','X'), ('v2','Z'), )
        opstringTuple2 = ( ('1','Y'), ('v3','Z'), )

        # Coeffs
        s.assertEqual( symop2.getCoeff( opstringTuple1 ) , qSymbOp.symbScalarFromStr('1') )
        s.assertEqual( symop2.getCoeff( opstringTuple2 ) , qSymbOp.symbScalarFromStr('2*k') )


    def test_identity(s):
                
        symop = qSymbOp.qSymbOp('7.0 [ ]')
        s.assertEqual(str(symop),"((7+0j)) []")
                
        symop = qSymbOp.qSymbOp('7 [] ++ [X_1 Z_v2]')
        s.assertEqual( str(symop) , "((7+0j)) []\n++ ((1+0j)) [X_1 Z_v2]")




    # def test_str_read_2b(s):
    #   pass

    def test_str_read_3(s):
        symop2 = qSymbOp.qSymbOp('[X_1 Z_v2] ++ 2*k [Y_1 Z_v3] ++ - y [Z_1 Z_v3 bigop_v4]')
        opstringTuple1 = ( ('1','X'), ('v2','Z'), )
        opstringTuple2 = ( ('1','Y'), ('v3','Z'), )
        # Third one in different order to ensure that code is sorting the tuples
        # [OH CRAP. YOU DON'T WANT TO SORT THE INDIVIDUAL TUPLES.]
        opstringTuple3 = ( ('1','Z'), ('v3','Z'), ('v4','bigop'), ) 
        opStringTuple3_wrongorder = ( ('v4','bigop'), ('1','Z'), ('v3','Z'), ) 

        s.assertEqual( symop2.getCoeff( opstringTuple1 ) , qSymbOp.symbScalarFromStr('1') )
        s.assertEqual( symop2.getCoeff( opstringTuple2 ) , qSymbOp.symbScalarFromStr('2*k') )
        s.assertEqual( symop2.getCoeff( opstringTuple3 ) , qSymbOp.symbScalarFromStr('-y') )
        with s.assertRaises( IndexError ):
            symop2.getCoeff( opStringTuple3_wrongorder )


    # Test subsituting in variables
    def test_subs(s):
        symop2 = qSymbOp.qSymbOp('[X_1 Z_v2] ++ 2*k [Y_1 Z_v3] ++ - y [Z_1 Z_v3 bigop_v4]')

        symop2.scalar_subs({'k':2.,'y':7.})
        # Note that if varname doesn't exist, nothing happens, no warning or error.

        opstringTuple1 = ( ('1','X'), ('v2','Z'), )
        opstringTuple2 = ( ('1','Y'), ('v3','Z'), )
        opstringTuple3 = ( ('1','Z'), ('v3','Z'), ('v4','bigop'), ) 

        s.assertEqual( symop2.getCoeff( opstringTuple1 ) , 1 )
        s.assertEqual( symop2.getCoeff( opstringTuple2 ) , 2*2 )
        s.assertEqual( symop2.getCoeff( opstringTuple3 ) , -7 )

    def test_is_symbolic(s):

        # opTuple = (('1','X'),)

        symop = qSymbOp.qSymbOp('2 [X_1]')
        s.assertFalse( qSymbOp.is_symbolic( symop.getCoeff('X_1') ) )
        # print(symop)

        symop = qSymbOp.qSymbOp('2.0 [X_1]')
        s.assertFalse( qSymbOp.is_symbolic( symop.getCoeff('X_1') ) )
        # print(symop)

        symop = qSymbOp.qSymbOp('1 + sqrt(-1) [X_1]')
        s.assertFalse( qSymbOp.is_symbolic( symop.getCoeff('X_1') ) )
        # print(symop)

        symop = qSymbOp.qSymbOp('2*I [X_1]')
        s.assertFalse( qSymbOp.is_symbolic( symop.getCoeff('X_1') ) )
        # print(symop)

        symop = qSymbOp.qSymbOp('-kk [X_1]')
        s.assertTrue( qSymbOp.is_symbolic( symop.getCoeff('X_1') ) )
        # print(symop)


    def test_neg_1(s):
        symop1 = qSymbOp.qSymbOp('7.0 [X_1 Z_v2]')
        symop1 = -symop1
        s.assertEqual(str(symop1),'((-7+0j)) [X_1 Z_v2]')

    def test_neg_2(s):
        symop1 = qSymbOp.qSymbOp('(-kk) [X_1 Z_v2]')
        symop1 = - symop1
        s.assertEqual(str(symop1),'(kk) [X_1 Z_v2]')




    def test_add(s):
        sumop = s.globalop1 + (- s.globalop2)
        s.assertEqual( str(sumop) , '(kk + 1) [X_1 Z_v2]' )
        s.assertEqual( sumop.ssid_set, {'v2','1'})
                
        # symop1 = qSymbOp.qSymbOp('[X_1 Z_v2]')
        # symop2 = qSymbOp.qSymbOp('2*k [Y_1 Z_v3]')

    # def test_iadd(s):
    #   symop1 = qSymbOp.qSymbOp('[X_1 Z_v2]')
    #   symop2 = qSymbOp.qSymbOp('-2*k [Y_1 Z_v3]')
    #   symop1 += symop2
    #   s.assertEqual( str(symop1) , '[X_1 Z_v2] -2*k [Y_1 Z_v3]' )


    def test_mult_1_by_scalar_(s):
        # mult by numeric scalar
        symop1 = qSymbOp.qSymbOp('-kk [X_1 Z_v2]')
        symop1 = 7.0 * symop1
        s.assertEqual( str(symop1) , '(-7.0*kk) [X_1 Z_v2]' )
        # mult by numeric scalar
        symop1 = qSymbOp.qSymbOp('-kk [X_1 Z_v2]')
        symop1 *= 7
        s.assertEqual( str(symop1) , '(-7*kk) [X_1 Z_v2]' )
        # mult by numeric scalar
        symop1 = qSymbOp.qSymbOp('-kk [X_1 Z_v2]')
        symop1 *= 2+2j
        s.assertEqual( str(symop1) , '(-kk*(2.0 + 2.0*I)) [X_1 Z_v2]' )
        # mult by symbolic scalar
        symop1 = qSymbOp.qSymbOp('-kk [X_1 Z_v2]')
        symop1 *= qSymbOp.symbScalarFromStr('p')
        s.assertEqual( str(symop1) , '(-kk*p) [X_1 Z_v2]' )

        # REALLY SHOULD BE DOING EQUALITIES DIFFERENTLY....
        # I GUESS YOU HAVEN'T WRITTEN AN EQUALITY FUNCTION YET....

        


    # def test_mult_2_by_scalar(s):
    #   # mult by numeric scalar
    #   symop2 = qSymbOp.qSymbOp('[X_1 Z_v2] + 2*k [Y_1 Z_v3]')
    #   symop2 *= 7.2
    #   s.assertEqual( str(symop1) , '7.2 [X_1 Z_v2] + 14.4*k [Y_1 Z_v3]' )
    #   # mult by symbolic scalar
    #   symop2 = qSymbOp.qSymbOp('[X_1 Z_v2] + 2*k [Y_1 Z_v3]')
    #   symop1 *= symbScalarFromStr('p')
    #   s.assertEqual( str(symop1) , 'p [X_1 Z_v2] + k*p [Y_1 Z_v3]' )

    def test_str_mult_1_1(s):
        # mult one-term by 1-term
        symop1 = qSymbOp.qSymbOp('[X_1 Z_v2]')
        symop2 = qSymbOp.qSymbOp('2*k [Y_1 Z_v3]')
        prodop = symop1*symop2
        s.assertEqual( str(prodop) , '(2*k) [X_1 Z_v2 Y_1 Z_v3]' )
        # Remember, the ORDER MATTERS *****

        # Identity
        symop1 = qSymbOp.qSymbOp('3 []')
        symop2 = qSymbOp.qSymbOp('2*k [Y_1 Z_v3]')
        prodop = symop1*symop2
        s.assertEqual( str(prodop) , '(6*k) [Y_1 Z_v3]' )



    # def test_str_mult_1_2(s):
    #   # mult one-term by 2-term
    #   symop1 = qSymbOp.qSymbOp('k [X_1 Z_v2]')
    #   symop2 = qSymbOp.qSymbOp('[Y_1 Z_v3] ++ 2.1 [Z_0 ad_v2]')
    #   prodop = symop1*symop2
        
    #   s.assertEqual( str(prodop) , '(k) [X_1 Z_v2 Y_1 Z_v3]\n++ (2.1*k) [X_1 Z_v2 Z_0 ad_v2]' )

    #   # Identity case
    #   symop1 = qSymbOp.qSymbOp('3 []')
    #   symop2 = qSymbOp.qSymbOp('2 [] ++ 2*k [Y_1 Z_v3]')
    #   prodop = symop1*symop2
    #   s.assertEqual( str(prodop) , '6 []\n++ (6*k) [Y_1 Z_v3]' )


    def test_str_mult_2_2(s):
        # mult two-term by 2-term
        symop1 = qSymbOp.qSymbOp('[X_0 Y_1] ++ 2 [Z_2 W_4]')
        symop2 = qSymbOp.qSymbOp('a [X_5 Y_6] ++ -b [Z_7 W_8]')
        prodop = symop1*symop2
        # print()
        # print(prodop)
        # print()
        # print()
        s.assertEqual( str(prodop) , '(a) [X_0 Y_1 X_5 Y_6]\n++ (-b) [X_0 Y_1 Z_7 W_8]\n++ (2*a) [Z_2 W_4 X_5 Y_6]\n++ (-2*b) [Z_2 W_4 Z_7 W_8]' )


        # Multiplying by identity
        identK = qSymbOp.qSymbOp("")
        #(incomplete)
        

        # Mult by the same operator
        # symop = qSymbOp.qSymbOp('[X_0] ++ [Z_2 W_4]')
        # res = symop*symop
        # print("symop**2")
        # print(res)
        #s.assertEqual( str(res), 

    #   # mult 2-term by same 2-term
    #   prodop = symop1*symop1
    #   s.assertEqual( str(prodop) , '[X_0 Y_1 X_0 Y_1] + 4 [X_0 Y_1 Z_2 W_4] + 4 [Z_2 W_4 Z_2 W_4]' )

    #   # test re-grouping when terms are the same


    def test_orderTerms(s):
        
        pass
        
        
    def test_simplifyQuadExponents(s):

        symop0   = qSymbOp.qSymbOp("a [  ]") 
        # --> [ ]

        symop1   = qSymbOp.qSymbOp("a [ q2_1 q_0 q_1 q_0 ]") 
        # --> [q2_0 q3_1]
        
        symop2   = qSymbOp.qSymbOp("b [ qhoPos_0 qhoMom_1 Qsq_0 Psq_2 q3_0 p_1 ]")
        # --> [ q6_0 p2_1 p2_2 ]
        
        symop3   = qSymbOp.qSymbOp("c [ q2_1 q_0 Sx_1 q_1 q_0 TT_0 ]")
        # --> [ q2_0 TT_0 q2_1 Sx_1 q_1 ]   # Sx and q do not necessarily commute

        symop4   = qSymbOp.qSymbOp("d [ q_0 p_0 q_1 q_1 p_0 p_1 q_1 ]")
        # --> [ q_0 p2_0 q2_1 p_1 q_1 ]

        symop123 = symop1 + symop2 + symop3
        # --> a [q3_1 q2_0] + b [ q6_0 p2_1 p2_2 ] + c [ q2_0 TT_0 q3_1 Sx_1 ]



        symop0.simplifyQuadExponents()
        res0 = str(symop0)
        gold0 = "(a) []"
        s.assertEqual(res0,gold0)

        symop1.simplifyQuadExponents()
        res1  = str(symop1)
        gold1 = "(a) [q2_0 q3_1]"
        s.assertEqual(res1,gold1)

        symop2.simplifyQuadExponents()
        res2  = str(symop2)
        gold2 = "(b) [q6_0 p2_1 p2_2]"
        s.assertEqual(res2,gold2)

        symop3.simplifyQuadExponents()
        res3  = str(symop3)
        gold3 = "(c) [q2_0 TT_0 q2_1 Sx_1 q_1]"
        s.assertEqual(res3,gold3)

        symop4.simplifyQuadExponents()
        res4  = str(symop4)
        gold4 = "(d) [q_0 p2_0 q2_1 p_1 q_1]"
        s.assertEqual(res4,gold4)

        symop123.simplifyQuadExponents()
        res123  = str(symop123)
        gold123 = "(a) [q2_0 q3_1]\n++ (b) [q6_0 p2_1 p2_2]\n++ (c) [q2_0 TT_0 q2_1 Sx_1 q_1]"
        s.assertEqual(res123,gold123)
        


        # Make sure that it's not over-writing duplicates, e.g. q1q0+q0q1 should be 2*q0q1:
        symop = qSymbOp.qSymbOp("[q_0 q_1] ++ [q_1 q_0]")
        symop.simplifyQuadExponents()
        res = str( symop )
        gold = "((2+0j)) [q_0 q_1]"
        s.assertEqual(res,gold)
        
        
        # mixx some other ops in there
        # change up the ordering
        # mix in all types of strings
        # commutation not messed up



if __name__ == '__main__':
    unittest.main()




























































