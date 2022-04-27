
import unittest


import integer2bit as i2b
import numpy as np

# from openfermion import QubitOperator






class i2b_tests(unittest.TestCase):

        def setup(self):
            pass


        def test_num_mappings(s):

            # Standard binary
            sb0 = i2b.dec2stdbin(0,7)
            sb1 = i2b.dec2stdbin(1,7)
            sb7 = i2b.dec2stdbin(7,7)
            sb10 = i2b.dec2stdbin(10,10)
            s.assertEqual(sb0,  [0,0,0])
            s.assertEqual(sb1,  [1,0,0])
            s.assertEqual(sb7,  [1,1,1])
            s.assertEqual(sb10, [0,1,0,1])

            # Unary
            wi0 = i2b.dec2unary(0, 2)
            wi1 = i2b.dec2unary(1, 2)
            wi1_lmax4 = i2b.dec2unary(1, 4)
            wi5 = i2b.dec2unary(5, 5)
            s.assertEqual(wi0, [1,0,0])
            s.assertEqual(wi1, [0,1,0])
            s.assertEqual(wi1_lmax4, [0,1,0,0,0])
            s.assertEqual(wi5, [0,0,0,0,0,1])


            # Gray 
            s.assertEqual( i2b.dec2gray(0,7) , [0, 0, 0] )
            s.assertEqual( i2b.dec2gray(1,7) , [1, 0, 0] )
            s.assertEqual( i2b.dec2gray(2,7) , [1, 1, 0] )
            s.assertEqual( i2b.dec2gray(3,7) , [0, 1, 0] )
            s.assertEqual( i2b.dec2gray(4,7) , [0, 1, 1] )
            s.assertEqual( i2b.dec2gray(5,7) , [1, 1, 1] )
            s.assertEqual( i2b.dec2gray(6,7) , [1, 0, 1] )
            s.assertEqual( i2b.dec2gray(7,7) , [0, 0, 1] )
            s.assertEqual( i2b.dec2gray(15,32), [0,0,0,1,0,0])




            # Block unary - stdbin
            g=3
            locEncFunc = i2b.dec2stdbin
            lmax = 11
            s.assertEqual( i2b.dec2blockunary(0,lmax,g,locEncFunc), [1, 0, 0, 0, 0, 0, 0, 0] )
            s.assertEqual( i2b.dec2blockunary(1,lmax,g,locEncFunc), [0, 1, 0, 0, 0, 0, 0, 0] )
            s.assertEqual( i2b.dec2blockunary(2,lmax,g,locEncFunc), [1, 1, 0, 0, 0, 0, 0, 0] )
            s.assertEqual( i2b.dec2blockunary(3,lmax,g,locEncFunc), [0, 0, 1, 0, 0, 0, 0, 0] )
            s.assertEqual( i2b.dec2blockunary(4,lmax,g,locEncFunc), [0, 0, 0, 1, 0, 0, 0, 0] )
            s.assertEqual( i2b.dec2blockunary(5,lmax,g,locEncFunc), [0, 0, 1, 1, 0, 0, 0, 0] )
            s.assertEqual( i2b.dec2blockunary(6,lmax,g,locEncFunc), [0, 0, 0, 0, 1, 0, 0, 0] )
            s.assertEqual( i2b.dec2blockunary(7,lmax,g,locEncFunc), [0, 0, 0, 0, 0, 1, 0, 0] )
            s.assertEqual( i2b.dec2blockunary(8,lmax,g,locEncFunc), [0, 0, 0, 0, 1, 1, 0, 0] )
            s.assertEqual( i2b.dec2blockunary(9,lmax,g,locEncFunc), [0, 0, 0, 0, 0, 0, 1, 0] )
            s.assertEqual( i2b.dec2blockunary(10,lmax,g,locEncFunc), [0, 0, 0, 0, 0, 0, 0, 1] )
            s.assertEqual( i2b.dec2blockunary(11,lmax,g,locEncFunc), [0, 0, 0, 0, 0, 0, 1, 1] )


            g=3
            locEncFunc = i2b.dec2gray
            lmax = 11
            s.assertEqual( i2b.dec2blockunary(0,lmax,g,locEncFunc), [1, 0, 0, 0, 0, 0, 0, 0] )
            s.assertEqual( i2b.dec2blockunary(1,lmax,g,locEncFunc), [1, 1, 0, 0, 0, 0, 0, 0] )
            s.assertEqual( i2b.dec2blockunary(2,lmax,g,locEncFunc), [0, 1, 0, 0, 0, 0, 0, 0] )
            s.assertEqual( i2b.dec2blockunary(3,lmax,g,locEncFunc), [0, 0, 1, 0, 0, 0, 0, 0] )
            s.assertEqual( i2b.dec2blockunary(4,lmax,g,locEncFunc), [0, 0, 1, 1, 0, 0, 0, 0] )
            s.assertEqual( i2b.dec2blockunary(5,lmax,g,locEncFunc), [0, 0, 0, 1, 0, 0, 0, 0] )
            s.assertEqual( i2b.dec2blockunary(6,lmax,g,locEncFunc), [0, 0, 0, 0, 1, 0, 0, 0] )
            s.assertEqual( i2b.dec2blockunary(7,lmax,g,locEncFunc), [0, 0, 0, 0, 1, 1, 0, 0] )
            s.assertEqual( i2b.dec2blockunary(8,lmax,g,locEncFunc), [0, 0, 0, 0, 0, 1, 0, 0] )
            s.assertEqual( i2b.dec2blockunary(9,lmax,g,locEncFunc),  [0, 0, 0, 0, 0, 0, 1, 0] )
            s.assertEqual( i2b.dec2blockunary(10,lmax,g,locEncFunc), [0, 0, 0, 0, 0, 0, 1, 1] )
            s.assertEqual( i2b.dec2blockunary(11,lmax,g,locEncFunc), [0, 0, 0, 0, 0, 0, 0, 1] )


            g=5
            locEncFunc = i2b.dec2gray
            lmax = 11
            s.assertEqual( i2b.dec2blockunary(0,lmax,g,locEncFunc), [1, 0, 0, 0, 0, 0, 0, 0, 0] )
            s.assertEqual( i2b.dec2blockunary(1,lmax,g,locEncFunc), [1, 1, 0, 0, 0, 0, 0, 0, 0] )
            s.assertEqual( i2b.dec2blockunary(2,lmax,g,locEncFunc), [0, 1, 0, 0, 0, 0, 0, 0, 0] )
            s.assertEqual( i2b.dec2blockunary(3,lmax,g,locEncFunc), [0, 1, 1, 0, 0, 0, 0, 0, 0] )
            s.assertEqual( i2b.dec2blockunary(4,lmax,g,locEncFunc), [1, 1, 1, 0, 0, 0, 0, 0, 0] )
            s.assertEqual( i2b.dec2blockunary(5,lmax,g,locEncFunc), [0, 0, 0, 1, 0, 0, 0, 0, 0] )
            s.assertEqual( i2b.dec2blockunary(6,lmax,g,locEncFunc), [0, 0, 0, 1, 1, 0, 0, 0, 0] )
            s.assertEqual( i2b.dec2blockunary(7,lmax,g,locEncFunc), [0, 0, 0, 0, 1, 0, 0, 0, 0] )
            s.assertEqual( i2b.dec2blockunary(8,lmax,g,locEncFunc), [0, 0, 0, 0, 1, 1, 0, 0, 0] )
            s.assertEqual( i2b.dec2blockunary(9,lmax,g,locEncFunc),  [0, 0, 0, 1, 1, 1, 0, 0, 0] )
            s.assertEqual( i2b.dec2blockunary(10,lmax,g,locEncFunc), [0, 0, 0, 0, 0, 0, 1, 0, 0] )
            s.assertEqual( i2b.dec2blockunary(11,lmax,g,locEncFunc), [0, 0, 0, 0, 0, 0, 1, 1, 0] )


            g=7
            locEncFunc = i2b.dec2gray
            lmax = 11
            s.assertEqual( i2b.dec2blockunary(0,lmax,g,locEncFunc), [1, 0, 0, 0, 0, 0] )
            s.assertEqual( i2b.dec2blockunary(1,lmax,g,locEncFunc), [1, 1, 0, 0, 0, 0] )
            s.assertEqual( i2b.dec2blockunary(2,lmax,g,locEncFunc), [0, 1, 0, 0, 0, 0] )
            s.assertEqual( i2b.dec2blockunary(3,lmax,g,locEncFunc), [0, 1, 1, 0, 0, 0] )
            s.assertEqual( i2b.dec2blockunary(4,lmax,g,locEncFunc), [1, 1, 1, 0, 0, 0] )
            s.assertEqual( i2b.dec2blockunary(5,lmax,g,locEncFunc), [1, 0, 1, 0, 0, 0] )
            s.assertEqual( i2b.dec2blockunary(6,lmax,g,locEncFunc), [0, 0, 1, 0, 0, 0] )
            s.assertEqual( i2b.dec2blockunary(7,lmax,g,locEncFunc), [0, 0, 0, 1, 0, 0] )
            s.assertEqual( i2b.dec2blockunary(8,lmax,g,locEncFunc), [0, 0, 0, 1, 1, 0] )
            s.assertEqual( i2b.dec2blockunary(9,lmax,g,locEncFunc), [0, 0, 0, 0, 1, 0] )
            s.assertEqual( i2b.dec2blockunary(10,lmax,g,locEncFunc), [0, 0, 0, 0, 1, 1] )
            s.assertEqual( i2b.dec2blockunary(11,lmax,g,locEncFunc), [0, 0, 0, 1, 1, 1] )
            

            # Ensure lmax doesn't mess things up, in block unary
            locEncFunc = i2b.dec2gray
            g=3
            s.assertEqual( i2b.dec2blockunary(0,0, g,locEncFunc), [1,0] )  # lmax=inp for all these
            s.assertEqual( i2b.dec2blockunary(1,1, g,locEncFunc), [1,1] )  # lmax=inp for all these
            s.assertEqual( i2b.dec2blockunary(2,2, g,locEncFunc), [0,1] )  # lmax=inp for all these
            s.assertEqual( i2b.dec2blockunary(3,3, g,locEncFunc), [0,0, 1,0] )  # lmax=inp for all these
            s.assertEqual( i2b.dec2blockunary(4,4, g,locEncFunc), [0,0, 1,1] )  # lmax=inp for all these
            s.assertEqual( i2b.dec2blockunary(5,5, g,locEncFunc), [0,0, 0,1] )  # lmax=inp for all these
            s.assertEqual( i2b.dec2blockunary(6,6, g,locEncFunc), [0,0, 0,0, 1,0] )  # lmax=inp for all these
            
            # Same, but testing bit mask now.
            locEncFunc = i2b.dec2gray
            g=3
            s.assertEqual( i2b.getBitMask(0,0, "blockunary",{'g':g,'localEncodingFunc':locEncFunc}), [1,1] )  # lmax=inp for all these
            s.assertEqual( i2b.getBitMask(1,1, "blockunary",{'g':g,'localEncodingFunc':locEncFunc}), [1,1] )  
            s.assertEqual( i2b.getBitMask(2,2, "blockunary",{'g':g,'localEncodingFunc':locEncFunc}), [1,1] )  
            s.assertEqual( i2b.getBitMask(3,3, "blockunary",{'g':g,'localEncodingFunc':locEncFunc}), [0,0, 1,1] ) 
            s.assertEqual( i2b.getBitMask(4,4, "blockunary",{'g':g,'localEncodingFunc':locEncFunc}), [0,0, 1,1] )  
            s.assertEqual( i2b.getBitMask(5,5, "blockunary",{'g':g,'localEncodingFunc':locEncFunc}), [0,0, 1,1] )  
            s.assertEqual( i2b.getBitMask(6,6, "blockunary",{'g':g,'localEncodingFunc':locEncFunc}), [0,0, 0,0, 1,1] )  
            




        def test_BitCounts(s):

            s.assertEqual( i2b.getBitCount(11,"stdbinary") , 4 )
            s.assertEqual( i2b.getBitCount(11,"gray")      , 4 )
            s.assertEqual( i2b.getBitCount(11,"unary")     , 12 )
            s.assertEqual( i2b.getBitCount(11,"blockunary",{'g':3}) , 8 )
            s.assertEqual( i2b.getBitCount(11,"blockunary",{'g':5}) , 9 )
            s.assertEqual( i2b.getBitCount(11,"blockunary",{'g':7}) , 6 )
            
            s.assertEqual( i2b.getBitCount(14,"stdbinary") , 4 )
            s.assertEqual( i2b.getBitCount(15,"stdbinary") , 4 )
            s.assertEqual( i2b.getBitCount(16,"stdbinary") , 5 )

        def test_MaxDFromNumBits(s):
            
            s.assertEqual( i2b.getMaxDFromNumBits(2,"gray"), 4 )
            s.assertEqual( i2b.getMaxDFromNumBits(2,"stdbinary"), 4 )
            s.assertEqual( i2b.getMaxDFromNumBits(2,"unary"), 2 )
            s.assertEqual( i2b.getMaxDFromNumBits(2,"blockunary",{'g':3}), 3 )
            s.assertEqual( i2b.getMaxDFromNumBits(2,"blockunary",{'g':5}), 3 )
            s.assertEqual( i2b.getMaxDFromNumBits(2,"blockunary",{'g':7}), 3 )
            
            s.assertEqual( i2b.getMaxDFromNumBits(3,"gray"), 8 )
            s.assertEqual( i2b.getMaxDFromNumBits(3,"stdbinary"), 8 )
            s.assertEqual( i2b.getMaxDFromNumBits(3,"unary"), 3 )
            s.assertEqual( i2b.getMaxDFromNumBits(3,"blockunary",{'g':3}), 4 )
            s.assertEqual( i2b.getMaxDFromNumBits(3,"blockunary",{'g':5}), 5 )
            s.assertEqual( i2b.getMaxDFromNumBits(3,"blockunary",{'g':7}), 7 )
            
            s.assertEqual( i2b.getMaxDFromNumBits(4,"gray"), 16 )
            s.assertEqual( i2b.getMaxDFromNumBits(4,"stdbinary"), 16 )
            s.assertEqual( i2b.getMaxDFromNumBits(4,"unary"), 4 )
            s.assertEqual( i2b.getMaxDFromNumBits(4,"blockunary",{'g':3}), 6 )
            s.assertEqual( i2b.getMaxDFromNumBits(4,"blockunary",{'g':5}), 6 )
            s.assertEqual( i2b.getMaxDFromNumBits(4,"blockunary",{'g':7}), 8 )
            
            s.assertEqual( i2b.getMaxDFromNumBits(5,"blockunary",{'g':3}), 7 )
            s.assertEqual( i2b.getMaxDFromNumBits(5,"blockunary",{'g':5}), 8 )
            s.assertEqual( i2b.getMaxDFromNumBits(5,"blockunary",{'g':7}), 10 )









        def test_generalfunc(s):

            # Here we test the int2bits() function,
            # which takes in a string that indicates the encoding, along with parameters

            # gray
            s.assertEqual( i2b.int2bits(3,7,enc="gray") , [0,1,0] )

            # stdbin
            s.assertEqual( i2b.int2bits(3,7,enc="stdbinary") , [1,1,0] )

            # unary
            g=3
            lmax = 11
            # s.assertEqual( i2b.dec2blockunary(7,lmax,g,locEncFunc), [0, 0, 0, 0, 1, 1, 0, 0] )
            s.assertEqual( i2b.int2bits( 7,lmax, "blockunary", {'g':g,'localEncodingFunc':i2b.dec2gray} ),
                [0, 0, 0, 0, 1, 1, 0, 0] )  # explicit local function
            s.assertEqual( i2b.int2bits( 7,lmax, "blockunary", {'g':g,'localEncodingFunc':"gray"} ),
                [0, 0, 0, 0, 1, 1, 0, 0] ) # string for local function


        def test_bitmasks(s):

            # For the dense mappings, bit mask should be all-true
            s.assertEqual( i2b.getBitMask(3,4,"stdbinary") , [True,True,True] )
            s.assertEqual( i2b.getBitMask(3,4,"gray")      , [True,True,True] )

            # For unary, only the 'hot' bit should be true
            s.assertEqual( i2b.getBitMask(3,7,"unary") , [False,False,False,True,False,False,False,False] )

            # For block unary, it's only the block on which the current number sits
            s.assertEqual( i2b.getBitMask(9, 11,"blockunary", {'g':5}), [0,0,0,1,1,1,0,0,0] )
            s.assertEqual( i2b.getBitMask(10,11,"blockunary", {'g':5}), [0,0,0,0,0,0,1,1,1] )
            s.assertEqual( i2b.getBitMask(6, 11,"blockunary", {'g':7}), [1,1,1,0,0,0] )
            s.assertEqual( i2b.getBitMask(7, 11,"blockunary", {'g':7}), [0,0,0,1,1,1] )




        def test_bu_string(s):

            # i2b.int2bits(inp,lmax,enc)

            # s.assertEqual( i2b.dec2blockunary(6,lmax,7,"gray"), [0, 0, 1, 0, 0, 0] )
            # s.assertEqual( i2b.dec2blockunary(7,lmax,7,"gray"), [0, 0, 0, 1, 0, 0] )
            # s.assertEqual( i2b.getBitMask(9, 11,"blockunary", {'g':5}), [0,0,0,1,1,1,0,0,0] )
            # s.assertEqual( i2b.getBitMask(10,11,"blockunary", {'g':5}), [0,0,0,0,0,0,1,1,1] )


            s.assertEqual( i2b.int2bits(6,11,"bu_gray_7"), [0, 0, 1, 0, 0, 0] )
            s.assertEqual( i2b.int2bits(7,11,"bu_gray_7"), [0, 0, 0, 1, 0, 0] )
            s.assertEqual( i2b.getBitMask(9, 11,"bu_gray_5"), [0,0,0,1,1,1,0,0,0] )
            s.assertEqual( i2b.getBitMask(10,11,"bu_gray_5"), [0,0,0,0,0,0,1,1,1] )




        # def test_2x2mat_sglbos2qub(s):
        #     # Single boson matrix-operator, onto qubit

        #     # Original matrix represenation
        #     matop = np.array([[1, 2+1j],[2-1j, 4.]])

        #     # Standard binary
        #     gold = 1* qhos_symb.sglQubOp(0,0, 0) \
        #             + (2+1j)* qhos_symb.sglQubOp(0,1, 0) \
        #             + (2-1j)* qhos_symb.sglQubOp(1,0, 0) \
        #             + 4* qhos_symb.sglQubOp(1,1, 0)
        #     res  = qhos_symb.sglbosop2qubop(matop,'stdbin')
        #     s.assertEqual(gold,res)


        #     # Unary
        #     gold = 1* qhos_symb.sglQubOp(1,1, 0) \
        #             + (2+1j)* qhos_symb.sglQubOp(1,0, 0)*qhos_symb.sglQubOp(0,1, 1) \
        #             + (2-1j)* qhos_symb.sglQubOp(0,1, 0)*qhos_symb.sglQubOp(1,0, 1) \
        #             + 4* qhos_symb.sglQubOp(1,1, 1)
        #     res  = qhos_symb.sglbosop2qubop(matop,'unary')
            
        #     s.assertEqual(gold,res)






if __name__ == '__main__':
    unittest.main()



























