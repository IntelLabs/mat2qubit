# qOpMats_TEST.py

import unittest

import qopmats
import numpy as np




class i2b_tests(unittest.TestCase):

    def setup(self):
        pass


    def test_spin(s):
        
        # Matrices for spin-1/2
        x1_2 = 1./2 * np.array([[ 0, 1 ],
                        [ 1, 0 ]],dtype=complex)
        y1_2 = 1./2 * np.array([[ 0, -1.j ],
                        [ 1.j, 0 ]],dtype=complex)
        z1_2 = 1./2 * np.array([[ 1, 0 ],
                        [ 0, -1 ]],dtype=complex)

        s.assertEqual( x1_2.tolist() , qopmats.spinX(0.5).tolist() )
        s.assertEqual( y1_2.tolist() , qopmats.spinY(0.5).tolist() )
        s.assertEqual( z1_2.tolist() , qopmats.spinZ(0.5).tolist() )
        

        # Matrices for spin-5/2
        x5_2 = np.zeros((6,6),dtype=complex)
        x5_2[0,1] = np.sqrt(5.)
        x5_2[1,2] = 2*np.sqrt(2)
        x5_2[2,3] = 3.
        x5_2[3,4] = 2*np.sqrt(2)
        x5_2[4,5] = np.sqrt(5.)
        x5_2 += np.transpose(x5_2)
        x5_2 *= 1./2
        
        y5_2 = np.zeros((6,6),dtype=complex)
        y5_2[0,1] = -1j*np.sqrt(5)
        y5_2[1,2] = -2j*np.sqrt(2)
        y5_2[2,3] = -3j
        y5_2[3,4] = -2j*np.sqrt(2)
        y5_2[4,5] = -1j*np.sqrt(5)
        y5_2 += np.conjugate(np.transpose(y5_2))
        y5_2 *= 1./2
        
        z5_2 = np.zeros((6,6),dtype=complex)
        z5_2[0,0] = 5
        z5_2[1,1] = 3
        z5_2[2,2] = 1
        z5_2[3,3] = -1
        z5_2[4,4] = -3
        z5_2[5,5] = -5
        z5_2 *= 1./2
        

        s.assertEqual(x5_2.tolist(), qopmats.spinX(2.5).tolist())
        s.assertEqual(y5_2.tolist(), qopmats.spinY(2.5).tolist())
        s.assertEqual(z5_2.tolist(), qopmats.spinZ(2.5).tolist())

    def test_qub(s):
        
        s.assertEqual(qopmats.qub['I2'].tolist() , np.eye(2).tolist() )
        s.assertEqual(qopmats.qub['Z'].tolist() , np.array([[1.+0j,0],[0,-1]]).tolist() )
        s.assertEqual(qopmats.qub['X'].tolist() , np.array([[0j,1.],[1.,0]]).tolist() )
        s.assertEqual(qopmats.qub['Y'].tolist() , np.array([[0,-1.j],[1.j,0]]).tolist() )
        s.assertEqual(qopmats.qub['sig+'].tolist() , np.array([[0.+0j,1],[0,0]]).tolist() )
        s.assertEqual(qopmats.qub['sig-'].tolist() , np.array([[0.+0j,0],[1,0]]).tolist() )
        s.assertEqual(qopmats.qub['n'].tolist() , np.array([[0.+0j,0],[0,1]]).tolist() )

    def test_n4(s):
        n=4

        s.assertEqual( qopmats.i(n).tolist() , np.eye(n).tolist() )







if __name__ == '__main__':
    unittest.main()




