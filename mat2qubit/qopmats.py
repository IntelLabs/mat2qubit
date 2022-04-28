# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# qOpMats.py
# Simple quantum operators


import numpy as np
from functools import reduce 



# Operators for half-spin systems
qub = dict()

qub['I2'] = np.eye(2) 
qub['Z'] = np.array([[1.+0j,0],[0,-1]]) 
qub['X'] = np.array([[0j,1.],[1.,0]]) 
qub['Y'] = np.array([[0,-1.j],[1.j,0]]) 
qub['sig+'] = np.array([[0.+0j,1],[0,0]]) 
qub['sig-'] = np.array([[0.+0j,0],[1,0]]) 
qub['n'] = np.array([[0.+0j,0],[0,1]]) 
qub['H'] = 1./np.sqrt(2) * np.array([[1+0j,1],[1,-1]])
qub['S'] = np.array([[1.,0],[0,1j]])
qub['T'] = np.array([[1.,0],[0, np.exp(1j*np.pi/4) ]])
qub['proj0'] = np.array([[1,0],[0,0]],dtype=complex)
qub['proj1'] = np.array([[0,0],[0,1]],dtype=complex)



# Single element
def singleElement(d, i,j):

    # same as Ket(i)Bra(j)

    mat = np.zeros((d,d),dtype=complex)
    mat[i,j] = 1.

    return mat


# Projectors
def proj(i ,d):
    pr = np.zeros((d,d),dtype=complex)
    pr[i,i] = 1.
    return pr


# Truncated qho ops
def i(n):
    return np.eye(n)

def cr(n):
    ad = np.zeros((n,n))
    for m in range(n-1):
        ad[m+1,m] = np.sqrt(m+1.)
    return ad

def an(n):
    return cr(n).T

def numop(n):
    numop = np.zeros((n,n))
    for i in range(n):
        numop[i,i] = float(i)
    return numop

def Nsq(n):
    nMat = numop(n)
    return np.dot(nMat,nMat)


def momQhoSq(n):
    p = momQho(n+1)
    pSq = np.dot(p,p)[:n,:n]
    return pSq

def momQho(n):
    a = an(n)
    return 1/( 1.j * np.sqrt(2.) ) * ( a - a.T )

def posQho(n):
    a = an(n)
    return 1/np.sqrt(2.) * ( a + a.T )

def posQhoSq(n):
    q = posQho(n+1)
    qSq = np.dot(q,q)[:n,:n]
    return qSq

def q2(n):
    return posQhoSq(n)

def q3(n):
    q = posQho(n+2)
    return reduce(np.dot, [q]*3)[:n,:n]

def q4(n):
    q = posQho(n+3)
    return reduce(np.dot, [q]*4)[:n,:n]

def q5(n):
    q = posQho(n+4)
    return reduce(np.dot, [q]*5)[:n,:n]

def q6(n):
    q = posQho(n+5)
    return reduce(np.dot, [q]*6)[:n,:n]






# 1st quant operator (Macridin-style)
# Interval [-L,L]
def X_1stQuant(N_x,L):

    return np.diag( np.linspace(-L,L,N_x, dtype=complex) )




# Z projection operator for a spin-s system
def spinZ(s,hbar=1.0):

    assert float(2*s).is_integer()

    N = int(2*s+1)

    Sz = np.zeros((N,N),dtype=complex)

    for a in range(1,N+1):

        Sz[a-1,a-1] = (s+1-a)

    return hbar*Sz

# X projection operator for a spin-s system
def spinX(s,hbar=1.0):

    assert float(2*s).is_integer()

    N = int(2*s+1)

    Sx = np.zeros((N,N),dtype=complex)

    # Upper diagonal
    for a in range(1,N):
        b = a+1

        Sx[a-1,b-1] = np.sqrt( (s+1)*(a+b-1) - a*b )

    # Make Hermitian
    Sx += np.transpose(np.conjugate(Sx))


    return (hbar/2) * Sx



# Y projection operator for a spin-s system
def spinY(s,hbar=1.0):


    assert float(2*s).is_integer()

    N = int(2*s+1)

    Sy = np.zeros((N,N),dtype=complex)

    # Upper diagonal
    for a in range(1,N):
        b = a+1

        Sy[a-1,b-1] =  -1.j*np.sqrt( (s+1)*(a+b-1) - a*b )

    # Make Hermitian
    Sy += np.transpose(np.conjugate(Sy))

    return (hbar/2) * Sy




# Operators for ion traps (Linke definition)
def R_axis_xy_plane(theta,phi):
    
    sinHalfTheta = np.sin(theta/2)

    r = np.zeros((2,2),dtype=complex)
    r[0,0] = r[1,1] = np.cos(theta/2)
    r[0,1] = -1j * np.exp(-1j*phi) * sinHalfTheta
    r[1,0] = -1j * np.exp( 1j*phi) * sinHalfTheta
    
    return r


# R-z rotation (standard definition--meh, sort of)
def R_z(theta):

    rz = np.zeros((2,2),dtype=complex)
    #rz[0,0] = np.exp(-1j*theta/2)
    #rz[1,1] = np.exp( 1j*theta/2)
    rz[0,0] = np.exp(-1j*theta)
    rz[1,1] = np.exp( 1j*theta)
    return rz

# R-x rot
def R_x(theta):

    #rx = np.array([[ np.cos(theta/2)    ,  -1j*np.sin(theta/2) ],
    #               [ -1j*np.sin(theta/2),   np.cos(theta/2) ]])
    rx = np.array([[ np.cos(theta)    ,  -1j*np.sin(theta) ],
                   [ -1j*np.sin(theta),   np.cos(theta) ]])
    return rx

# R-y rot
def R_y(theta):

    #ry = np.array([[ np.cos(theta/2),  -np.sin(theta/2) ],
    #               [ np.sin(theta/2),   np.cos(theta/2) ]])
    ry = np.array([[ np.cos(theta),  -np.sin(theta) ],
                   [ np.sin(theta),   np.cos(theta) ]])
    return ry

# CNOT
def cnot():
    g = np.zeros((4,4),dtype=complex)

    g[0,0] = g[1,1] = g[2,3] = g[3,2] = 1.
    return g


# XX(chi) gate
def m_s_gate(chi):
    # Linke: Can be varied between 0 and pi/4
    # Maximally entangling for +/- pi/4

    # This is expm(-i chi XX/2), or something similar
    
    c = np.cos(chi)
    s = -1.j * np.sin(chi)
    
    msgate = np.zeros((4,4),dtype=complex)
    msgate[0,0] = msgate[1,1] = msgate[2,2] = msgate[3,3] = c
    msgate[0,3] = msgate[1,2] = msgate[2,1] = msgate[3,0] = s
    
    return msgate
























