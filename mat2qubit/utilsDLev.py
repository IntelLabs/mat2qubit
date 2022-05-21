# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# utilsDLev.py

"""Utility function for mat2qubit."""

from __future__ import absolute_import

from . import dLevelSystemEncodings
dLevelSubsystem = dLevelSystemEncodings.dLevelSubsystem
compositeDLevels = dLevelSystemEncodings.compositeDLevels
compositeOperator = dLevelSystemEncodings.compositeOperator
compositeQasmBuilder = dLevelSystemEncodings.compositeQasmBuilder
from . import integer2bit as i2b
from . import qopmats
#from . import dLevelSubsystem,compositeDLevels,compositeOperator,compositeQasmBuilder

#from mat2qubit import dLevelSubsystem,compositeDLevels,compositeOperator,compositeQasmBuilder
#from mat2qubit import integer2bit as i2b
#from mat2qubit import qopmats




from openfermion import QubitOperator, BosonOperator, QuadOperator
from openfermion import is_hermitian, count_qubits, hermitian_conjugated, qubit_operator_sparse


import numpy as np
import scipy.linalg as la
import scipy.sparse as spr

import copy



import itertools
import functools

import random

EQ_TOLERANCE=1e-12







def bitstring_to_state_id(bitlist):
    '''Takes bitstring and returns position in overall state vector
    
    Args:
        bitlist: iterable of 0 & 1, e.g. (0,1,1,0)
        
    Returns: State id (integer)
    '''

    # b = "01001111"
    # x = int(b, 2)

    # Reversing bitlist because it's defined as first digit being least significant.
    return int(''.join( [str(b) for b in list(bitlist)[::-1]] ), 2)



def integerstring_to_bitstring(intstring,sysdlev):
    '''Takes list of integers and returns a list of bits 
    
    Args:
        intstring: iterable of integers, e.g. (0,3,1,1)
        sysdlev: object containing encoding and d info (compositeDLevels or compositeOperator)

    Returns: tuple of 0 & 1
    '''


    assert isinstance(sysdlev, compositeDLevels)  # compositeOperator is a subclass of compositeDLevels

    dvals = [ ss.d for ss in sysdlev.subsystems ]
    encs  = [ ss.enc for ss in sysdlev.subsystems ]
    encparams = [ ss.encParams for ss in sysdlev.subsystems ]
    num_ss = len(dvals)
    assert num_ss==len(intstring), "Not equal: {},{}".format(num_ss,intstring)

    # definition: int2bits(inp,lmax,enc,params=None)
    # Remember, lmax = d-1
    bstring = [ i2b.int2bits(intstring[i],dvals[i]-1,encs[i],encparams[i]) for i in range(num_ss) ]


    return tuple( itertools.chain.from_iterable(bstring) )


def integerstring_to_state_id(intstring,sysdlev):
    '''Takes list of integers and returns position in overall state vector
    
    Args:
        intstring: iterable of integers, e.g. (0,3,1,1)
        sysdlev: object containing encoding and d info (compositeDLevels or compositeOperator)

    Returns: tuple of 0 & 1
    '''

    bitlist = integerstring_to_bitstring(intstring,sysdlev)
    return bitstring_to_state_id(bitlist)



def int_strings_to_psi_notnormalized(int_strings, d, enc, dtype=complex ):
    '''Returns Hilbert space-sized vector with 1's in each int string's position.

    Args:
        int_strings: iterable of iterable of strings [e.g. ((0,2), (1,2))]
        d:           levels per site
        enc:         encoding (string)
    '''
    
    num_ss = len(int_strings[0])

    compos_op = compositeDLevels( )
    for i in range(num_ss):
        compos_op.appendSubsystem( dLevelSubsystem(d=d,enc=enc) )

    nqub = compos_op.getNumQub()
    sizehilb = 2**nqub
    psi_notnrml = np.zeros(sizehilb,dtype=dtype)
    
    for intstr in int_strings:
        # Assert all lengths the same
        assert len(intstr)==num_ss
        
        state_id = integerstring_to_state_id( intstr , compos_op )
        psi_notnrml[state_id] = 1.
    

    return psi_notnrml
    

def int_strings_to_psi(int_strings, d, enc, dtype=complex):
    '''Same as int_strings_to_psi_notnormalized() but normalized state'''
    
    psi_nn = int_strings_to_psi_notnormalized(int_strings, d, enc, dtype)
    return psi_nn / la.norm(psi_nn)
 

def int_strings_to_projector(int_strings, d, enc, dtype=complex ):
    '''Returns projector for given integer strings.

    Args:
        int_strings: iterable of iterable of strings [e.g. ((0,2), (1,2))]
        d:           levels per site
        enc:         encoding (string)
    '''
    
    states = int_strings_to_psi_notnormalized(int_strings, d, enc, dtype=complex )

    return spr.diags( states )


def permutation_superposition_notnormalized( sites , d , enc , dtype=complex ):
    '''Returns equal superposition of all permutations, *not* normalized.
    
    Note: for now, assumes all d and encodings are the same.

    Args:
        sites: number of sites
        d:     levels per site
        enc:   encoding (string)
        
    Returns: np.array
    '''

    assert sites<=d, "Must have sites<=d  [{},{}]".format(sites,d)

    compos_op = compositeDLevels( )
    for i in range(sites):
        compos_op.appendSubsystem( dLevelSubsystem(d=d,enc=enc) )


    perms = list( itertools.permutations( list(range(d)) , sites ) )

    psi_nn = int_strings_to_psi_notnormalized( perms, d,enc )

    return psi_nn



def permutation_superposition( sites , d , enc , dtype=complex ):
    '''Returns equal superposition of all permutations
    
    Note: for now, assumes all d and encodings are the same.

    Args:
        sites: number of sites
        d:     levels per site
        enc:   encoding (string)
        
    Returns: np.array
    '''

    psi = permutation_superposition_notnormalized( sites , d , enc , dtype=dtype )
    
    psi = psi/la.norm(psi)

    return psi



def permutation_projector( sites , d , enc , dtype=float ):
    '''Returns sparse superposition
    
    Note: for now, assumes all d and encodings are the same.

    Args:
        sites: number of sites
        d:     levels per site
        enc:   encoding (string)

    Returns: scipy.sparse
    '''

    states = permutation_superposition_notnormalized( sites , d , enc , dtype=dtype )

    return spr.diags( states )
    









    

def parseBraket(f):
    '''
    Returns compositeOperator() instance

    Parses a braket-format operator file. This function ignores BREAKs.
    
    f can be a file object or a StringIO/cStringIO/etc

    File format:
    NUMREGISTERS 3
    REGSIZES 200,200,200
    5.60216 |0:0><0:0|
    5.60216 |0:0,1:0><0:0,1:0|
    11.2043 |0:0,1:0><0:0,1:0|
    11.2043 |0:0,1:0,2:0><0:0,1:0,2:0|
    16.8065 |0:0,2:0><0:0,2:0|
    BREAK
    0.785398 |0:1><0:0|
    -0.785398 |0:1,1:1><0:0,1:1|
    1.11072 |0:1,1:1><0:0,1:1|
    1.11072 |0:0,1:1><0:0,1:0|
    0.785398 |0:0><0:1|
    0.785398 |0:0,1:1><0:1,1:1|
    1.11072 |0:0,1:0><0:0,1:1|
    1.11072 |0:0,1:1><0:1,1:1|
    ...
    '''


    line = f.readline()

    # First line should have "NUMREGISTERS"
    spl = line.split()
    assert spl[0]=="NUMREGISTERS"
    numReg = int(spl[1])

    # Next line is the register sizes
    line = f.readline()
    spl = line.split()
    assert spl[0]=="REGSIZES"
    regsizes = [int(x) for x in spl[1].split(',')]
    assert numReg==len(regsizes)

    # Initialize object
    composDOp = compositeOperator(  )
    for regid,regsize in enumerate(regsizes):
        composDOp.appendSubsystem( dLevelSubsystem(regsize) )

    line = f.readline()

    while line:

        # Ignoring BREAKs. Use parseBraketToQasmBuilder() if BREAKs relevant.
        if line.strip()=="BREAK":
            line = f.readline()
            continue

        spl = line.split()
        k = float(spl[0])
        ketbra = spl[1].split("><")
        ket = ketbra[0][1:]
        bra = ketbra[1][:-1]

        # Ket and Bra, 2xL arrays with {ssid,level}
        ketArr = np.array([ ssid_val.split(':') for ssid_val in ket.split(',') ]).astype(int).T
        braArr = np.array([ ssid_val.split(':') for ssid_val in bra.split(',') ]).astype(int).T

        # assert ketArr.shape==braArr.shape  # Same num terms in ket and bra
        assert ketArr[0,:].tolist()==braArr[0,:].tolist()  # Same subsystem ids in ket and bra

        opString = []
        for i in range( ketArr.shape[1] ):
            ssid = ketArr[0,i]
            d = composDOp.subsystems[ssid].d
            elementOperator = qopmats.singleElement(d, ketArr[1,i],braArr[1,i])
            opString.append( (ketArr[0,i],  elementOperator  ) )

        composDOp.addHamTerm( k,opString )

        line = f.readline()

    return composDOp



def parseBraketToQasmBuilder(f):
    '''
    Returns compositeQasmBuilder() instance, its variable pointerToCompositeSys set.
    '''
    
    line = f.readline()

    # First line should have "NUMREGISTERS"
    spl = line.split()
    assert spl[0]=="NUMREGISTERS"
    numReg = int(spl[1])

    # Next line is the register sizes
    line = f.readline()
    spl = line.split()
    assert spl[0]=="REGSIZES"
    regsizes = [int(x) for x in spl[1].split(',')]
    assert numReg==len(regsizes)

    # Initialize Qasm builder
    qasmBuilder = compositeQasmBuilder()

    # Initialize composite-system inside qasmBuilder
    qasmBuilder.pointerToCompositeSys = compositeDLevels( )
    for regid,regsize in enumerate(regsizes):
        qasmBuilder.pointerToCompositeSys.appendSubsystem( dLevelSubsystem(regsize) )


    line = f.readline()

    while line:


        # Including the BREAKs here.
        if line.strip()=="BREAK":
            qasmBuilder.addBreak()

        else:
            spl = line.split()
            k = float(spl[0])
            ketbra = spl[1].split("><")
            ket = ketbra[0][1:]
            bra = ketbra[1][:-1]

            # Ket and Bra, 2xL arrays with {ssid,level}
            ketArr = np.array([ ssid_val.split(':') for ssid_val in ket.split(',') ]).astype(int).T
            braArr = np.array([ ssid_val.split(':') for ssid_val in bra.split(',') ]).astype(int).T

            # assert ketArr.shape==braArr.shape  # Same num terms in ket and bra
            assert ketArr[0,:].tolist()==braArr[0,:].tolist()  # Same subsystem ids in ket and bra

            opString = []
            for i in range( ketArr.shape[1] ):
                ssid = ketArr[0,i]
                d = regsizes[ssid]
                elementOperator = qopmats.singleElement(d, ketArr[1,i],braArr[1,i])
                opString.append( (ketArr[0,i],  elementOperator  ) )

            qasmBuilder.addHamTerm( k,opString )

        line = f.readline()

    return qasmBuilder



































