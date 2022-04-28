# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# helperDLev.py

"""Helper functions for mat2qubit."""


from openfermion import QubitOperator, count_qubits, qubit_operator_sparse
import numpy as np 
import functools
import operator

import random

EQ_TOLERANCE=1e-12



    
# Single-qubit operators |i><j| (four possibilities)
def sglQubOp(i,j, qubid):

    if   (i,j)==(0,0):
        return 0.5 * (QubitOperator.identity() + QubitOperator( "Z{}".format(qubid) ))
    elif (i,j)==(1,1):
        return 0.5 * (QubitOperator.identity() - QubitOperator( "Z{}".format(qubid) ))
    elif (i,j)==(0,1):
        return 0.5 * (QubitOperator( "X{}".format(qubid) ) + 1j*QubitOperator("Y{}".format(qubid) ))
    elif (i,j)==(1,0):
        return 0.5 * (QubitOperator( "X{}".format(qubid) ) - 1j*QubitOperator("Y{}".format(qubid) ))
    else:
        raise Exception("i,j must be binary. (  i,j = "+str(i)+","+str(j)+"  )")



def pauli_op_to_matrix( pop , n_qubits=None , m2q_ordering=True ):
    '''Takes in Pauli operator [QubitOperator] and outputs matrix based on mat2qubit's ordering.

    Note: default mat2qubit's ordering is the reverse of what some orderings use.

    Args:
        Pauli op (QubitOperator)

    Returns:
        Matrix repr (scipy.sparse.spmatrix)
    '''

    # Set numqubits if not set
    pop_cnt = count_qubits(pop)
    if not n_qubits:
        n_qubits = pop_cnt
    assert n_qubits >= pop_cnt

    # Reverse the ordering before running qubit_operator_sparse
    if m2q_ordering:
        poplist = [ QubitOperator([ (n_qubits-1-str_k[0],str_k[1]) for str_k in term[0] ],term[1]) for term in pop.terms.items() ]
        pop = functools.reduce(operator.add,poplist)

    else:
        # new_pop = pop
        pass

    # Return sparse matrix
    return qubit_operator_sparse( pop , n_qubits )



# Counts CNOTs, assuming absolutely no circuit optimization
def countCNOTs_trot1_noopt(qubop):
    assert( isinstance(qubop,QubitOperator) )

    # print(qubop)
    assert is_hermitian(qubop), ("Qubop: "+str(qubop))

    ctrCnots = 0
    terms = qubop.terms

    for term in terms:

        nPaulis = len(term)
        if nPaulis>1:
            ctrCnots += 2* ( len(term) - 1 )

    return ctrCnots

'''
def getFullHilbRepFromLocOp(locop,locId,subspaceSizes, dtype_default=complex):

    numSS = len(subspaceSizes)

    assert locop.shape[0]==locop.shape[1] , locop.shape # confirm square
    assert locop.shape[0]==subspaceSizes[locId]

    fullHilbOp = np.eye(1,dtype=dtype_default)

    for i in range(locId):
        fullHilbOp = np.kron(fullHilbOp,np.eye(subspaceSizes[i]))
    fullHilbOp = np.kron(fullHilbOp,locop)
    for i in range(locId+1,numSS):
        fullHilbOp = np.kron(fullHilbOp,np.eye(subspaceSizes[i]))

    return fullHilbOp
'''




































