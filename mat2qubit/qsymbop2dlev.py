# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

'''
qsymbop2dlev.py

Converting qSymbOp operator to a mat2qubit operator.
'''


import numpy as np
import itertools

from . import dLevelSubsystem,compositeDLevels,compositeOperator
from . import qopmats
from . import qSymbOp
#from . import qSymbOp as qSymbOp_module
#qSymbOp = qSymbOp_module.qSymbOp

#from mat2qubit import dLevelSubsystem,compositeDLevels,compositeOperator, qopmats,qSymbOp

from copy import copy


# Built-in operators. These match with the mat2qubit library.
# Should be ignored if input dict has a corresponding entry.
operatorChars  = {}
operatorChars['n']   = "numop"
operatorChars['Nsq']   = "Nsq"
operatorChars['p']   = "qhoMom"
operatorChars['Psq'] = "Psq"
operatorChars['p2']  = "p2"
operatorChars['q']   = "q"
operatorChars['q2']  = "q2"  # pos squared
operatorChars['q3']  = "q3"  # pos**3
operatorChars['q4']  = "q4"  # pos**4
operatorChars['q5']  = "q5"  # pos**5
operatorChars['q6']  = "q6"  # pos**6
operatorChars['cr']  = "qhoCr"
operatorChars['an']  = "qhoAn"
operatorChars['ad']  = "qhoCr"
operatorChars['a']   = "qhoAn"
operatorChars['Sx']  = "Sx"
operatorChars['Sy']  = "Sy"
operatorChars['Sz']  = "Sz"
dmax_deflt = 100
projector_dict = dict( [("Pr{}".format(a),"Pr{}".format(a)) for a in range(dmax_deflt) ] )
operatorChars.update(projector_dict)
ketbra_dict = dict( [("k{}b{}".format(a,b),"k{}b{}".format(a,b)) for a,b in itertools.product(range(dmax_deflt),range(dmax_deflt)) ] )
operatorChars.update(ketbra_dict)






def symbop_to_dlevcompositeop(inpSymbop,ssname_order,dvals,encodings,inpOpChars={}):
    '''Convert qSymbOp to mat2qubit operator.

    dvals and encodings may be single value, iterable, or dict (with subsystems as keys).

    The input dict of opchars_to_func allows for strings, function defs, and np.array
    '''

    
    numSs = len(ssname_order)
    
    if isinstance(dvals,int):
        # All values meant to be the same
        dvals = dict(zip(ssname_order,[dvals]*numSs))
    if not isinstance(dvals,dict):
        # Assume they inserted a list or iterable
        assert len(dvals)==numSs
        dvals = dict(zip(ssname_order,dvals))
    
    if isinstance(encodings,str):
        # All encs meant to be the same
        encodings = dict(zip(ssname_order,[encodings]*numSs))
    if not isinstance(encodings,dict):
        # Assume they insterted a list or iterable
        assert len(encodings)==numSs
        encodings = dict(zip(ssname_order,encodings))

    assert len(dvals)==numSs
    assert len(encodings)==numSs

    assert set(ssname_order)==set(dvals.keys())
    assert set(ssname_order)==set(encodings.keys())

    composOp = compositeOperator()

    ssname_to_ssid = dict( zip( ssname_order,range(numSs) ) )


    # Create each d-level particle
    for ssname in ssname_order:

        dlev = dLevelSubsystem(d=dvals[ssname],enc=encodings[ssname],inpName=ssname)
        composOp.appendSubsystem( dlev )




    # Populate the operators
    for opTupleSymb,kSymb in inpSymbop.fullOperator.items():

        # Convert constant
        try:
            k = float(kSymb)
        except TypeError:
            try:
                k = complex(kSymb)
            except TypeError:
                # print("Error. Cannot convert coefficient {} to complex or float.".
                #                   format(kSymb))
                raise

        tupleList = []



        # Identity
        if len(opTupleSymb)==1 and len(opTupleSymb[0])==0:

            composOp.addIdentityTerm(k)
            continue


        # Non-identity
        for locop in opTupleSymb:

            ssname = locop[0]
            opname = locop[1]

            # assert (opname in operatorChars.keys()) or \
            #       (opname in inpOpChars.keys()), 'opname: '+opname

            if opname in inpOpChars.keys():  # inpOpChars should override operatorChars
                op = copy(inpOpChars[opname])
            elif opname in operatorChars.keys():
                op = copy(operatorChars[opname])
            else:
                raise Exception("Invalid name for op name: " + 
                        "{}. User may use inpOpChars.".format(opname))
            

            ssid = ssname_to_ssid[ssname]

            tupleList.append( (ssid,op) )


        composOp.addHamTerm(k,tupleList)


    return composOp




def symbop_pauli_to_mat(inpString,ssname_order):
    '''Takes a string (e.g. '2.1 [X_0 Y_2 Z_3]') and returns matrix operator.
    
    Input is a string, e.g. '2.1 [X_0 Y_2 Z_3]', *not* a qsymbop.

    Only allows operators X,Y,Z. Only allows numbers as ssid's.
    
    Outputs full Hilbert represenation

    Args:
        
    '''
    
    # If you want to look through to make sure all the ssid's are
    # numeric, can use: qSymbOp.ssid_set
    #
    # At the moment, doesn't check whether all inputted locops are {X,Y,Z}

    #symbop = qSymbOp.qSymbOp(inpString)
    symbop = qSymbOp(inpString)

    pauli_defs = {'X':qopmats.qub['X'],
                  'Y':qopmats.qub['Y'],
                  'Z':qopmats.qub['Z']}
    

    dVals = dict(zip(ssname_order, [2]*len(ssname_order) ))
    encodings = dict(zip(ssname_order, ["stdbinary"]*len(ssname_order) ))

    compOp = symbop_to_dlevcompositeop(symbop,ssname_order,dVals,encodings,pauli_defs)
    
    return compOp.toFullMatRep()
    






def symbop_to_QubitOperator(inpSymbop,ssname_order,dvals,encodings,inpOpChars={}):
    '''Input symbolic operator, output a Pauli operator based on d-level encodings.'''


    dlevecompos = symbop_to_dlevcompositeop(inpSymbop,ssname_order,dvals,encodings,inpOpChars)

    return dlevecompos.opToPauli()


















































