# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# integer2bit.py

'''
Functions for converting integer to bits for various encodings.
'''

# Idea is to have functions that return a function

import numpy as np

encodings = ["stdbinary","unary","gray","blockunary"]


# 'Standard binary' rep of an integer
def dec2stdbin(inp,lmax):
    """'Standard binary' rep of an integer"""
    # Returns list of 0's and 1's
    binlist = [int(x) for x in bin(inp)[2:]]
    binlist = list(reversed(binlist))

    padToPlaces = int( np.ceil( np.log2(lmax+1) ) )

    if padToPlaces>len(binlist):
        binlist += [0]*(padToPlaces-len(binlist))
    return binlist


# 'Unary' rep of an integer
def dec2unary(inp,lmax):
    """'Unary' rep of an integer"""
    # Returns list of 0's and 1's
    
    padToPlaces = lmax+1

    assert(inp<padToPlaces)

    list01 = [0]*padToPlaces
    list01[inp] = 1
    
    return list01


# Gray code
# Uses reversible circuit
def dec2gray(inp,lmax):
    """Gray code"""
    bitlist = dec2stdbin(inp,lmax)

    nbits = len(bitlist)
    for bitId in range(1,nbits):
        if bitlist[bitId]==1:
            changedBit = bitlist[bitId-1]
            bitlist[bitId-1] = 1 if changedBit==0 else 0
    return bitlist


# Returns a function for blockUnary
def getfunc_dec2blockunary(g,localEncodingFunc):
    """Returns a function for blockUnary"""

    # localEncodingFunc can either be a string or a function.
    # If localEncodingFunc is a string, get the explicit function:
    if isinstance(localEncodingFunc,str):
        enc = localEncodingFunc
        if not (enc in encodings):
            raise Exception(enc+" is not one of the available encoding types")
            # re-assign variable to appropriate function
        localEncodingFunc = getfunc_int2bits(enc)

    def blockUnaryFunc(inp,lmax):
        # Bits per block
        blockSize = int(np.ceil(np.log2(g+1)))
        # Number of blocks
        nBlocks = int(np.ceil(  (lmax+1)/g ))
        # Value of block
        blockVal = (inp % g) + 1
        # Bitstring of block
        blockBitList = localEncodingFunc(blockVal,g)

        # What block is inp in?
        blockId = int( np.floor( inp/g ) )
        
        # print("")
        # print("***inp: {} ***".format(inp) )
        # print("blockVal: ", blockVal)
        # print("nBlocks: ",nBlocks)
        # print("blockId: ",blockId)

        bits = [0]*(blockSize*nBlocks)
        for blockBit,bitval in enumerate(blockBitList):
            globalBit = blockBit + blockSize*blockId
            bits[globalBit] = bitval
        return bits

    return blockUnaryFunc


# Block unary function for arbitrary params
def dec2blockunary(inp,lmax,g,localEncodingFunc):
    """Block unary function for arbitrary params"""

    f = getfunc_dec2blockunary(g,localEncodingFunc)
    return f(inp,lmax)


# Returns the relevant bits--the ones that need to be considered
def getBitMask(intval,lmax,enc,params=None):
    '''Returns the relevant bitmask subset.
    
    Returns a list of True and False.'''

    enc,params = processEncodingString(enc,params)

    if enc=="stdbinary" or enc=="gray":
        return [True]*int(np.ceil(np.log2(lmax+1)))
    elif enc=="unary":
        mask = [False]*(lmax+1)
        mask[intval] = True
        return mask
    elif enc=="blockunary":

        g = params['g']

        # Bits per block
        blockSize = int(np.ceil(np.log2(g+1)))
        # Number of blocks
        nBlocks = int(np.ceil(  (lmax+1)/g ))
        # Value of block
        blockVal = (intval % g) + 1
        # What block is the val in?
        blockId = int( np.floor( intval/g ) )

        mask = [False]*(blockSize*nBlocks)
        bitIds = (blockId*blockSize) + np.arange(blockSize)
        for i in bitIds:
            mask[i] = True
        return mask
    else:
        raise Exception("Invalid encoding: ",enc)


# Function generator for arbitrary encoding
def getfunc_int2bits(enc,params=None):
    """Function generator for arbitrary encoding"""

    enc,params = processEncodingString(enc,params)

    # if not (enc in encodings):
    #     raise Exception(enc+" is not one of the available encoding types.")

    if enc=="stdbinary":
        return dec2stdbin
    elif enc=="unary":
        return dec2unary
    elif enc=="gray":
        return dec2gray
    elif enc=="blockunary":
        return getfunc_dec2blockunary(**params)


def int2bits(inp,lmax,enc,params=None):
    
    f = getfunc_int2bits(enc,params)
    
    if inp>lmax: raise IndexError("inp out of bounds: ({},{})".format(inp,lmax) )

    return f(inp,lmax)


def getBitCount(lmax,enc,params=None):

    # assert(enc in encodings), "Input 'enc' was: "+enc
    enc,params = processEncodingString(enc,params)
    
    d = lmax+1

    if enc=="stdbinary" or enc=="gray":
        return int(np.ceil(np.log2(d)))
    elif enc=="unary":
        return d
    elif enc=="blockunary":
        g = params['g']
        bitcount = int(np.ceil(np.log2(g+1))) * int(np.ceil(float(d)/g))
        return bitcount
    else:
        raise Exception("Invalid encoding: ",enc)


def getMaxDFromNumBits(nbits,enc,params=None):
    
    enc,params = processEncodingString(enc,params)

    if enc=="gray" or enc=="stdbinary":
        return 2**nbits
    elif enc=="unary":
        return nbits
    elif enc=="blockunary":
        g = params['g']
        blocksize = int(np.ceil(np.log2(g+1)))
        # assert (nbits/blocksize).is_integer(), "For now we require nbits/blocksize to be an integer, in this function."
        remainder = nbits % blocksize
        remainder = 2**remainder - 1
        return int( g*np.floor(nbits/blocksize) + remainder )
    else:
        raise Exception("Invalid encoding: ",enc)
        

def IsValidEncodingType(strEnc):
    """At the moment, the only type of parametrized encoding that you
    can do is e.g. "bu_gray_3".
    """

    if strEnc in encodings:
        return True

    spl = strEnc.split("_")
    if spl[0]=="bu":

        assert spl[1] in encodings, "{} --> {} not valid.".format(strEnc,spl[1])
        assert spl[2].isdigit(), "{} --> {} not an int.".format(strEnc,spl[2])

        return True


    # If reached here, it's not a valid encoding
    return False


def processEncodingString(enc,encParams):

    if enc in encodings:
        return (enc,encParams)

    assert IsValidEncodingType(enc), "'{}' is not valid encoding.".format(enc)

    spl = enc.split("_")
    if spl[0]=="bu":

        enc = "blockunary"
        encParams = {}
        encParams['localEncodingFunc'] = spl[1]
        encParams['g'] = int(spl[2])

    return (enc,encParams)


# # Gray code
# # https://stackoverflow.com/questions/38738835/generating-gray-codes
# def dec2gray(inp,lmax):
#     gray = stdbin2gray( dec2stdbin(inp,lmax=inp) )  # unpadded
#     padToPlaces = int( np.ceil( np.log2(lmax+1) ) )
#     if padToPlaces>len(gray):
#         gray += [0]*(padToPlaces-len(gray))
#     return gray

# # https://rosettacode.org/wiki/Gray_code#Python
# # Content is available under GNU Free Documentation License 1.2 unless otherwise noted.
# def stdbin2gray(bits):
#     # Only works on unpadded strings
#     bits = bits[:1] + [i ^ ishift for i, ishift in zip(bits[:-1], bits[1:])]
#     bits.reverse()
#     return bits
























