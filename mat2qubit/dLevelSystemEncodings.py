# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# dLevelSystemEncodings.py

"""Contains classes for single- and multi-particle systems"""

import functools
import itertools
import re
from copy import deepcopy

# Common libraries
import numpy as np
import scipy.sparse as spr

# Openfermion methods
from openfermion import QubitOperator
from openfermion.utils import is_hermitian

# This package
from . import helperDLev
from . import integer2bit as i2b
from . import qopmats

sglQubOp = helperDLev.sglQubOp
countCNOTs_trot1_noopt = helperDLev.countCNOTs_trot1_noopt
pauli_op_to_matrix = helperDLev.pauli_op_to_matrix

# from . import helperDLev.sglQubOp as sglQubOp
# from . import helperDLev.countCNOTs_trot1_noopt as countCNOTs_trot1_noopt
# from . import helperDLev.countCNOTs_trot1_noopt as pauli_op_to_matrix


# import mat2qubit.integer2bit as i2b
# from mat2qubit import qopmats
# from mat2qubit.helperDLev import sglQubOp,countCNOTs_trot1_noopt,pauli_op_to_matrix


# Built-in ops. These operators don't require matrix repr input.
dmax_builtinops = 100
builtInOps = (
    [
        "ident",
        "numop",
        "Nsq",
        "qhoCr",
        "qhoAn",
        "qhoPos",
        "qhoMom",
        "q",
        "p",
        "Qsq",
        "Psq",
        "p2",
        "q2",
        "q3",
        "q4",
        "q5",
        "q6",  # Powers of position operator
        "a",
        "ad",
        "n",  # Creation/annihilation ops
        "Sx",
        "Sy",
        "Sz",
        "X_1stQ",
    ]
    + ["Pr{}".format(a) for a in range(dmax_builtinops)]
    + [
        "k{}b{}".format(a, b)
        for a, b in itertools.product(range(dmax_builtinops), range(dmax_builtinops))
    ]
)
# Pr{} -- Projector to one level
# k{}b{} -- "ketbra", e.g. k1b2==|1><2|

# IMPORTANT NOTE: The projection operators, just like any other operator, ignores
# the bits outside the bitmask. This means that in non-compact codes, "Pr{}"
# does *not* really behave like a projector. Hence the 'PrX'
# operators are used mainly in the sense of denoting a value on the bits.


class dLevelSubsystem:

    """Class for single d-level particle.

    This class defines a particle with a particular 'd' and encoding. It
    can output the Pauli representation of an d-by-d operator.

    Attributes:
        d (int): Number of levels in the 'particle'.
        enc (string): Encoding
        encParams (dict): Optional dict of params related to encoding (used
            for block unary)
        name (string): Optional name for the 'particle'.
    """

    def __init__(self, d, enc=None, encParams=None, inpName=None):
        """
        Args:
            d (int): Number of levels in the 'particle'.
            enc (string): Encoding
            encParams (dict): Optional dict of params related to encoding (used
                for block unary)
            name (string): Optional name for the 'particle'.
        """
        self.d = d
        self.enc = enc
        assert encParams is None or isinstance(encParams, dict)
        self.encParams = deepcopy(encParams)

        self.qubitShift = 0
        self.nqubits = 0
        if enc is not None:
            self.nqubits = i2b.getBitCount(d - 1, enc, encParams)

        self.name = ""
        if inpName:
            self.name = inpName

        # self.tags = set()
        # if inpTag:
        #     self.tags.add(inpTag)

    def setEncoding(self, enc, encParams=None):
        """Set the encoding for the particle.

        For block unary, user may use e.g. enc='bu_gray_3' instead of
            specifying encParams.
        """

        # This parses single-string encodings like "bu_gray_3" or else returns no change
        enc, encParams = i2b.processEncodingString(enc, encParams)

        assert encParams is None or isinstance(encParams, dict)

        self.enc = enc
        self.encParams = deepcopy(encParams)

        self.nqubits = i2b.getBitCount(self.d - 1, enc, encParams)

    def set_d(self, d):
        """Set d for the particle.

        Args:
            d (int): Number of levels in particle.
        """

        assert isinstance(d, (int, np.integer))

        self.d = d
        if self.enc is None:
            self.nqubits = 0
        else:
            self.nqubits = i2b.getBitCount(self.d - 1, self.enc, self.encParams)

    def setQubitShift(self, shiftval):
        """Set Qubit Shift

        When converted to Pauli operator, this specifies how many qubits to
        shift by. The method is used by compositeDLevels() and compositeOperator().

        Args:
            shiftval (int): Number of qubits to shift by.
        """
        self.qubitShift = shiftval

    def opToMatrixRep(self, inpOp):
        """Returns the matrix representation of a d-by-d operator.

        If inpOp is a string, returns matrix repr. If inpOp is an np.array,
            it returns the input.

        Args:
            inpOp (string, numpy.array): d-by-d operator
        """

        if isinstance(inpOp, np.ndarray):
            return inpOp

        else:

            assert inpOp in builtInOps, inpOp

            if inpOp == "ident":
                return np.eye(self.d)
            elif inpOp == "numop":
                return qopmats.numop(self.d)
            elif inpOp == "Nsq":
                return qopmats.Nsq(self.d)
            elif inpOp == "qhoCr":
                return qopmats.cr(self.d)
            elif inpOp == "qhoAn":
                return qopmats.an(self.d)

            elif inpOp == "ad":
                return qopmats.cr(self.d)
            elif inpOp == "a":
                return qopmats.an(self.d)
            elif inpOp == "n":
                return qopmats.numop(self.d)

            elif inpOp == "qhoPos" or inpOp == "q":
                return qopmats.posQho(self.d)
            elif inpOp == "qhoMom" or inpOp == "p":
                return qopmats.momQho(self.d)
            elif inpOp == "Qsq":
                return qopmats.posQhoSq(self.d)
            elif inpOp == "Psq" or inpOp == "p2":
                return qopmats.momQhoSq(self.d)

            elif inpOp == "q2":
                return qopmats.q2(self.d)
            elif inpOp == "q3":
                return qopmats.q3(self.d)
            elif inpOp == "q4":
                return qopmats.q4(self.d)
            elif inpOp == "q5":
                return qopmats.q5(self.d)
            elif inpOp == "q6":
                return qopmats.q6(self.d)

            elif inpOp == "Sx":
                return qopmats.spinX(float(self.d - 1) / 2.0)
            elif inpOp == "Sy":
                return qopmats.spinY(float(self.d - 1) / 2.0)
            elif inpOp == "Sz":
                return qopmats.spinZ(float(self.d - 1) / 2.0)
            elif inpOp == "X_1stQ":
                return qopmats.X_1stQuant(self.d, 1)

            elif inpOp[:2] == "Pr":  # Projector
                lev = int(inpOp[2:])
                return qopmats.proj(lev, self.d)

                # IMPORTANT NOTE: The projection operators, just like any other
                # operator, ignores the bits outside the bitmask.
                # This means that in non-compact codes, "Pr{}" does *not* really
                # behave like a projector. Hence the 'PrX' operators are used mainly
                # in the sense of denoting a value on the bits.

            elif (
                len(re.findall(r"k\d+b\d+$", inpOp)) == 1
            ):  # KetBra ('^' and $' are 'anchors' in regex)

                spl = inpOp.split("b")
                ket = int(spl[0][1:])
                bra = int(spl[1])
                return qopmats.singleElement(self.d, ket, bra)

            else:
                raise Exception("Operator '" + inpOp + "' not supported")

    def opToPauli(self, inpOp):
        """Returns the Pauli operator for the inputted operator

        This may be a string (member of builtInOps) or an explicit matrix
            representation as a numpy.array.

        Args:
            inpOp (string, numpy.array): d-by-d operator
        """

        if self.enc is None:
            raise Exception("Encoding for subsystem not yet set.")

        # Return identity right away if that's correct operator
        if isinstance(inpOp, np.ndarray):
            if np.array_equal(inpOp, np.eye(self.d)):
                return QubitOperator.identity()
        elif inpOp == "ident":
            return QubitOperator.identity()

        # Replace strings with appropriate matrices
        if isinstance(inpOp, str):
            inpOp = self.opToMatrixRep(inpOp)

        # Confirm matrix size
        assert inpOp.shape == (self.d, self.d), inpOp.shape

        # L_max = d - 1
        lmax = self.d - 1

        # Init full pauli operator to 0.
        fullPauliOp = QubitOperator()

        # Loop through matrix elements
        for i in range(self.d):
            for j in range(self.d):
                if inpOp[i, j] == 0.0:
                    continue

                matval = inpOp[i, j]

                I = i2b.int2bits(i, lmax, self.enc, self.encParams)  # noqa: E741
                J = i2b.int2bits(j, lmax, self.enc, self.encParams)
                imask = i2b.getBitMask(i, lmax, self.enc, self.encParams)
                jmask = i2b.getBitMask(j, lmax, self.enc, self.encParams)
                # OR operation on the two masks (union of the masks):
                mask = np.array(imask) | np.array(jmask)

                # Initialize pauli string to identity*matrixval
                pauliString = QubitOperator("", matval)

                # Then you only focus on the bits in that mask
                for bitId, boolUseBit in enumerate(mask):
                    if not boolUseBit:
                        continue

                    # Possible inputs: |0><0|,|0><1|,|1><0|,|1><1|
                    pauliString *= sglQubOp(I[bitId], J[bitId], bitId + self.qubitShift)

                fullPauliOp += pauliString

        return fullPauliOp

    def locopProductToPauli(self, locopList):
        """Convert list of products to Pauli operator

        Args:
            locopList (iterable of 1-local d-level ops): e.g.:
                ('q','n',np.array([[0.,-1j],[1j,0]]))

        """

        # Begin with identity
        matrep = np.eye(self.d)

        for locop in locopList:

            # Order matters, since not all loc ops commute.
            matrep = np.dot(matrep, self.opToMatrixRep(locop))

        return self.opToPauli(matrep)

    def getNumQub(self, recalc=True):

        if recalc:
            self.nqubits = i2b.getBitCount(self.d - 1, self.enc, self.encParams)

        return self.nqubits

    def __str__(self):

        return "Particle, d={}, enc={} [{}]".format(self.d, self.enc, self.encParams)


class compositeDLevels:

    """Class for holding multiple d-level particles.

    This class defines a composite set of d-level particles. It stores only
    the particles themselves and keeps track of relevant qubit shifts. The
    interactions between particles are implemented in the child class
    compositeOperator().

    Attributes:
        subsystems: List of dLevelSubsystem objects
        totalQubits (int): Total qubits of composite system
    """

    def __init__(self, defaultEnc=None):

        self.subsystems = []  # List of dlevel subsystem objects
        self.totalQubits = 0

    def appendSubsystem(self, inpSubSys, doCopy=True):
        """Add a subsystem (particle) to the composite system

        Args:
            inpSubSys (dLevelSubsystem): The input particle
            doCopy (bool): Deepcopy the inputted dLevelSubsystem
        """

        # doCopy determines whether to run deepcopy
        if doCopy:
            self.subsystems.append(deepcopy(inpSubSys))
        else:
            self.subsystems.append(inpSubSys)

        # Get most recently added subsystem
        ss = self.subsystems[-1]

        # Assign the correct qubit spacing, and update self.totalQubits
        ss.qubitShift = self.totalQubits
        self.totalQubits += ss.nqubits

        # return the id of the subsystem
        return len(self.subsystems) - 1

    def createAndAppendSubSystem(self, d, enc=None, encParams=None):
        """Not implemented"""
        raise Exception("Not implemented.")

    def opStringToPauli(self, coeff, opString):
        """Returns Pauli operator for a single multi-subsystem product term.

        opString is a list of tuples, [(ss1id,op1),(ss2id,op2),...];
        but for plain k*Identity, the tuple is just "ident" instead.

        To avoid superfluous terms in the Pauli representation (see arxiv:
        1909.12847 Sec III.C), the code goes as far as possible in the matrix
        representation before multiplying Pauli operators.

        Args:
            coeff (float): Coefficient for the term
            opString (iterable of pairs):
                e.g. [(0,"qhoPos"),(2,np.array([[0.,-1j],[1j,0]])),...]
                or [("ident"),]
        """

        if opString == "ident":
            return coeff * QubitOperator.identity()

        # Determine unique ssid's in this string
        # uniqueSsids = np.unique( np.array(opString)[:,0] ).astype(int)
        uniqueSsids = np.unique([t[0] for t in opString]).astype(int)

        # Dict that stores each ssid's operator
        ssidDictOps = {ssid: [] for ssid in uniqueSsids}

        # Create ordered list for each ssid
        for ctr, opTuple in enumerate(opString):
            ssid = opTuple[0]
            op = opTuple[1]

            ssidDictOps[ssid].append(op)

        # Now assemble the Pauli Operator
        pauliOp = QubitOperator.identity()  # Identity

        for ssid, locopList in ssidDictOps.items():
            pauliOp *= self.subsystems[ssid].locopProductToPauli(locopList)

        return coeff * pauliOp

    def opStringToMatRep(self, coeff, opString, ignore_encoding=False):
        """Returns matrix repr in full Hilbert space.

        Args:
            coeff (float): Coefficient for the term
            opString (iterable of pairs):
                e.g. [(0,"qhoPos"),(2,np.array([[0.,-1j],[1j,0]])),...]
                or [("ident"),]
            ignore_encoding: If True, simply outputs the full operator without
                             considering qubits at all. (e.g. a d=3 and d=5 system
                             will lead to a Hilbert space of 15.)
        """

        if ignore_encoding:

            # Loop over opString
            # (can't assume anything. It could be e.g. 'q0 p7 p0 q2 q0')
            # For dict of matrices (Identities of each subsystem size)
            num_ss = len(self.subsystems)
            mats_by_ssid = dict(
                [
                    (i, spr.eye(ssize))
                    for (i, ssize) in zip(
                        range(num_ss), [ss.d for ss in self.subsystems]
                    )
                ]
            )
            if opString != "ident":
                for ssid, locop in tuple(opString):
                    locmat = self.subsystems[ssid].opToMatrixRep(locop)
                    mats_by_ssid[ssid] = mats_by_ssid[ssid].dot(locmat)

            # [::-1] for reverse order
            spr_matrep_opstr = coeff * functools.reduce(
                spr.kron, list(mats_by_ssid.values())[::-1]
            )

        else:

            # Convert to Pauli
            pop = self.opStringToPauli(coeff, opString)

            # Then use pauli_op_to_matrix( pop , nq=None , m2q_ordering=True )
            nq = self.getNumQub()
            spr_matrep_opstr = pauli_op_to_matrix(pop, nq)

        return spr_matrep_opstr

    def setEncoding(self, ss, enc, encParams=None):
        """Not implemented"""
        raise NotImplementedError()
        # assert( enc in i2b.encodings )
        # assert( encParams==None or isinstance(encParams,dict) )

    def setEncodingForAllSS(self, enc, encParams=None):
        """Sets same encoding for all particles.

        Note that the qubitshift in each particle is updated appropriately.

        Args:
            enc (string): Encoding
            encParams (dict): Parameters for encoding (optional)
        """

        qubshift = 0

        for ssid, ss in enumerate(self.subsystems):

            ss.setEncoding(enc, encParams)  # Inside here, ss.nqubits is updated
            ss.setQubitShift(qubshift)
            # Now update qubshift
            qubshift += ss.nqubits

    def setDForAllSS(self, d):
        """Set d for all subsystems (particles)."""

        qubshift = 0

        for ssid, ss in enumerate(self.subsystems):

            ss.set_d(d)  # Inside function, ss.nqubits is determined
            ss.setQubitShift(qubshift)
            qubshift += ss.nqubits

    def getNumQub(self, recalc=True):
        """Get number of qubits for whoe composite system."""

        ds = [ss.getNumQub(recalc=recalc) for ss in self.subsystems]
        return sum(ds)


class compositeOperator(compositeDLevels):

    """Class for multi-particle operators.

    Child class of compositeDLevels. Adds interactions between particles
    to the parent class. Each member of list will is:
    (coeff,(ssid1,op1),(ssid2,op2)) , (coeff2,(ssid,op)) , ... )
    though for a k*Identity term, the list member will be
    (coeff,"ident")

    Note: This class does not consider commutation relations in any
        sophisticated way. It just takes the tensor product of the operators
        that are inserted.

    Attributes:
        subsystems: List of dLevelSubsystem objects
        totalQubits (int): Total qubits of composite system
        hamTerms (list): List of Hamiltonian terms
    """

    def __init__(self, inpCompositeSys=None, defaultEnc=None):

        assert inpCompositeSys is None or isinstance(inpCompositeSys, compositeDLevels)
        # isinstance(inpCompositeSys,super(compositeOperator)) )

        # Run the constructor of parent object. And take from inpCS.
        if inpCompositeSys is None:
            super(compositeOperator, self).__init__(defaultEnc)
        else:
            self.subsystems = deepcopy(inpCompositeSys.subsystems)
            self.totalQubits = inpCompositeSys.totalQubits

        self.hamTerms = []

    def addHamTerm(self, coeff, opString):
        """Add a term (coefficient times operator product) to operator.

        Note that one may of course have multiple operators on the same particle,
        in the same string.

        Args:
            coeff (float): coefficient for the operator term
            opString (iterable of tuples,"ident"):
                [(ss1id,op1),(ss2id,op2),...] or "ident"
        """

        # Consider adding a check for whether all ssid are valid

        if opString == "ident":
            self.addIdentityTerm(coeff)
            return

        # Rudimentary check to at least see if it's an iterable of tuples
        assert isinstance(opString[0], tuple)

        self.hamTerms.append((coeff, tuple(deepcopy(opString))))

    def addIdentityTerm(self, coeff):
        """Add identity term, with given coefficient."""
        self.hamTerms.append((coeff, "ident"))

    def opToPauli(self):
        """Return Pauli representation of this multi-particle operator."""

        pauliHam = QubitOperator()  # 0.0

        # for cmdId,cmd in enumerate(self.circCommands):
        for term in self.hamTerms:

            coeff = term[0]
            opString = term[1]

            pauliHam += self.opStringToPauli(coeff, opString)

        return pauliHam

    def toFullMatRep(self, ignore_encoding=False):
        """Returns full Hilbert space matrix representation

        If ignore_encoding is True, then it's the full Hilbert
        representation without considering qubits at all.
        """

        if ignore_encoding:
            ssSizes = [ss.d for ss in self.subsystems]
            hilbSize = np.prod(ssSizes)
        else:
            hilbSize = 2 ** self.getNumQub()

        # Start with identity
        fullmatrep = spr.lil_matrix((hilbSize, hilbSize), dtype=complex)

        for term in self.hamTerms:

            coeff = term[0]
            opString = term[1]

            fullmatrep += self.opStringToMatRep(coeff, opString, ignore_encoding)

        return fullmatrep

    def compareHamsEqual(self, inp):
        """Returns true if equal (checks if sorted operator terms are equal)."""

        assert isinstance(inp, compositeOperator)

        if not (sorted(self.hamTerms) == sorted(inp.hamTerms)):
            return False

        return True

    def __eq__(self, obj):
        """Returns true if equal (checks if sorted operator terms are equal)."""

        assert isinstance(obj, compositeOperator)

        if not isinstance(obj, compositeOperator):
            return False

        if not (sorted(self.hamTerms) == sorted(obj.hamTerms)):
            return False

        return True

    def __str__(self):

        return str(self.hamTerms)

    def __mul__(self, inp):

        # Assert is compositeOperator
        assert isinstance(inp, compositeOperator)

        # Find out which compositeOperator has more subsystems
        numSS_a = len(self.subsystems)
        numSS_b = len(inp.subsystems)

        # Confirm all subsystems have same properties
        for ssid in range(min(numSS_a, numSS_b)):
            assert self.subsystems[ssid].d == inp.subsystems[ssid].d
            assert self.subsystems[ssid].enc == inp.subsystems[ssid].enc
            assert self.subsystems[ssid].encParams == inp.subsystems[ssid].encParams

        # Create new object
        newOp = compositeOperator()

        # Populate with subsystems
        if numSS_a >= numSS_b:
            op_with_more_ss = self
        else:
            op_with_more_ss = inp
        for ssid in range(max(numSS_a, numSS_b)):  # *Max* of the two
            newOp.appendSubsystem(deepcopy(op_with_more_ss.subsystems[ssid]))

        # Multiply terms together
        for (i, termA) in enumerate(self.hamTerms):

            for (j, termB) in enumerate(inp.hamTerms):

                newCoeff = termA[0] * termB[0]
                if termA[1] == "ident":
                    newOpString = termB[1]
                elif termB[1] == "ident":
                    newOpString = termA[1]
                else:
                    newOpString = termA[1] + termB[1]
                newOp.addHamTerm(newCoeff, newOpString)

        # Return
        return newOp


class compositeQasmBuilder:
    """Simple class for outputting Trotterized QASM circuits."""

    def __init__(self):

        # List of commands. Just HAM and ENC
        self.circCommands = []

        self.pointerToCompositeSys = None

    def addHamTerm(self, coeff, opString):
        """Add term to the Hamiltonian"""
        # coeff is a number
        # opString is a list of tuples [ (ssid,) , (,) ]

        cmd = ["HAM"]
        cmd.append(coeff)
        cmd.append(deepcopy(opString))
        self.circCommands.append(cmd)

    def addBreak(self):
        self.circCommands.append(
            [
                "BREAK",
            ]
        )

    def addComment(self, cmt):
        self.circCommands.append(["COMMENT", cmt])

    def addEncodingConversion(self, ssid, encFrom, encTo, ssidTo=None):
        """Not implemented"""
        pass

    def hamToPauli(self, compositeSys):

        assert isinstance(compositeSys, compositeDLevels)

        pauliHam = QubitOperator()  # 0.0

        for cmdId, cmd in enumerate(self.circCommands):

            cmdType = cmd[0]
            coeff = cmd[1]
            opString = cmd[2]

            if cmdType == "HAM":  # Ignore all others

                pauliHam += compositeSys.opStringToPauli(coeff, opString)

        return pauliHam

    def yieldPauliOpsIncludeBreaks(self, compositeSys):

        # Include breaks means that you'll stop at ENC commands
        # (and possibly other future break-type commands)

        pauliOp = QubitOperator.zero()  # 0.0

        assert isinstance(compositeSys, compositeDLevels)

        for cmdId, cmd in enumerate(self.circCommands):

            cmdType = cmd[0]

            if cmdType == "HAM":
                coeff = cmd[1]
                opString = cmd[2]

                pauliOp += compositeSys.opStringToPauli(coeff, opString)

            elif cmdType == "BREAK":

                if pauliOp != QubitOperator.zero():
                    yield pauliOp
                    pauliOp = QubitOperator.zero()

            elif cmdType == "ENC":
                raise Exception("ENC not yet implemented.")
            else:
                raise Exception("Command type " + cmdType + " not recognized.")

        yield pauliOp

    def yieldPauliOpsEachTerm(self, compositeSys):
        pass

    def getPauliOpsIgnoreEnc(self, compositeSys):
        pass

    # Counts upper bound of #CNOTs
    def countCnotUBound_IncludeBreaks(self, compositeSys):

        nCnot = 0

        for pauliString in self.yieldPauliOpsIncludeBreaks(compositeSys):

            # assert(is_hermitian(pauliString))

            nCnot += countCNOTs_trot1_noopt(pauliString)

        return nCnot
