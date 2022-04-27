'''qSymbOp is a parent class for symbolic operations in quantum mechanics'''

import numpy as np
import re
import sympy

from copy import copy,deepcopy

'''
Notes:

Purpose of this code is to be able to have *both* coefficients and subspaces
be symbolic.

Another main PURPOSE is that it is useful for MIXED types of DOF.

Between terms you are *required* to use '++'

TODO: Allow for arbitrary commutation relations.
'''



class qSymbOpError(Exception):
    pass


# Wrapper for the sympy library (So that sympy doesn't have to be imported to module)
# (In case you ever write your own code for symbolic scalar, or use different lib)
symbScalarFromStr = sympy.sympify


def is_symbolic(symbScalar):
    '''Returns true of scalar is symbolic.
    '''

    try:
        complex(symbScalar)
    except TypeError:
        return True

    return False


class qSymbOp(object):
    '''Parent class for symbolic quantum operators.
    '''

    '''
    Things to eventually add:
    * Should be able to have a child class that restricts many things,
        and those restrictions should be abstractly handled by parent.
    * This should include commutation relations
    * Want to eventually be able to have each particle belong to a different CLASS
        of particle. You'd be able to say that some are elec dofs, some are vibr dofs,
        etcs. You could easily have two diff types of bosonic operator too, btw.
    '''


    addOpsOnTheFly = True
    addSubSystemsOnTheFly = True  # If false, SS's have to be added explicitly beforehand
    # useOnlyBuiltInSSOps = False

    # builtInSSOps = ()  # Should not be changed by class nor children

    # Whether to use symbolic coefficients or not (sympy module)
    symbolicCoefficients = True

    # Assumption is that nothing commutes, not even between subspaces
    diffSsidsCommute = True  # False for e.g. fermions, True for e.g. bosons
    # locProductsCommute = False  # Rarely if even set to truewould be set to true
    
    # Local products for non-commuting mats, e.g. X0*Y0=-jZ0
    localProducts = {}



    def __init__(s,inpText=None,mult=None):

        # Dict of ordered tuples of pairs
        s.fullOperator = {}

        # Set of ssid's. ssid is a string.
        s.ssid_set = set()


        if inpText:  # (the operator equals zero if inpText=None)
            
            s.parseFullOpStr(inpText,mult)

        else:
            # Operator equals zero in this case
            pass


    def parseFullOpStr(s,inpText,mult):
        '''Parse operator string, such as '[X_1 Z_v2] ++ 2*k [Y_1 Z_v3]'
        '''

        # If no brackets in string, it should just be ''

        strOpTerms = [ elem.strip() for elem in inpText.split('++') ]

        for strOpTerm in strOpTerms:
            
            # If string started with '++', then first elem will be empty
            if strOpTerm=="":
                continue

            # Take coeff and opstr
            k_opstr = [ x.strip() for x in strOpTerm.split('[') ]
            assert len(k_opstr)==2
            k_opstr[1] = k_opstr[1][:-1] # Remove the ']'

            # Get the tuple key for the opStr
            opStringKey = s.parseOpStringStr(k_opstr[1])

            # Parse the coefficient
            coeff = k_opstr[0]
            if coeff=='':
                coeff = 1
            if s.symbolicCoefficients:
                coeff = symbScalarFromStr(coeff)
            else:
                assert coeff.isnumeric()   # DOESN'T HANDLE COMPLEX NUMBERS
                coeff = float(coeff)

            s.addSingleOpStringTerm(opStringKey,coeff)


    def addSingleOpStringTerm(s,opstring,k):
        '''Add single operator string term to current object.
        '''

        # Add into full operator
        if opstring in s.fullOperator.keys():
            # Key already in dict
            s.fullOperator[opstring] += k
            # Remove if now zero
            if s.fullOperator[opstring]==0.:
                del s.fullOperator[opstring]
        else:
            # Key not already in dict
            s.fullOperator[opstring] = k


    def parseOpStringStr(s,inpstr):
        '''Returns a tuple of pairs (each pair is a tuple)
        
        Example: ( ('1','X'), ('v2','Z'), )
        '''

        opStringTuple = ()

        # Parse through loc ops (terms like "X_1 Z_v2" )
        locop_arr = inpstr.split()
        for locop in locop_arr:

            # Operator and subsystem
            op_ss = locop.split('_')
            assert len(op_ss)==2
            op = op_ss[0]
            ssid = op_ss[1]

            # No effect if ssid already there
            s.ssid_set.add(ssid)

            # Add this ssid-op pair to the opstring tuple
            opStringTuple += ((ssid,op),)



        # You'll have to do something with sorting later on, based on a different input.
        # sorted() does this in both fields automatically. wonderful.
        # opStringKey = tuple(sorted(opStringTuple))
        opStringKey = opStringTuple

        # Case of identity
        if len(locop_arr)==0:
            opStringKey = ((),)

        # This is a tuple
        return opStringKey





    def getCoeff(s,opString):
        '''Get coefficient (scalar) before some opString
        '''
        
        if isinstance(opString,str):
            opStringKey = s.parseOpStringStr(opString)
        else:
            assert s.isValidOpTuple(opString)
            # opStringKey = tuple(sorted(opString)) # No sorting! Add commutation features later
            opStringKey = opString

        if opStringKey in s.fullOperator.keys():
            return s.fullOperator[opStringKey]
        else:
            raise IndexError("Operator string not a term in full operator.")



    def multTwoOpStrings(s,opString1,opString2):
        '''Multiply two operator strings.

        When commutation relations etc are added, this will become more complex;
        But for now, you're just concatenating two lists
        without any simplification at all.
        '''

        return opString1 + opString2


    def isValidOpTuple(s,opTuple):
        '''Return true if it is a valid operator tuple.'''
        if np.array(opTuple).shape[1]==2:
            return True
            
        return False
        
        # Should probably add more, like checking ssid's if setting is set to strict


    def addSsid(s,ssid):
        '''
        Manually add ssid. This might be used when you have more subsystems,
        but they don't appear in the expression.
        '''
        
        # No effect if ssid already there
        s.ssid_set.add(ssid)
    
    def orderTerms(s):
        '''
        Orders terms based on keys.
        '''
        
        s.fullOperator = dict( sorted(s.fullOperator.items()) )


    def simplifyQuadExponents(s):
        """
        Combine quad/bosonic exponents, e.g. [q2_1 q_0 q_1 q_0] --> [q3_1 q2_0]
        In above expression, q2==q squared; q3==q cubed.
        
        Operators inside ops_to_ignore are excluded.
        Effectively assumes bosonic commutation.
        Considers the following operator names: {'q','p','q{}','p{}','qhoPos','qhoMom',
              'Qsq','Psq'}. Outputs in terms of {'q{}' & 'p{}'}
        
        Remember that, on same particle, we assume that *nothing* commutes.
        i.e., p and q don't commute, but they also don't commute with any other ops on the same particle.
        
        At the moment, only works when diffSsidsCommute=True. The order *within* a given ssid is preserved,
        since commutation can no be assumed within a given index.

        Args:
            inpOp (qSymbOp) - Input Operator

        """

        if not s.diffSsidsCommute:
            raise Exception("This function implemented only for diffSsidsCommute=True.")

        # assert isinstance( inpOp , qSymbOp )


        q_ops = ['qhoPos','Qsq','q','q2','q3','q4','q5','q6']
        p_ops = ['qhoMom','Psq','p','p2']
        quadOps = q_ops + p_ops


        # New dict for operator, to override original in the end
        simplified_op = {}

        
        # ( ('1','X'), ('v2','Z'), )


        # Loop through each opstring
        for opstring,coeff in s.fullOperator.items():
        # s.fullOperator[opstring] = k

            # Within each opstring:
            # create dict of all the ssid's. dict of lists.
            # update the most recent item

            # Simple dict structure to build up simplified version
            # Since diffSsidsCommute=True, order outside of a given ssid doesn't matter
            newopstring_dict = dict.fromkeys(sorted(s.ssid_set))
            for ssid in newopstring_dict:
                newopstring_dict[ssid] = ['']

            # Identity
            if opstring==((),):
                simplified_op[opstring] = coeff
                continue


            for locop in opstring:
                ssid = locop[0]
                op   = locop[1]
                if op=='qhoPos': op = 'q'
                if op=='Qsq': op = 'q2'
                if op=='qhoMom': op = 'p'
                if op=='Psq': op = 'p2'

                if   (op in q_ops) and (newopstring_dict[ssid][-1] in q_ops):
                    pow_a = 1 if newopstring_dict[ssid][-1]=='q' else int(newopstring_dict[ssid][-1][1:])
                    pow_b = 1 if op=='q' else int(op[1:])
                    newopstring_dict[ssid][-1] = 'q'+str(pow_a+pow_b)


                elif (op in p_ops) and (newopstring_dict[ssid][-1] in p_ops):
                    pow_a = 1 if newopstring_dict[ssid][-1]=='p' else int(newopstring_dict[ssid][-1][1:])
                    pow_b = 1 if op=='p' else int(op[1:])
                    newopstring_dict[ssid][-1] = 'p'+str(pow_a+pow_b)

                else:
                    newopstring_dict[ssid].append(op)


            # Re-combine and put in simplified_op
            new_opstring = []
            for ssid in newopstring_dict:
                # Taking off first entry cuz it's just a blank ''
                new_opstring += [ (ssid,locop) for locop in newopstring_dict[ssid][1:] ]
            if tuple(new_opstring) in simplified_op.keys():
                simplified_op[tuple(new_opstring)] += coeff
            else:
                simplified_op[tuple(new_opstring)] = coeff


        # Replace original
        s.fullOperator = simplified_op

        # Return
        return



    def getSsids(s):
        '''Get subsystem id's.'''
        return copy(s.ssid_set)

    def getOpstringsContainingLocalOp(s,localOp):
        '''
        Should be able to enter it as string 'X_0' or 'X_0 Y_0' or a list of tuples

        Isn't this function kind of pointless? What would it ever be used for
        '''
        pass

    def scalar_subs(s,subsDict):
        '''Substitute symbolic coefficients with numerical values.
        '''

        for opTuple,k in s.fullOperator.items():

            s.fullOperator[opTuple] = k.subs(subsDict)



    def get_latex_string(s):
        '''Not yet implemented.'''

        # This is good, because the ssid's can be subscripts
        pass




    def __str__(s):
        """
        Returns printable output text string
        """
        
        outputList = []

        for opTuple,k in s.fullOperator.items():

            if is_symbolic(k):
                k_str = "({})".format(str(k))
            else:
                # k_str = "({})".format( str(float(k)) )
                k_str = "({})".format( str(complex(k)) )

            if opTuple==((),):
                line = k_str + " []"
            else:
                line = k_str + " ["
                line += ' '.join( [ "{}_{}".format(pair[1],pair[0]) for pair in opTuple ] )
                line += "]"

            outputList.append(line)

        return '\n++ '.join(outputList)


    def __neg__(s):

        newop = deepcopy(s)

        for opKey in newop.fullOperator.keys():
            newop.fullOperator[opKey] = - newop.fullOperator[opKey]

        return newop



    def __add__(s, inp):
        

        # Create new object
        newOp = qSymbOp()
        newOp.fullOperator = deepcopy(s.fullOperator)

        # Add second operator dict to first
        for inpOpKey in inp.fullOperator.keys():

            coeff = inp.fullOperator[inpOpKey]
            newOp.addSingleOpStringTerm(inpOpKey,coeff)


        # Combine the sets
        newOp.ssid_set = s.ssid_set.union(inp.ssid_set)


        # Will need to check that properties of both match, like 
        # the commuting properties.

        # Return
        return newOp


  #   def __iadd__(s, inp):
  #     pass
    

  #   def __sub__(s, inp):
  #     pass

  #   def __isub__(s, inp):
  #     pass


  #   def __repr__(s):
  #     return str(s)

    def __rmul__(s,inp):

        return s*inp


    def __mul__(s, inp):

        # First, case where inp is a scalar (including sympy val)
        if isinstance(inp,(int,float,complex,sympy.Basic)):
            newOp = deepcopy(s)
            for key in newOp.fullOperator.keys():
                newOp.fullOperator[key] *= inp
            return newOp


        # Otherwise, both assumed to be qSymbOp's:
        # Create new object
        newOp = qSymbOp()
        # newOp.fullOperator = deepcopy(s.fullOperator)

        # Combine the ss sets
        newOp.ssid_set = s.ssid_set.union(inp.ssid_set)


        # Loop over 's' operator
        for key_1,k_1 in s.fullOperator.items():

            # Loop over 'inp' operator
            for key_2,k_2 in inp.fullOperator.items():

                # Handle identity:
                if key_1==((),):
                    newkey = key_2
                elif key_2==((),):
                    newkey = key_1
                else:
                    newkey = s.multTwoOpStrings(key_1,key_2)

                newOp.addSingleOpStringTerm(newkey, k_1*k_2 )


        # Return
        return newOp



  #   def __imul__(s, inp):
  #     pass

  #   def __rmul__(s, inp):
  #       pass
    
  #   def __eq__(s, inp):
  #     pass




  #   def __truediv__(s, inp):
  #     raise TypeError("__truediv__ not yet supported.")

  #   def __div__(s, inp):
  #     raise TypeError("__div__ not yet supported.")

  #   def __itruediv__(s, inp):
  #     raise TypeError("__itruediv__ not yet supported.")

  #   def __idiv__(s, inp):
  #     raise TypeError("__idiv__ not yet supported.")

  #   def __pow__(s, inp):
        # raise TypeError("__pow__ not yet supported.")

























