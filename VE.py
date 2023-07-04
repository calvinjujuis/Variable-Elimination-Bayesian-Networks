import numpy as np

######### restrict function
# Inputs: 
# factor -- multidimensional array (one dimension per variable in the domain)
# variable -- integer indicating the variable to be restricted
# value -- integer indicating the value to be assigned to variable
#
# Output:
# resulting_factor -- multidimensional array (the dimension corresponding to variable has been restricted to value)
#########
def restrict(factor,variable,value):
    shape = list(factor.shape)
    shape[variable] = 1
    resulting_factor = factor.take([value], axis=variable).reshape(shape)
    return resulting_factor

######### sumout function
# Inputs: 
# factor -- multidimensional array (one dimension per variable in the domain)
# variable -- integer indicating the variable to be summed out
#
# Output:
# resulting_factor -- multidimensional array (the dimension corresponding to variable has been summed out)
#########
def sumout(factor,variable):
    resulting_factor = np.sum(factor, variable, keepdims=True)
    return resulting_factor

######### multiply function
# Inputs: 
# factor1 -- multidimensional array (one dimension per variable in the domain)
# factor2 -- multidimensional array (one dimension per variable in the domain)
#
# Output:
# resulting_factor -- multidimensional array (elementwise product of the two factors)
#########
def multiply(factor1,factor2):
    b = np.broadcast(factor1, factor2)
    out = np.empty(b.shape)
    out.flat = [u * v for (u,v) in b]
    return out

######### normalize function
# Inputs: 
# factor -- multidimensional array (one dimension per variable in the domain)
#
# Output:
# resulting_factor -- multidimensional array (entries are normalized to sum up to 1)
#########
def normalize(factor):
    resulting_factor = factor / np.sum(factor)
    return resulting_factor


def variableExists(factor, var):
    return factor.shape[var] > 1

######### inference function
#Inputs: 
#factor_list -- list of factors (multidimensional arrays) that define the joint distribution of the domain
#query_variables -- list of variables (integers) for which we need to compute the conditional distribution
#ordered_list_of_hidden_variables -- list of variables (integers) that need to be eliminated according to thir order in the list
#evidence_list -- list of assignments where each assignment consists of a variable and a value assigned to it (e.g., [[var1,val1],[var2,val2]])
#
#Output:
#answer -- multidimensional array (conditional distribution P(query_variables|evidence_list))
#########
def inference(factor_list,query_variables,ordered_list_of_hidden_variables,evidence_list):
    for var, val in evidence_list:
        for i in range(len(factor_list)):
            if variableExists(factor_list[i], var):
                restrictedFactor = restrict(factor_list[i], var, val)
                print("Restrict:")
                print(np.squeeze(factor_list[i]))
                print("Restricted:")
                print(np.squeeze(restrictedFactor))
                factor_list[i] = restrictedFactor
    print("\n")
    step = 1
    for var in ordered_list_of_hidden_variables:
        factorWithVar = [factor for factor in factor_list if variableExists(factor, var)]
        if factorWithVar == []:
            continue
        removedFactors = factorWithVar
        factor_list = [factor for factor in factor_list if not variableExists(factor, var)]
        while len(factorWithVar) > 1:
            newFactor = multiply(factorWithVar[0], factorWithVar[1])
            factorWithVar = factorWithVar[2:]
            factorWithVar.append(newFactor)
        newFactor = sumout(factorWithVar[0], var)
        factor_list.append(newFactor)
        
        print("Step " + str(step) + ":")
        step += 1
        print("Remove:")
        for f in removedFactors:
            print(np.squeeze(f)) 
        print("Add:")
        print(np.squeeze(newFactor))
    while len(factor_list) > 1:
        newFactor = multiply(factor_list[0], factor_list[1])
        factor_list = factor_list[2:]
        factor_list.append(newFactor)
    return normalize(factor_list[0])

Acc = 0
Fraud = 1
Trav = 2
FP = 3
OP = 4
PT = 5

# f(Fraud | Trav)
f1 = np.array([[0.996, 0.99],[0.004,0.01]])
f1 = f1.reshape(1, 2, 2, 1, 1, 1)

# f(Trav)
f2 = np.array([0.95, 0.05])
f2 = f2.reshape(1, 1, 2, 1, 1, 1)

# f(FP | Fraud, Trav)
f3 = np.array([[[0.99, 0.01], [0.1, 0.9]], [[0.9, 0.1], [0.1, 0.9]]])
f3 = f3.reshape(1, 2, 2, 2, 1, 1)

# f(Acc)
f4 = np.array([0.2, 0.8])
f4 = f4.reshape(2, 1, 1, 1, 1, 1)

# f(OP | Acc, Fraud)
f5 = np.array([[[0.9, 0.1], [0.7, 0.3]], [[0.4, 0.6], [0.2, 0.8]]])
f5 = f5.reshape(2, 2, 1, 1, 2, 1)

# f(PT | Acc)
f6 = np.array([[0.99, 0.01],[0.9, 0.1]])
f6 = f6.reshape(2, 1, 1, 1, 1, 2)

# VE inference
f7 = inference([f1,f2,f3,f4,f5,f6],[Fraud],[Trav, FP, OP, Acc, PT],[[OP,1], [FP, 0]])
print(f"P(Fraud | OP = 1, FP = 0)={np.squeeze(f7)}\n")


