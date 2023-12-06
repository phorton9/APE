from scipy.optimize import minimize, rosen
import numpy as np
from scipy.special import ndtri
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

def number(alpha,beta,rhoT,rho0):

    '''
    Args:
        alpha(Float): Value in (0,1) that corresponds to the type I error. 
        beta(Float): Value in (0,1) that corresponds to the type Ii error.
        rhoT(Float): Test correlation value between [0,1]
        rho0(Float): /Null hypothesis value between [0,1]
    Returns:
        Float: Positive scalar value.
    '''
    atanT = 0.5*np.log((1+rhoT)/(1-rhoT))
    atan0 = 0.5*np.log((1+rho0)/(1-rho0))
    
    za = np.abs(ndtri(alpha))
    zb = np.abs(ndtri(beta))
    return ((za + zb)/(atanT - atan0))**2 + 3
    
def relaxed(X, const, penalty, alpha0, beta0):
    '''
    Args:
        X(np.array): Array of initial starting points for alpha and beta. Length is two times the number of groups. 
        const(np.array): Array of parameters that is (n,3) where n is the number of groups. Each subarray should consist of np.array([cost(float), rhoT(float), rho0(float)])
        penalty(Float): Penalty parameter (lambda) for constrained optimization function. 
        alpha0 (Float): Global type I error rate between (0,1)
        beta0(Float): Global type II error rate between (0,1)
    Returns:
        Float: Positive value for penalized optimization function.
    ''' 

    total = 0

    numberAll = 0
    for i in range(len(const)):
        cost = const[i][0]

        alpha = X[i*2]
        beta = X[i*2+1]
        rhoT = const[i][1]
        rho0 = const[i][2]
        vars()["N"+str(i)] = number(alpha,beta,rhoT,rho0)
        numberAll += vars()["N"+str(i)]

    
    for i in range(len(const)):
        cost = const[i][0]
        alpha = X[i*2] 
        beta = X[i*2+1]
        rhoT = const[i][1]
        rho0 = const[i][2]
        N = number(alpha,beta,rhoT,rho0)
        
        
        
        
        total += cost * N
        penalized = 0
        rhoOverall = 0
        betaOverall = np.zeros((len(const)))
        alphaOverall = 0
        p1 = 0
        p2 = 0
        p3 = 0
        p4 = 0
        for j in range(len(const)):
            rhoOverall += vars()["N"+str(j)] * const[j][2] / numberAll            

            betaOverall[j] = (X[j*2+1])
            alphaOverall += (X[j*2])
            
        p1 += min(0, alpha)**2
        p2 += min(0, beta)**2
        p3 += min(0, 1-alpha)**2
        p4 += min(0, 1 - beta)**2
        
        betaOverall = np.max(betaOverall)   
    p6 = max(0, (betaOverall - beta0))**2
    p7 = max(0, alphaOverall - alpha0)**2
    
    penalized = (penalty)*(p1 + p2 + p3 + p4 + p6 + p7)
        
    total += penalized
    
        
    return total
    
def dZtrue(eRate):
    '''
    Args:
        eRate(float): Value in (0,1) that corresponds to an error rate. 

    Returns:
        Float: Positive scalar value.
    ''' 
    
    z=ndtri(eRate)
    
    return np.sqrt(2*np.pi)*np.exp(z**2/2)
    
def dN(alphas, betas, rhoT, rho0, respect, i):
    '''
    Args:
        alphas(np.array): (n,) shape that contains the type I error rates for groups i=1,2,...n
        betas(np.array): (n,) shape that contains the type II error rates for groups i=1,2,...n
        rhoT(np.array): (n,) shape that contains the alternative hypothesis correlation for each group
        rho0(np.array): (n,) shape that contains the null hypothesis correlation for each group
        respect(string): designation for which variable to use for partial derivative - alpha or beta
        i (integer): Index corresponding to the subgroup.
    Returns:
        Float: Derivative of the number of samples with respect to alpha_i or beta_i
    ''' 
    
    if respect == 'alpha':
    
        dz = -dZtrue(1-alphas[i])
    else:
        dz = -dZtrue(1-betas[i])
    
    
    atanT = 0.5*np.log((1+rhoT[i])/(1-rhoT[i]))
    atan0 = 0.5*np.log((1+rho0[i])/(1-rho0[i]))
    
    
    za = np.abs(ndtri(alphas[i]))
    zb = np.abs(ndtri(betas[i]))

    
    
    return 2 * (za + zb)/(atanT - atan0)**2*dz
    
    
def dRho(alphas, betas, rhoT, rho0, respect, i):
    '''
    Args:
        alphas(np.array): (n,) shape that contains the type I error rates for groups i=1,2,...n
        betas(np.array): (n,) shape that contains the type II error rates for groups i=1,2,...n
        rhoT(np.array): (n,) shape that contains the alternative hypothesis correlation for each group
        rho0(np.array): (n,) shape that contains the null hypothesis correlation for each group
        respect(string): designation for which variable to use for partial derivative - alpha or beta
        i (integer): Index corresponding to the subgroup.
    Returns:
        Float: Derivative of the correlation with respect to alpha_i or beta_i
    ''' 
    sumN = 0
    sumNrho = 0
    for x in range(len(alphas)):
        sumN+= number(alphas[x], betas[x], rhoT[x], rho0[x])
        sumNrho+= number(alphas[x], betas[x], rhoT[x], rho0[x])*rho0[x]

    dNi = dN(alphas, betas, rhoT, rho0, respect, i)


    return dNi*(sumN*rho0[i] - sumNrho)/(sumN**2)
    
def dAB(alphas, betas, rhoT, rho0, respect, i):
    '''
    Args:
        alphas(np.array): (n,) shape that contains the type I error rates for groups i=1,2,...n
        betas(np.array): (n,) shape that contains the type II error rates for groups i=1,2,...n
        rhoT(np.array): (n,) shape that contains the alternative hypothesis correlation for each group
        rho0(np.array): (n,) shape that contains the null hypothesis correlation for each group
        respect(string): designation for which variable to use for partial derivative - alpha or beta
        i (integer): Index corresponding to the subgroup.
    Returns:
        Float: Derivative of the global error rates with respect to alpha_i or beta_i 
    ''' 
    
    if i >= len(alphas):
        return 'Error in length'
    
    output = 0
    if respect == 'alpha':
        for x, alpha in enumerate(alphas):
            if i == x:
                output =1
    else:
        if betas[i] == np.max(betas):
            output = 1
                
    return output
    
def gradientL(cost, alphas, betas, rhoT, rho0, penalty, alphaTarget, betaTarget):
    '''
    Args:
        cost(np.array): (n,) shape that contains the unit costs for groups i=1,2,...n
        alphas(np.array): (n,) shape that contains the type I error rates for groups i=1,2,...n
        betas(np.array): (n,) shape that contains the type II error rates for groups i=1,2,...n
        rhoT(np.array): (n,) shape that contains the alternative hypothesis correlation for each group
        rho0(np.array): (n,) shape that contains the null hypothesis correlation for each group
        penalty(string): designation for which variable to use for partial derivative - alpha or beta
        alphaTarget(float): global type I error rate
        betaTarget(float): global type II error rate
    Returns:
        np.array: (2n,) array of gradients used for gradient descent
    ''' 
    gradient = []
    numbers = []
    alphaOverall = 0
    betaOverall = np.zeros((len(alphas)))

    for x in range(len(alphas)):
        tempN = number(alphas[x], betas[x], rhoT[x], rho0[x])
        numbers.append(tempN)
        alphaOverall += alphas[x]
        betaOverall[x] = (betas[x])
    
    betaOverall = np.max(betaOverall)
    rhoOverall = np.array(numbers)@np.array(rho0)/np.sum(numbers)
    t1 = 'Overall'
    s='s'
    for term in ['alpha', 'beta']:
        for x in range(len(alphas)):
            p1 = dN(alphas, betas, rhoT, rho0, term, x) * cost[x]
            if term == 'alpha':
                p3= 2*penalty*max(alphaOverall - alphaTarget,0)*dAB(alphas, betas, rhoT, rho0, term, x)
                p4 = 2*penalty*min(alphas[x],0)
                p5 = 2*penalty*min(1-alphas[x],0)

            else:
                p3 = 2*penalty*max(betaOverall - betaTarget,0)*dAB(alphas, betas, rhoT, rho0, term, x)
                p4 = 2*penalty*min(betas[x],0)
                p5 = 2*penalty*min(1-betas[x],0)
                

            temp = p1+p3+p4+p5
            gradient.append(temp)

    return np.array(gradient)

def atanh(correlation):
    """Takes Fisher Z-transformation of a correlation
    correlation(float): pearson's correlation"""
    return 0.5*np.log((1+correlation)/(1-correlation))

def calc_R(c1,c2,Za, Zb):
    """Takes the combination of test coefficients and returns the ___ ratio
    c1(float): 
    c2(float)
    Za(float): Z-score corresponding to probability of type I error
    Zb(float): Z-score corresponding to probability of type II error"""
    return (c1 + c2)**2/(Za + Zb)**2

def calc_n(R, If, K):
    """Calculates the number of samples per group as a fixed proportion of the max samples
    R(float): 
    If(float): effect size for test
    K(int): number of iterations (looks) for the sequential group test"""
    return int(np.ceil(R*If/K))

def corLLR(data, covAlt, covH0):
    """Calculates the log-likelihood ratio for the bivariate normal data used for correlation analysis
    data(np.array): data generated from bivariate normal distribution
    covAlt(np.array(2x2)): covariance matrix under the alternative hypothesis
    covH0(np.array(2x2)): covariance matrix under the null hypothesis"""
    length = len(data) ##number of points
    term1 = np.sum(-1/2*(np.sum(data@np.linalg.inv(covAlt)*data,axis = 1) - np.sum(data@np.linalg.inv(covH0)*data,axis = 1)))
    term2 = -(length)/2*(np.log(np.linalg.det(covAlt)) - np.log(np.linalg.det(covH0)))
    LLR = term1+term2

    return LLR

def upperThreshold(atanT, atan0, Ik, j, k, c2, delta):
    """Returns the threshold used for the decision to reject H0 in group sequential"""
    return (atanT - atan0)*np.sqrt(Ik) - c2*(j/k)**(delta-1/2)

def lowerThreshold(c1, j, k, delta):
    """Returns the threshold used for the decision to accept H0 in group sequential"""
    return c1*(j/k)**(delta-1/2)