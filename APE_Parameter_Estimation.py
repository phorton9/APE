from APE_Functions import *
import time
try:
    groups = int(input('Enter the number of groups: \n'))
except:
    print("Not an integer")

assert groups > 1, "Enter at least 2 groups"

rhoT = []
for i in range(groups):
    try:
        rho = float(input(f'Enter the alternative correlation for group {i+1}: \n'))
        rhoT.append(rho)
    except:
        print("Not a decimal")

rho0 = []
for i in range(groups):
    try:
        rho = float(input(f'Enter the null correlation for group {i+1}: \n'))
        rho0.append(rho)
    except:
        print("Not a decimal")

rhoT = np.array(rhoT)
rho0 = np.array(rho0)

rho_check = rhoT > rho0

assert sum(rho_check) == groups, "Alternative correlations must be larger than null correlations"
        
        
cost = []
for i in range(groups):
    try:
        c = float(input(f'Enter the sample cost for group {i+1}: \n'))
        cost.append(c)
    except:
        print("Not a decimal")
cost = np.array(cost)      
cost_check = cost > 0

ratio = np.max(cost)
cost /= ratio

assert sum(cost_check) == groups, "All costs must be positive"
 
try:
    alphaTarget = float(input(f'Enter the family wise type I error target (alpha): \n'))

except:
    print("Not a decimal")
    
assert alphaTarget < 0.5, "Alpha is too large"

try:
    betaTarget = float(input(f'Enter the family wise type II error target (beta): \n'))

except:
    print("Not a decimal")
    
assert betaTarget < 0.5, "Beta is too large"

alphas = np.array([0.1]*groups)
betas = np.array([0.1]*groups)
etaList = [0.00000001,0.00000001,0.00000001,0.00000001,0.00000001,0.00000001,0.00000001,0.00000001,0.0000000010,0.0000000001,0.0000000001,0.00000000001,0.000000000001,0.0000000000001,0.00000000000001,0.000000000000001]

penalties = np.logspace(-1,12,14)



for x,penalty in enumerate(penalties):
    iters = 0
    norms = 1
    mult = 1
    while norms > 10**(-11) and iters < 10000:
        grad = gradientL(cost, alphas, betas, rhoT, rho0, penalty, alphaTarget, betaTarget)
        newAB = np.append(alphas,betas) - (etaList[x])*grad
        alphas, betas = newAB[:groups], newAB[groups:]
        norms = np.linalg.norm(grad)
        oldAB = newAB
        iters += 1
        
    
##Print Results
total_cost = 0
for n in range(groups):
    vars()['n'+ str(n)] = np.ceil(number(alphas[n],betas[n],rhoT[n], rho0[n]))
    total_cost+= vars()['n'+ str(n)]*cost[n]
    
total_cost *= ratio
print('\n')
print('******RESULTS******')
print(f'Total cost: {total_cost}')

print(f'Alphas: {alphas}')
print(f'Betas: {betas}')

print(f'Global Alpha: {np.round(np.sum(newAB[:groups]),4)}')

print(f'Global Beta: {np.round(np.max(newAB[groups:]),4)}')
