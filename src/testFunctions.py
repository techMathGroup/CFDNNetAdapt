
### definition of testing functions
# included functions:
#   ZDT1, ZDT2, ZDT3, ZDT4, ZDT6
#   CF1, CF2, CF3
#   F1, F2, F3, F4, F5, F6, F7, F8, F9 (F6 not yet)

import math
import numpy as np
from scipy.optimize import root_scalar

## ZDT1
def ZDT1(x):
    n = len(x)
    f1 = x[0]
    g = 1 + 9*sum(x[1:])/(n - 1)
    f2 = g*(1 - math.sqrt(x[0]/g))

    return [f1, f2]

def optSolsZDT1(n, nPars):
    x1s = np.linspace(0.0, 1.0, n)
    optSols = list()
    for i in range(n):
        pars = [x1s[i]] + [0]*(nPars - 1)
        optSols.append(pars)

    return optSols

## ZDT2
def ZDT2(x):
    n = len(x)
    f1 = x[0]
    g = 1 + 9*sum(x[1:])/(n - 1)
    f2 = g*(1 - (x[0]/g)**2)

    return [f1, f2]

def optSolsZDT2(n, nPars):
    x1s = np.linspace(0.0, 1.0, n)
    optSols = list()
    for i in range(n):
        pars = [x1s[i]] + [0]*(nPars - 1)
        optSols.append(pars)

    return optSols

# ZDT3
def ZDT3(x):
    n = len(x)
    f1 = x[0]
    g = 1 + 9*sum(x[1:])/(n - 1)
    f2 = g*(1 - math.sqrt(x[0]/g) - x[0]/g*math.sin(10*math.pi*x[0]))

    return [f1, f2]

# alias optimal curve
def optCurveZDT3(x0):
    f = 1 - math.sqrt(x0) - x0*math.sin(10*math.pi*x0)
    return f

# optimal curve derivative
def deroptZDT3(x0):
    diff = -1/(2*math.sqrt(x0)) - math.sin(10*math.pi*x0) - 10*math.pi*x0*math.cos(10*math.pi*x0)
    return diff

# real optimal solutions
def optSolsZDT3(n, nPars):
    x1s = np.linspace(0.0, 1.0, n)
    badSols = list()
    for i in range(n):
        pars = [x1s[i]] + [0]*(nPars - 1)
        badSols.append(pars)
    
    badf1s = list()
    badf2s = list()
    for i in range(len(badSols)):
        f1, f2 = ZDT3(badSols[i])
        badf1s.append(f1)
        badf2s.append(f2)
    
    fs = [optCurveZDT3(i) for i in x1s]
    
    # find roots
    mins = [0.07, 0.24, 0.44, 0.64, 0.84]
    maxs = [ 0.1, 0.27, 0.46, 0.66, 0.86]
    sols = [0.0]
    for i in range(len(mins)):
        solution = root_scalar(deroptZDT3, x0 = mins[i], x1 = maxs[i])
        sols.append(solution.root)
    sols.append(1.0)
    fds = [optCurveZDT3(i) for i in sols]
    
    # filter optimal solutions
    optSols = list()
    for i in range(n):
        x1 = x1s[i]
        f2 = badf2s[i]
        for j in range(1,len(sols)):
            left = sols[j-1]
            right = sols[j]
            top = fds[j-1]
            bottom = fds[j]
    
            if x1 >= left and x1 < right:
                if f2 <= top and f2 > bottom:
                    optSols.append(badSols[i])
                else:
                    continue

    return optSols

## ZDT4
def ZDT4(x):
    n = len(x)
    f1 = x[0]
    suma = 0
    for i in range(1,n):
        suma += x[i]**2 - 10*math.cos(4*math.pi*x[i])
    g = 1 + 10*(n - 1) + suma
    f2 = g*(1 - math.sqrt(x[0]/g))

    return [f1, f2]

def optSolsZDT4(n, nPars):
    x1s = np.linspace(0.0, 1.0, n)
    optSols = list()
    for i in range(n):
        pars = [x1s[i]] + [0]*(nPars - 1)
        optSols.append(pars)

    return optSols

## ZDT6
def ZDT6(x):
    n = len(x)
    f1 = 1 - math.exp(-4*x[0])*(math.sin(6*math.pi*x[0]))**6
    g = 1 + 9*(sum(x[1:])/(n - 1))**0.25
    f2 = g*(1 - (f1/g)**2)

    return [f1, f2]

def optSolsZDT6(n, nPars):
    x1s = np.linspace(0.0, 1.0, n)
    optSols = list()
    for i in range(n):
        pars = [x1s[i]] + [0]*(nPars - 1)
        optSols.append(pars)

    return optSols

## CF1
def CF1(x):
    # pars
    a = 1
    N = 10
    penalty = 1e3

    n = len(x)
    g1 = 0
    g2 = 0
    j1 = 0 # at least one is expected
    j2 = 0 # at least one is expected
    for i in range(1,n):
        if (i+1)%2==0: # even
            g2 += (x[i] - x[0]**(0.5*(1.0 + 3*(i+1 - 2)/(n-2))))**2
            j2 += 1

        else: # odd
            g1 += (x[i] - x[0]**(0.5*(1.0 + 3*(i+1 - 2)/(n-2))))**2
            j1 += 1

    g1 *= 2/j1
    g2 *= 2/j2

    f1 = x[0] + g1
    f2 = 1 - x[0] + g2

    # constrained
    c = f1 + f2 - a*abs(math.sin(N*math.pi*(f1 - f2 + 1))) - 1

    if c < -1e-10:
        f1 = penalty
        f2 = penalty

    return [f1, f2]

def optSolsCF1(n, nPars):
    # pars
    a = 1
    N = 10
    penalty = 1e3

    # optimal solution
    x1s = [i/2/N for i in range(0,2*N)]
    optSols = list()
    for i in range(len(x1s)):
        pars = [x1s[i]]
        for j in range(1,nPars):
            pars.append(x1s[i]**(0.5*(1.0 + 3*(j+1 - 2)/(nPars-2))))

        # evaluate
        f1, f2 = CF1(pars)
        if f1 < penalty and f2 < penalty:
            optSols.append(pars)

    return optSols

## CF2
def CF2(x):
    # pars
    a = 1
    N = 10
    penalty = 1e3

    n = len(x)
    g1 = 0
    g2 = 0
    j1 = 0 # at least one is expected
    j2 = 0 # at least one is expected
    for i in range(1,n):
        if (i+1)%2==0: # even
            g2 += (x[i] - math.sin(6*math.pi*x[0] + (i+1)*math.pi/n))**2
            j2 += 1

        else: # odd
            g1 += (x[i] - math.sin(6*math.pi*x[0] + (i+1)*math.pi/n))**2
            j1 += 1

    g1 *= 2/j1
    g2 *= 2/j2

    f1 = x[0] + g1
    f2 = 1 - math.sqrt(x[0]) + g2

    # constrained
    t = f2 + math.sqrt(f1) - a*math.sin(N*math.pi*(math.sqrt(f1) - f2 + 1)) - 1

    c = t/(1 + math.exp(4*abs(t)))

    if c < -1e-10:
        f1 = penalty
        f2 = penalty

    return [f1, f2]

def optSolsCF2(n, nPars):
    # pars
    a = 1
    N = 10
    penalty = 1e3

    # optimal solution
    x1s = np.linspace(0,1,n)
    optSols = list()
    for i in range(len(x1s)):
        pars = [x1s[i]]
        for j in range(1,nPars):
            pars.append(math.sin(6*math.pi*x1s[i] + (j+1)*math.pi/nPars))
    
        # evaluate
        f1, f2 = CF2(pars)
        if f1 < penalty and f2 < penalty:
            optSols.append(pars)

    return optSols

## F1
def F1(x):
    n = len(x)
    g1 = 0
    g2 = 0
    j1 = 0 # at least one is expected
    j2 = 0 # at least one is expected
    for i in range(1,n):
        if (i+1)%2==0: # even
            g2 += (x[i] - x[0]**(0.5*(1.0 + 3*(i+1 - 2)/(n-2))))**2
            j2 += 1

        else: # odd
            g1 += (x[i] - x[0]**(0.5*(1.0 + 3*(i+1 - 2)/(n-2))))**2
            j1 += 1

    g1 *= 2/j1
    g2 *= 2/j2

    f1 = x[0] + g1
    f2 = 1 - math.sqrt(x[0]) + g2

    return [f1, f2]

def optSolsF1(n, nPars):
    # optimal solution
    x1s = np.linspace(0,1,n)
    optSols = list()
    for i in range(len(x1s)):
        pars = [x1s[i]]
        for j in range(1,nPars):
            pars.append(x1s[i]**(0.5*(1.0 + 3*(j+1 - 2)/(nPars-2))))
        optSols.append(pars)

    return optSols

## F2
def F2(x):
    n = len(x)
    g1 = 0
    g2 = 0
    j1 = 0 # at least one is expected
    j2 = 0 # at least one is expected
    for i in range(1,n):
        if (i+1)%2==0: # even
            g2 += (x[i] - math.sin(6*math.pi*x[0] + (i+1)*math.pi/n))**2
            j2 += 1

        else: # odd
            g1 += (x[i] - math.sin(6*math.pi*x[0] + (i+1)*math.pi/n))**2
            j1 += 1

    g1 *= 2/j1
    g2 *= 2/j2

    f1 = x[0] + g1
    f2 = 1 - math.sqrt(x[0]) + g2

    return [f1, f2]

def optSolsF2(n, nPars):
    # optimal solution
    x1s = np.linspace(0,1,n)
    optSols = list()
    for i in range(len(x1s)):
        pars = [x1s[i]]
        for j in range(1,nPars):
            pars.append(math.sin(6*math.pi*x1s[i] + (j+1)*math.pi/nPars))
        optSols.append(pars)

    return optSols

## F3
def F3(x):
    n = len(x)
    g1 = 0
    g2 = 0
    j1 = 0 # at least one is expected
    j2 = 0 # at least one is expected
    for i in range(1,n):
        if (i+1)%2==0: # even
            g2 += (x[i] - 0.8*x[0]*math.sin(6*math.pi*x[0] + (i+1)*math.pi/n))**2
            j2 += 1

        else: # odd
            g1 += (x[i] - 0.8*x[0]*math.cos(6*math.pi*x[0] + (i+1)*math.pi/n))**2
            j1 += 1

    g1 *= 2/j1
    g2 *= 2/j2

    f1 = x[0] + g1
    f2 = 1 - math.sqrt(x[0]) + g2

    return [f1,f2]

def optSolsF3(n, nPars):
    # optimal solution
    x1s = np.linspace(0,1,100)
    optSols = list()
    for i in range(len(x1s)):
        pars = [x1s[i]]
    
        for j in range(1,nPars):
            if (j+1)%2==0: # even
                pars.append(0.8*x1s[i]*math.sin(6*math.pi*x1s[i] + (j+1)*math.pi/nPars))
            else: # odd
                pars.append(0.8*x1s[i]*math.cos(6*math.pi*x1s[i] + (j+1)*math.pi/nPars))
        optSols.append(pars)

    return optSols

## F4
def F4(x):
    n = len(x)
    g1 = 0
    g2 = 0
    j1 = 0 # at least one is expected
    j2 = 0 # at least one is expected
    for i in range(1,n):
        if (i+1)%2==0: # even
            g2 += (x[i] - 0.8*x[0]*math.sin(6*math.pi*x[0] + (i+1)*math.pi/n))**2
            j2 += 1

        else: # odd
            g1 += (x[i] - 0.8*x[0]*math.cos(6*math.pi*x[0]/3 + (i+1)*math.pi/n/3))**2
            j1 += 1

    g1 *= 2/j1
    g2 *= 2/j2

    f1 = x[0] + g1
    f2 = 1 - math.sqrt(x[0]) + g2

    return [f1,f2]

def optSolsF4(n, nPars):
    # optimal solution
    x1s = np.linspace(0,1,100)
    optSols = list()
    for i in range(len(x1s)):
        pars = [x1s[i]]
    
        for j in range(1,nPars):
            if (j+1)%2==0: # even
                pars.append(0.8*x1s[i]*math.sin(6*math.pi*x1s[i] + (j+1)*math.pi/nPars))
            else: # odd
                pars.append(0.8*x1s[i]*math.cos(6*math.pi*x1s[i]/3 + (j+1)*math.pi/nPars/3))
        optSols.append(pars)

    return optSols

## F5
def F5(x):
    n = len(x)
    g1 = 0
    g2 = 0
    j1 = 0 # at least one is expected
    j2 = 0 # at least one is expected
    for i in range(1,n):
        if (i+1)%2==0: # even
            g2 += (x[i] - (0.3*x[0]**2*math.cos(24*math.pi*x[0] + 4*(i+1)*math.pi/n) + 0.6*x[0])*math.sin(6*math.pi*x[0] + (i+1)*math.pi/n))**2
            j2 += 1

        else: # odd
            g1 += (x[i] - (0.3*x[0]**2*math.cos(24*math.pi*x[0] + 4*(i+1)*math.pi/n) + 0.6*x[0])*math.cos(6*math.pi*x[0] + (i+1)*math.pi/n))**2
            j1 += 1

    g1 *= 2/j1
    g2 *= 2/j2

    f1 = x[0] + g1
    f2 = 1 - math.sqrt(x[0]) + g2

    return [f1,f2]

def optSolsF5(n, nPars):
    # optimal solution
    x1s = np.linspace(0,1,100)
    optSols = list()
    for i in range(len(x1s)):
        pars = [x1s[i]]
    
        for j in range(1,nPars):
            if (j+1)%2==0: # even
                pars.append((0.3*x1s[i]**2*math.cos(24*math.pi*x1s[i] + 4*(j+1)*math.pi/nPars) + 0.6*x1s[i])*math.sin(6*math.pi*x1s[i] + (j+1)*math.pi/nPars))
            else: # odd
                pars.append((0.3*x1s[i]**2*math.cos(24*math.pi*x1s[i] + 4*(j+1)*math.pi/nPars) + 0.6*x1s[i])*math.cos(6*math.pi*x1s[i] + (j+1)*math.pi/nPars))
        optSols.append(pars)
    
    return optSols

## F6

## F7
def F7(x):
    n = len(x)
    g1 = 0
    g2 = 0
    j1 = 0 # at least one is expected
    j2 = 0 # at least one is expected
    for i in range(1,n):
        yi = x[i] - x[0]**(0.5*(1.0 + 3*(i+1 - 2)/(n - 2)))
        if (i+1)%2==0: # even
            g2 += 4*yi**2 - math.cos(8*yi*math.pi) + 1.0
            j2 += 1

        else: # odd
            g1 += 4*yi**2 - math.cos(8*yi*math.pi) + 1.0
            j1 += 1

    g1 *= 2/j1
    g2 *= 2/j2

    f1 = x[0] + g1
    f2 = 1 - math.sqrt(x[0]) + g2

    return [f1,f2]

def optSolsF7(n, nPars):
    # optimal solution
    x1s = np.linspace(0,1,100)
    optSols = list()
    for i in range(len(x1s)):
        pars = [x1s[i]]
    
        for j in range(1,nPars):
            if (j+1)%2==0: # even
                pars.append(x1s[i]**(0.5*(1.0 + 3*(j+1 - 2)/(nPars - 2))))
            else: # odd
                pars.append(x1s[i]**(0.5*(1.0 + 3*(j+1 - 2)/(nPars - 2))))
        optSols.append(pars)

    return optSols

## F8
def F8(x):
    n = len(x)
    g1 = 0
    g2 = 0
    sum1 = 0
    sum2 = 0
    pi1 = 1
    pi2 = 1
    j1 = 0 # at least one is expected
    j2 = 0 # at least one is expected
    for i in range(1,n):
        yi = x[i] - x[0]**(0.5*(1.0 + 3*(i+1 - 2)/(n - 2)))
        if (i+1)%2==0: # even
            sum2 += yi**2
            pi2 *= math.cos(20*yi*math.pi/math.sqrt(i+1))
            j2 += 1

        else: # odd
            sum1 += yi**2
            pi1 *= math.cos(20*yi*math.pi/math.sqrt(i+1))
            j1 += 1

    g1 = 4*sum1 - 2*pi1 + 2
    g2 = 4*sum2 - 2*pi2 + 2

    g1 *= 2/j1
    g2 *= 2/j2

    f1 = x[0] + g1
    f2 = 1 - math.sqrt(x[0]) + g2

    return [f1,f2]

def optSolsF8(n, nPars):
    # optimal solution
    x1s = np.linspace(0,1,100)
    optSols = list()
    for i in range(len(x1s)):
        pars = [x1s[i]]
    
        for j in range(1,nPars):
            if (j+1)%2==0: # even
                pars.append(x1s[i]**(0.5*(1.0 + 3*(j+1 - 2)/(nPars - 2))))
            else: # odd
                pars.append(x1s[i]**(0.5*(1.0 + 3*(j+1 - 2)/(nPars - 2))))
        optSols.append(pars)

    return optSols

## F9
def F9(x):
    n = len(x)
    g1 = 0
    g2 = 0
    j1 = 0 # at least one is expected
    j2 = 0 # at least one is expected
    for i in range(1,n):
        if (i+1)%2==0: # even
            g2 += (x[i] - math.sin(6*math.pi*x[0] + (i+1)*math.pi/n))**2
            j2 += 1

        else: # odd
            g1 += (x[i] - math.sin(6*math.pi*x[0] + (i+1)*math.pi/n))**2
            j1 += 1

    g1 *= 2/j1
    g2 *= 2/j2

    f1 = x[0] + g1
    f2 = 1 - x[0]**2 + g2

    return [f1,f2]

def optSolsF9(n, nPars):
    # optimal solution
    x1s = np.linspace(0,1,100)
    optSols = list()
    for i in range(len(x1s)):
        pars = [x1s[i]]
    
        for j in range(1,nPars):
            if (j+1)%2==0: # even
                pars.append(math.sin(6*math.pi*x1s[i] + (j+1)*math.pi/nPars))
            else: # odd
                pars.append(math.sin(6*math.pi*x1s[i] + (j+1)*math.pi/nPars))
        optSols.append(pars)

    return optSols
