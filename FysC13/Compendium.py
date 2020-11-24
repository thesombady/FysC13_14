import numpy as np
import matplotlib.pyplot as plt
import os
from math import sin, cos, pi
"""
def GeneralPotential(r,a,b,n,m):
    if n<m:
        raise ValueError("Cant computer since n needs to be bigger than m")
    else:
        if isinstance(r, (np.ndarray, np.generic)):
            return a/(r**n) - b/(r**m)
potential = GeneralPotential(np.linspace(0.0001,3,100), 3, 18, 15, 9)
plt.plot(np.linspace(0.0001,3,100), potential)
plt.plot(0)
plt.ylim(-50,50)
plt.grid()
plt.show()
"""
"""
def SheerStress(x):
    return x**2

def gradient(x):
    div = 2*x
    return np.linspace(0, 3, 100)*div - SheerStress(x)

xlist = np.linspace(0, 3, 100)
ylist = SheerStress(xlist)
list1 = [0.5,1.8]
plt.plot(xlist, ylist, label = r"Function $x^2$")
for i in range(len(list1)):
    plt.plot(xlist, gradient(list1[i]), '--', label = f"Tangent at {list1[i]}")
plt.grid()
plt.ylim(-1,5)
plt.legend()
plt.show()
"""
plt.axvline(1.57, color = "black", linestyle = "--")
plt.axvline(3.14, color = "black", linestyle = "--")
xlist = np.linspace(0, 5, 1000)
ylist = [abs(sin(2 * i+ pi / 2)) for i in xlist]
plt.plot(xlist, ylist, label="Test")
plt.xlim(1.3, 3.5)
plt.show()
