import numpy as np
import matplotlib.pyplot as plt
import os
from math import sin, cos, pi, floor
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
"""
plt.axvline(-pi/2, color = "black", linestyle = "--")
plt.axvline(pi/2, color = "black", linestyle = "--")
plt.axvline(-pi, color = "red", linestyle = "--")
plt.axvline(pi, color = "red", linestyle = "--")
xlist = np.linspace(-5, 5, 1000)
ylist = [abs(sin(i)) for i in xlist]
ylist2 = [(1/10*cos(i) + 2) for i in xlist]
plt.plot(xlist, ylist, label="Test")
plt.plot(xlist, ylist2, label = "Test2")
plt.xlim(-4*pi/3, 4*pi/3)
plt.show()
"""''

plt.axvline(-pi/2, color = "black", linestyle = "--")
plt.axvline(+pi/2, color = "black", linestyle = "--")
xlist = np.linspace(-3/2*pi, 3/2*pi, 1000)
ylist = [abs(sin(xlist[i])) for i in range(len(xlist))]
plt.plot(xlist, ylist, label = "The standing wave", linestyle = "-")
xlist2 = np.linspace(-3/2*pi, 3/2*pi, floor(16*pi/2))
ylist2 = [abs(sin(xlist2[i])) for i in range(len(xlist2))]
plt.plot(xlist2, ylist2, '.', label = r"Quantized $k$", color = "red")
plt.legend()
plt.show()
print("Hello")
