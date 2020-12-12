import numpy as np
import matplotlib.pyplot as plt
import os, sys
import PhysicsNum as pn
from scipy.optimize import curve_fit

Voltage = np.array([3, 3.5, 4, 4.5, 5, 5.5, 6, 0.5, 1, 1.5, 2])
Vol_ovr = np.array([2.682, 2.772, 2.842, 2.905, 2.956, 2.99, 3.039, 0.365, 0.866, 1.372, 1.962])
Current = np.array([0, 0.02, 0.03, 0.05, 0.07, 0.09, 0.10, 0, 0, 0, 0])
Intnsty = np.array([0.03, 0.08, 0.13, 0.18, 0.21, 0.25, 0.29, 0, 0, 0, 0])
wavelength = 452 * 10 **(-9) #452 nm
"""
def exp(a, b, x):
	return a*np.exp(-x/b)-a
Expfit, Covarience = curve_fit(exp, Vol_ovr, Current)
print(Expfit)
"""
"""
Expfit, constants = pn.ExpRegression(Voltage, Current)
xlist = np.linspace(0, 3, 100)
ylist = Expfit(xlist)
print(constants)
""" #Works but is off.

exp = lambda x,a,b: a * np.exp(-x/b)
Expfit, Covarience = curve_fit(exp, Vol_ovr, Current)
xlist = np.linspace(0,3, 100)
ylist = exp(xlist, Expfit[0], Expfit[1])
plt.plot(xlist, ylist, '--', label = "Fit")
plt.plot(Vol_ovr, Current, '.', label = "Data")
plt.grid()
plt.legend()

plt.show()
