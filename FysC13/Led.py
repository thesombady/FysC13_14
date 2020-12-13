import numpy as np
import matplotlib.pyplot as plt
import os, sys
import PhysicsNum as pn
from scipy.optimize import curve_fit


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
def task1_1():
	Voltage = np.array([3, 3.5, 4, 4.5, 5, 5.5, 6, 0.5, 1, 1.5, 2])
	Vol_ovr = np.array([2.682, 2.772, 2.842, 2.905, 2.956, 2.99, 3.039, 0.365, 0.866, 1.372, 1.962])
	Current = np.array([0, 0.02, 0.03, 0.05, 0.07, 0.09, 0.10, 0, 0, 0, 0])
	Intnsty = np.array([0.03, 0.08, 0.13, 0.18, 0.21, 0.25, 0.29, 0, 0, 0, 0])
	wavelength = 452 * 10 **(-9) #452 nm
	exp = lambda x,a,b: a * np.exp(x/b)-a
	Expfit, Covarience = curve_fit(exp, Vol_ovr, Current)
	xlist = np.linspace(0,3.05, 100)
	ylist = Expfit[0]*np.exp(xlist/Expfit[1])-Expfit[0]
	print(Expfit)
	print(Covarience)
	plt.plot(xlist, ylist, '--', label = "Fit")
	plt.plot(Vol_ovr, Current, '.', label = "Data")
	plt.grid()
	plt.legend()
	plt.xlabel("Voltage [V]")
	plt.ylabel("Current [A]")
	plt.title("Voltage versus Current")
	plt.savefig("Task1.png")
	plt.show()
task1_1()

def task1_2():
	Voltage = np.array([3, 3.5, 4, 4.5, 5, 5.5, 6, 0.5, 1, 1.5, 2])
	Vol_ovr = np.array([2.682, 2.772, 2.842, 2.905, 2.956, 2.99, 3.039, 0.365, 0.866, 1.372, 1.962])
	Current = np.array([0, 0.02, 0.03, 0.05, 0.07, 0.09, 0.10, 0, 0, 0, 0])
	Intnsty = np.array([0.03, 0.08, 0.13, 0.18, 0.21, 0.25, 0.29, 0, 0, 0, 0])
	wavelength = 452 * 10 **(-9)
	linearfit = pn.ForceLinearreg(Current, Intnsty)
	xlist = np.linspace(0, max(Current), len(Current))
	print(linearfit)
	plt.plot(xlist, linearfit[0] * xlist, '--', label = "Linear fit to data-set")
	plt.plot(Current, Intnsty, '.', label = "Data")
	plt.xlabel("Current [A]")
	plt.ylabel(r"Intensity $[\frac{W}{m^2}]$")
	plt.title("Intensity versus Current")
	plt.legend()
	plt.grid()
	plt.savefig("Task1_2.png")
	plt.show()
task1_2()

def task1_3():
	Voltage = np.array([3, 3.5, 4, 4.5, 5, 5.5, 6, 0.5, 1, 1.5, 2])
	Vol_ovr = np.array([2.682, 2.772, 2.842, 2.905, 2.956, 2.99, 3.039, 0.365, 0.866, 1.372, 1.962])
	Current = np.array([0, 0.02, 0.03, 0.05, 0.07, 0.09, 0.10, 0, 0, 0, 0])
	Intnsty = np.array([0.03, 0.08, 0.13, 0.18, 0.21, 0.25, 0.29, 0, 0, 0, 0])
	wavelength = 452 * 10 **(-9)
	exp = lambda x, a, b: a*np.exp(x/b)-a
	expfit, covarience = curve_fit(exp, Vol_ovr, Intnsty)
	xlist = np.linspace(0, max(Vol_ovr), len(Vol_ovr) * 10)
	ylist = expfit[0] * (np.exp(xlist/expfit[1]) - expfit[0])
	plt.plot(xlist, ylist, '--', label = "Exponential fit to data-set")
	print(expfit)
	print(covarience)
	plt.plot(Vol_ovr, Intnsty, '.', label = "Data")
	plt.xlabel("Voltage [V]")
	plt.ylabel(r"Intensity $[\frac{W}{m^2}]$")
	plt.legend()
	plt.title("Intensity versus Voltage")
	plt.grid()
	plt.savefig("task1_3.png")
	plt.show()
task1_3()
