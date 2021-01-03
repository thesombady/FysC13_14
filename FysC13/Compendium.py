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
"""
"""
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
"""
def Fermi_Dirac():
    kb = 8.6e-5
    xval = np.linspace(0,5,1000)
    T = [0,260,1000]
    for t in T:
        yval = 1 / (1+np.exp((xval-2.5)/(kb*t)))
        plt.plot(xval, yval, '-', label = f"Temp: {t} K")
    plt.xlabel("States -> eV")
    plt.ylabel("Probability")
    plt.title(r"Fermi-Dirac distribution for $\mu = 3$ eV")
    plt.legend()
    #plt.savefig("Fermi_Dirac.png")
    plt.show()
#Fermi_Dirac()
def Particle_In_Box():
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    n_x = np.linspace(-10,10,1000)
    n_y = np.linspace(-10,10,1000)
    n_z = np.linspace(-10,10,1000)
    hbar = 8.6e-5
    l = 10e-9
    m_e = 511e3 #electron mass in eV
    #E_x = hbar**2*n_x**2/(l*m_e*2)*2*np.pi
    #E_y = hbar**2*n_y**2/(l*m_e*2)*2*np.pi
    X,Y = np.meshgrid(n_x, n_y)
    func = lambda x,y: hbar**2/(2*m_e*l)*2*np.pi*(x**2 + y**2)
    func1 = np.vectorize(func)
    zval = func1(X,Y)
    X = hbar**2/(2*m_e*l)*2*np.pi*(n_x)
    Y = hbar**2/(2*m_e*l)*2*np.pi*(n_y)
    Z = hbar**2/(2*m_e*l)*2*np.pi*(n_z)
    ax.contourf(X=X,Y=Y,Z = zval)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.title(r"Eigen-energies for $\bar{k}$")
    plt.show()



Particle_In_Box()
