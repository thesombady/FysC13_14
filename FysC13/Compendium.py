import numpy as np
import matplotlib.pyplot as plt
import os
from math import sin, cos, pi, floor
import scipy
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
    n_x = np.linspace(-10,10,20)
    n_y = np.linspace(-10,10,20)
    hbar = 8.6e-5
    l = 10e-9
    m_e = 511e3 #electron mass in eV
    X,Y = np.meshgrid(n_x, n_y)
    func = lambda x,y: hbar**2/(2*m_e*l)*2*np.pi*(x**2 + y**2)/(10e-9)
    func1 = np.vectorize(func)
    zval = func(X,Y)
    ax.contourf(X=n_x,Y=n_y, Z= zval, levels =100)#, extend3d=True)
    ax.set_xlabel(r'$k_x$')
    ax.set_ylabel(r'$k_y$')
    ax.set_zlabel('Energy')
    plt.title(r"Eigen-energies for $\bar{k}$")
    plt.xticks([])
    plt.yticks([])
    plt.gca().axes.zaxis.set_ticklabels([])
    plt.show()
#Particle_In_Box()
def DensityOfStates():
    kb = 8.6e-5
    hbar = 4.135667696*10**(-15)
    Energy = np.linspace(0.0001,2,1000)#in eV
    m_e = 511e3
    V = (0.2e-17)**(3)#volume in m
    gE = V/(2*np.pi**2)*(2*m_e/((hbar/(2*np.pi)) ** 2)) ** (3/2) * np.sqrt(Energy)
    T = [0, 280, 500]#Temperatures in Kelvin
    color = {
        0: (1,0,0,0.3),
        280 : (0,1,0,0.3),
        500 : (0,0,1,0.3)
    }
    for t in T:
        zlist = 1/(1+np.exp((Energy-1)/(kb*t)))
        plt.plot(Energy, zlist*gE, label = f"Fermi-Dirac at T = {t}K")
        plt.fill_between(Energy, y1=0, y2=zlist*gE, color = color[t])
    plt.plot(Energy,gE, label = r"$g(E)$")
    plt.xlabel("Energy")
    plt.ylabel(r"Density of states, $g(E)$")
    plt.legend()
    plt.title("Density of states")
    plt.xticks([])
    plt.yticks([])
    plt.show()
#DensityOfStates()
def Screening():
    kb = 8.6e-5
    hbar = 4.135667696*10**(-15)
    Energy = np.linspace(0.0001,2,1000)#in eV
    m_e = 511e3
    V = (0.2e-17)**(3)#volume in m
    gE = V/(2*np.pi**2)*(2*m_e/((hbar/(2*np.pi)) ** 2)) ** (3/2) * np.sqrt(Energy)
    zlist = 1/(1+np.exp((Energy-1)/(kb*280)))
    fig, ax = plt.subplots(2, sharex=True, sharey = True)
    x2 = np.linspace(-0.1,2,1000)
    g2 = gE = V/(2*np.pi**2)*(2*m_e/((hbar/(2*np.pi)) ** 2)) ** (3/2) * np.sqrt(x2)
    z2 = zlist = 1/(1+np.exp((x2-1)/(kb*280)))
    ax[1].plot(x2, g2, label = "Screened")
    ax[1].fill_between(x2, y1=-0.1, y2=z2*gE, color = (0.3,0,1,1))
    ax[1].fill_between(Energy, y1=0, y2=zlist*gE, color = (0.3,0,1,0.3))
    ax[0].plot(Energy, zlist*gE, label = f"Fermi-Dirac at T = 280 K", color = "Blue")
    ax[0].fill_between(Energy, y1=0, y2=zlist*gE, color = (0.3,0,1,0.3))
    ax[0].plot(Energy,gE,'.', label = r"$g(E)$", color="Red", markersize=0.5)
    plt.xlabel("Energy")
    plt.ylabel(r"Density of states, $g(E)$")
    ax[0].axvline(0, color = "Black", label = "X = 0", linestyle="-")
    ax[1].axvline(0, color = "Black", label = "X = 0", linestyle="-")
    ax[0].axvline(1, color = "gray", label = "X = Fermi", linestyle="-")
    ax[1].axvline(1, color = "gray", label = "X = Fermi", linestyle="-")
    plt.legend()
    for ax in fig.get_axes():
        ax.label_outer()
    fig.suptitle("Density of states")
    plt.xticks([])
    plt.yticks([])
    plt.show()
#Screening()

def mu():
    xlist = np.linspace(0,10,1000)#Energy in eV
    hbar = 4.13e-15/(2*np.pi) #in eV
    me = 511e3 #in eV
    V = 0.1e-53
    ge = V/(2*np.pi**2)*(2*me/(hbar**2))**(3/2)*np.sqrt(xlist)
    kb = 8.6e-5
    zlist = 1/(1+np.exp((xlist-1)/(kb*280)))
    plt.plot(xlist, ge*zlist,'.')
    plt.show()
mu()
