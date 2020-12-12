import os, sys
import numpy as np
import matplotlib.pyplot as plt
def Excersise1():
    K = np.linspace(0, 10, 1000)
    hbar = (1.05) * 10 ** (-34)
    g = np.pi/2
    m = 9.11 * 10 ** (-19)
    u = 0
    hbarsq = hbar ** 2
    #E1 is positive
    E1 = (hbarsq * (K - g)**2 + hbarsq * K **2)/(4*m) + np.sqrt(((hbarsq * (K - g)**2 + hbarsq * K **2)/(4*m))**2
        + u ** 2 - (hbarsq ** 2* K**2 * (K-g)**2)/(4*m)**2)
    #E2 is negative
    E2 = (hbarsq * (K - g)**2 + hbarsq * K **2)/(4*m) - np.sqrt(((hbarsq * (K - g)**2 + hbarsq * K **2)/(4*m))**2
        + u ** 2 - (hbarsq ** 2* K**2 * (K-g)**2)/(4*m)**2)
    plt.plot(K, E1, 'b', label = "Positive solution")
    plt.plot(K, E2, 'r', label = "Negative solution")
    plt.title("Energy versus k")
    plt.xlabel(f'k')
    plt.ylabel(r'E in $J$ ')
    plt.grid()
    plt.legend()
    plt.show()
    u = 1 * 10 ** (-49)
    #E1 is positive
    E1 = (hbarsq * (K - g)**2 + hbarsq * K **2)/(4*m) + np.sqrt(((hbarsq * (K - g)**2 + hbarsq * K **2)/(4*m))**2
        + u ** 2 - (hbarsq ** 2* K**2 * (K-g)**2)/(4*m)**2)
    #E2 is negative
    E2 = (hbarsq * (K - g)**2 + hbarsq * K **2)/(4*m) - np.sqrt(((hbarsq * (K - g)**2 + hbarsq * K **2)/(4*m))**2
        + u ** 2 - (hbarsq ** 2* K**2 * (K-g)**2)/(4*m)**2)
    plt.plot(K, E1, 'b', label = "Positive solution")
    plt.plot(K, E2, 'r', label = "Negative solution")
    plt.title("Energy versus k")
    plt.xlabel(f'k')
    plt.ylabel(r'E in $J$ ')
    plt.grid()
    plt.legend()
    plt.show()
#Excersise1()
hbar = 6.626 * 10 ** (-34)
me = 9.311 * 10 ** (-31)
U = 60 * 1.602 * 10 ** (-19)
Energy = lambda sigma: - U + U * hbar * me /(2 * sigma**2)
sigma = 1
i = 0
while True:
    print(Energy(sigma)/(1.602 * 10 ** (-19)))
    sigma -=0.00001
    print(i)
    i += 1
    if i == 1000000:
        break
