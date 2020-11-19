import numpy as np
import matplotlib.pyplot as plt
import os

def P(a):
    xlist = np.linspace(-5,-5,100)
    ylist = np.exp(xlist*2*np.pi/a)
    return xlist, ylist
Values1 = P(10)
plt.plot(Values1[0],Values1[1])
plt.show()
