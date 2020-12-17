import pandas as pd
import PhysicsNum as pn
import numpy as np
import os, sys
import matplotlib.pyplot as plt

PATH1 = os.path.join("/Users/andreasevensen/Desktop/XrayDiffraction", "AG.xyd")
PATH2 = os.path.join("/Users/andreasevensen/Desktop/XrayDiffraction", "Al2O3.xyd")
PATH3 = os.path.join("/Users/andreasevensen/Desktop/XrayDiffraction", "mixture.xyd")

def Parser(Path):
    try:
        with open(Path, 'r') as file:
            Data = file.readlines()
        Xlist = []
        Ylist = []
        for i in range(len(Data)):
            Values = Data[i].split(' ')
            val2 = Values[-1].replace("\n", '')
            Xlist.append(float(Values[0]))
            Ylist.append(float(val2))
        return np.array(Xlist), np.array(Ylist)
    except:
        raise Exception("[Parser]: Cant find the input")

"""
AGData = Parser(PATH1)
Al2O3Data = Parser(PATH2)
MixtureData = Parser(PATH3)
plt.plot(AGData[0], AGData[1], '.', markersize = 1, label = "Ag Data")
plt.legend()
plt.grid()
plt.show()
"""
