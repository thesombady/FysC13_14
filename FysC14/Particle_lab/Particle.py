import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, sys
import io

Path = os.path.join(os.getcwd(), "FysC14/Particle_lab/Data.txt")
#Dataframe = pd.read_csv(Path)
#print(Dataframe)
#print(Dataframe.keys())

with open(Path, "r") as file:
    Data = file.read()
Dataset = io.StringIO(Data.decode('utf-8'))
Dataframe = Dataset.readlines()
print(Dataframe)
