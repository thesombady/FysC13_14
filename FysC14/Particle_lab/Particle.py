import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, sys

Path = os.path.join(os.getcwd(), "FysC14/Particle_lab/Data.txt")
#Data = pd.read_txt()

with open(Path, "r") as file:
    print(file.read())
