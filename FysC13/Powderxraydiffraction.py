import PhysicsNum as pn
import numpy as np
import os, sys
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import math
try:
    PATH1 = os.path.join("/Users/andreasevensen/Desktop/XrayDiffraction", "AG.xyd")
    PATH2 = os.path.join("/Users/andreasevensen/Desktop/XrayDiffraction", "Al2O3.xyd")
    PATH3 = os.path.join("/Users/andreasevensen/Desktop/XrayDiffraction", "mixture.xyd")
except:
    try:
        Folder = os.path.join(os.getcwd(), "XrayData")
        PATH1 = os.path.join(Folder, "Ag.xyd")
        PATH2 = os.path.join(Folder, "Al2O3.xyd")
        PATH3 = os.path.join(Folder, "mixture.xyd")
    except:
        raise PATHError("System; Cant locate the files")
def Parser(Path):
    """Parser function provides the parsered data provided from a .xyd file. This method requires that the data
    is strictly ordered"""
    try:
        with open(Path, 'r') as file:
            Data = file.readlines()
        Xlist = []
        Ylist = []
        for i in range(len(Data)):
            Values = Data[i].split(' ')
            val2 = Values[-1].replace("\n", '')#Remove a newline and replace by nothing
            Xlist.append(float(Values[0]))
            Ylist.append(float(val2))
        return np.array(Xlist), np.array(Ylist)
    except:
        raise Exception("[Parser]: Cant find the input")

class Gaussian(object):
    Name = "Name Of Tested Object"
    Wavelength = 1.54
    def __init__(self, Data):
        """Data containing both x-values and y-values"""
        try:
            self.xlist = Data[0]
            self.ylist = Data[1]
        except:
            raise ImportError("[Gaussian]: Cant import the data")
        self.Computed = []
        self.Simluated = []
        self.MuValues = []
        self.SigmaValues = []
        self.Amplitud = []
        self.ErrorMu = []
        self.ErrorSigma = []
        self.RieArea = []

    def plot(self):
        plt.title(r"Intensity versus $2\cdot \Theta$ angle for {} data".format(self.Name))
        plt.plot(self.xlist, self.ylist, '.', markersize = 1.5, label = self.Name + " Data")
        plt.xlabel(r"$2\cdot \Theta$ [degree]")
        plt.ylabel("Intensity [a.u]")
        plt.grid()
        plt.legend()
        plt.show()

    def ComputeGaussian(self, index1, index2, Plot = False):
        xvalues = self.xlist[index1:index2]
        yvalues = self.ylist[index1:index2]
        mean = sum(xvalues * yvalues)/sum(yvalues)
        guesssigma = np.sqrt(sum(yvalues * (xvalues - mean)**2) / sum(yvalues))
        Gaussianfunc = lambda x, a, mu, sigma: a*np.exp(-(x - mu)**2/(2*sigma**2))
        Fit, covarience = curve_fit(Gaussianfunc, xvalues, yvalues, p0 = [max(yvalues), mean, guesssigma])
        xlist = np.linspace(xvalues[0] - 5, xvalues[-1] + 5, len(xvalues)*100)
        ylist = Gaussianfunc(xlist, Fit[0], Fit[1], Fit[2])
        self.Simluated.append(Gaussianfunc(self.xlist, Fit[0], Fit[1], Fit[2]))
        func = lambda x: Fit[0]*np.exp(-(x - Fit[1])**2/(2*Fit[2]**2))
        Area = pn.RiemanSum(func, self.xlist[0], self.xlist[-1])
        self.RieArea.append(Area)
        if Plot == True:
            plt.plot(xlist, Gaussianfunc(xlist, Fit[0], Fit[1], Fit[2]), '-', label = "Guassian fit")
            plt.plot(self.xlist, self.ylist, '.', markersize = 1.5, label = self.Name)
            plt.xlabel(r"$2\cdot \Theta$ [degree]")
            plt.title(r"Intensity versus $2\cdot \Theta$ angle for {} data".format(self.Name))
            plt.ylabel("Intensity [a.u]")
            plt.grid()
            plt.legend()
            plt.show()
        self.ErrorMu.append(covarience[2][2])
        self.ErrorSigma.append(covarience[1][1])
        self.MuValues.append(Fit[1])
        self.SigmaValues.append(Fit[2])
        self.Computed.append((xlist, ylist))
        self.Amplitud.append(Fit[0])
        return func, Fit, covarience

    def TestValues(self, index1, index2):
        plt.plot(self.xlist, self.ylist, '.', markersize = 1.5)
        plt.xlabel(r"$2\cdot \Theta$ [degree]")
        plt.ylabel("Intensity [a.u]")
        plt.axvline(self.xlist[index1])
        plt.axvline(self.xlist[index2])
        plt.grid()
        plt.show()

    def Plotall(self):
        try:
            plt.plot(self.xlist, self.ylist, '.', markersize = 1.5, label = self.Name + " Data")
            for i in range(len(self.Computed)):
                plt.plot(self.Computed[i][0], self.Computed[i][1], '-', label = f"Fitted peak {i+1}")
            plt.title(r"Intensity versus $2\cdot \Theta$ angle for {} data".format(self.Name))
            plt.ylabel("Intensity [a.u]")
            plt.xlabel(r"$2\cdot \Theta$ [degree]")
            plt.legend()
            plt.grid()
            plt.show()
        except Exception as e:
            raise e

    def SimlulatedData(self):
        Exponential = lambda x, A, tau, b: A*np.exp(-x/tau) + b
        xlist = self.xlist.copy()
        ylist = self.ylist.copy()
        for i in range(1,len(ylist)):
            if ylist[i]>ylist[i-1]:
                ylist[i] = ylist[i-1]
            else:
                continue
        newxlist = np.linspace(xlist[0], xlist[-1], len(xlist) * 10)
        Fit, Covarience = curve_fit(Exponential, xlist, ylist, p0 = [max(ylist), 10, 100])
        plt.plot(newxlist, Exponential(newxlist, Fit[0], Fit[1], Fit[2]),'--', label = "Exponential fit of the data")
        try:
            plt.plot(self.xlist, self.ylist, '.', markersize = 1.5, label = self.Name + " data")
            for i in range(len(self.Simluated)):
                plt.plot(self.xlist, self.Simluated[i] + ylist, '-', label = f"Fitted peak {i+1}")
            plt.title(r"Intensity versus $2\cdot \Theta$ angle for {} data".format(self.Name))
            plt.ylabel("Intensity [a.u]")
            plt.xlabel(r"$2\cdot \Theta$ [degree]")
            plt.legend()
            plt.grid()
            plt.show()
        except Exception as e:
            raise e

    def printinfo(self):
        print(f"Sample:{self.Name}\nPeak locactions :{self.MuValues}\nAmplitude :{self.Amplitud}\n#Peaks:{len(self.MuValues)}\nArea:{self.RieArea}\nMiller:{self.Miller}")

    def __str__(self):
        return "Mu values :{}\nSigma values:{}".format([self.MuValues[i] for i in range(len(self.MuValues))], [self.SigmaValues[i] for i in range(len(self.SigmaValues))])

    def __repr__(self):
        return "Mu values :{}\nSigma values:{}".format([self.MuValues[i] for i in range(len(self.MuValues))], [self.SigmaValues[i] for i in range(len(self.SigmaValues))])

    def Compute(self, a = 1, b = None , c = None, alpha = 1, Beta = 1, gamma = 0, Accurarcy = 0.01):
        """Distances in units of Å, and returns the appropiate miller incidies"""
        Distances = []
        for i in range(len(self.MuValues)):
            value = math.radians(self.MuValues[i]/2)
            d = self.Wavelength/(2*math.sin(value))
            #print(f"d = {d} [Å] for the data {self.Name} for peak {i+1}.")
            Distances.append(d)
        if b == None or c == None:
            b = a; c = a
        Miller = {}
        for h in [0,1,2,3,4]:
            for k in [0,1,2,3,4]:
                for l in [0,1,2,3,4]:
                    for i in range(len(Distances)):
                        Difference = abs((1/Distances[i])**2-((h/a)**2 + (k/b)**2 + (l/c)**2))
                        if Difference < Accurarcy:
                            Miller[f'{i+1}'] =  f"({h},{k},{l})"
        self.Miller = Miller

        def Scherrers():
            T = lambda k, beta, theta: k*self.Wavelength/(beta * math.cos(math.radians(theta)))
            FWHM = lambda sigma: 2*math.sqrt(2*math.log(2))*sigma
            FWHM_Values = []
            for val in (self.SigmaValues):
                FWHM_Values.append(FWHM(np.radians(val)))
            Mean = []
            for i in range(len(FWHM_Values)):
                #for j in range(len(self.MuValues)):
                Val = T(0.94, FWHM_Values[i], self.MuValues[i]/2)#Change to j if we decide to change
                print(f"Scherrer's is then {Val} Å for peak {i+1}")#Change to j if we decide to change
                Mean.append(Val)
            print(f"The mean Scherrer's shape is then {sum(Mean)/len(Mean)} [Å]")
            self.Scherrer = Mean
        Scherrers()


def SilverComputation():
    """Silver, Ag, has a Cubic structure with a lattice constant of a = 4.086 Å"""
    global Silver
    Silver = Gaussian(Parser(PATH1))
    Silver.Name = "Silver"
    Peak1 = Silver.ComputeGaussian(1150, 1275)
    Peak2 = Silver.ComputeGaussian(1550, 1675)
    Peak3 = Silver.ComputeGaussian(2875, 3050)
    Peak4 = Silver.ComputeGaussian(3730, 3900)
    Peak5 = Silver.ComputeGaussian(4050, 4125)
    Peak6 = Silver.ComputeGaussian(5150, 5210)
    Silver.SimlulatedData()
    Silver.Compute(4.086, Accurarcy = 0.01)
    Silver.printinfo()
SilverComputation()

def Al2O3Compuation():
    """Al2O3 has a hexogonal structure with the following constants, a = b = 4.7589 Å and c = 12.991 Å, α = β= 90° and γ = 120 """
    global Al2O3
    Al2O3 = Gaussian(Parser(PATH2))
    Al2O3.Name = "Al2O3"
    Peak1 = Al2O3.ComputeGaussian(340, 405)
    Peak2 = Al2O3.ComputeGaussian(610, 650)# A small peak
    Peak3 = Al2O3.ComputeGaussian(950, 1050)
    Peak4 = Al2O3.ComputeGaussian(1125, 1250)
    Peak5 = Al2O3.ComputeGaussian(1500, 1625)
    Peak6 = Al2O3.ComputeGaussian(2110, 2225)
    Peak7 = Al2O3.ComputeGaussian(2425, 2575)
    Peak8 = Al2O3.ComputeGaussian(2620, 2660)# Very small peak
    Peak9 = Al2O3.ComputeGaussian(2710, 2780)# Very small peak
    Peak10 = Al2O3.ComputeGaussian(3050, 3150)
    Peak11 = Al2O3.ComputeGaussian(3160, 3275)
    Peak12 = Al2O3.ComputeGaussian(3750, 3830)
    Peak13 = Al2O3.ComputeGaussian(3995, 4075)
    Peak14 = Al2O3.ComputeGaussian(4245, 4300)# Very small peak
    Peak15 = Al2O3.ComputeGaussian(4375, 4460)# Very small peak
    Peak16 = Al2O3.ComputeGaussian(4565, 4620)# Very small peak
    Peak17 = Al2O3.ComputeGaussian(4699, 4760)# Somewhat small peak
    Peak18 = Al2O3.ComputeGaussian(4950, 5070)
    Al2O3.printinfo()
Al2O3Compuation()

def MixtureComputation():
    global Mixture
    Mixture = Gaussian(Parser(PATH3))
    Mixture.Name = "Ag & Al2O3 mix"
    Peak1 = Mixture.ComputeGaussian(325,400)
    Peak2 = Mixture.ComputeGaussian(960, 1050)
    Peak3 = Mixture.ComputeGaussian(1125, 1275)
    Peak4 = Mixture.ComputeGaussian(1500, 1575)
    Peak5 = Mixture.ComputeGaussian(1580, 1650)
    Peak6 = Mixture.ComputeGaussian(2125, 2195)
    Peak7 = Mixture.ComputeGaussian(2450, 2530)
    Peak8 = Mixture.ComputeGaussian(2710, 2775)
    Peak9 = Mixture.ComputeGaussian(2900, 3000)
    Peak10 = Mixture.ComputeGaussian(3060, 3125)
    Peak11 = Mixture.ComputeGaussian(3175, 3250)
    Peak12 = Mixture.ComputeGaussian(3750, 3875)
    Peak13 = Mixture.ComputeGaussian(4010, 4070)
    Peak14 = Mixture.ComputeGaussian(4075, 4125)
    Peak15 = Mixture.ComputeGaussian(4250, 4300)
    Peak16 = Mixture.ComputeGaussian(4390, 4460)
    Peak17 = Mixture.ComputeGaussian(4560, 4625)
    Peak18 = Mixture.ComputeGaussian(4700, 4775)
    Peak19 = Mixture.ComputeGaussian(4975, 5050)
    Mixture.SimlulatedData()
    Mixture.Compute()
    Mixture.printinfo()
MixtureComputation()

def Excersise8():
    I2 = 409.96845246324233
    I1 = 267.0403396310394
    X1 = lambda k: 1/(1+k*I2/I1)
    dirivative = lambda k: 1/0.000001*(X1(k+0.000001)-X1(k))
    xval = np.linspace(0,10,100)
    xdir = np.array(list(map(dirivative, xval)))
    value = -xdir[0]
    realx = lambda x: value*x
    x1list = np.array(list(map(X1, xval)))
    X1val = X1(value)
    X2 = 1 - X1val
    print(X1val, X2)
Excersise8()
