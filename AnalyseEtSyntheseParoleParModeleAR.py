# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 11:14:14 2020
@contact: atto / abatt@univ-smb.fr 
"""
#%%
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy.signal import lfilter, hamming, freqz, deconvolve, convolve



DataParole = loadmat('DataParole.mat')
DataParole = DataParole['DataParole']
wait = input("Ajuster le volume - Puis Appuyer sur une touche du clavier pour continuer.")
import sounddevice as sd
sd.play(DataParole, 8192) # son emis via haut parleur externe 

z = DataParole
plt.plot(z)
plt.ylabel('Data Parole')
plt.show()

n1 = 200;
n2 = len(z);
n3 = n2-n1+1
#n1 et n2 sont les debut et fin de la s�rie a analyser

y=z[n1:n2]

#longueur 
import numpy as np
m=150
np.floor([n3/m])
ordreAR =8

y1=y[1:m]


plt.plot(y1)
plt.ylabel('Data Parole')
plt.title("Trame 1")
plt.show()
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tsa.ar_model import AR
#Je n'arrivait pas à trouver de fonction lpc pour oython j'ai donc utiliser statsmodel pour avoir les coefficients du model
#train the autoregression model
model = AR(y1)
model_fitted = model.fit()
print('The lag value chose is: %s' % model_fitted.k_ar)
print('The coefficients of the model are:\n %s' % model_fitted.params)

import random as aleas

a = model_fitted.params
z=[k*0 for k in range(len(y1))]
for k in range(1,len(y1)):
    z[k]=-a[0]*y[k-1]-a[1]*y[k-2]-a[2]*y[k-3]-a[3]*y[k-4]-a[4]*y[k-5]-a[5]*y[k-6]-a[6]*y[k-7]-a[7]*y[k-8]
    
plt.plot(range(len(y1[4:])),z[4:],label='Data =series stationnaires 1')
plt.title("Serie stationnaire 1")
plt.show()

print(model_fitted.sigma2)
yf1=lfilter(a[0:9],1,y1)

"""Synth2=lfilter(1,a[0:9],rand)
Synth3 = lfilter(1,a[0:9],yf1)"""

plt.plot(y1[4:], 'b-', label='data')
plt.plot(model_fitted.fittedvalues[4:], 'r-', label='data')
plt.show()

plt.plot(y1[0:], 'b-', label='data')
plt.plot(yf1[0:], 'g-', label='data')
plt.show()



n = 150
res = y1- yf1
"""
for i in range(1,25):
    print(i)
    y2 = y[n*i:n*i+n]
    model = AR(y2)
    model_fitted = model.fit()
    coeffsAR = model_fitted.params
    yf2 = lfilter(coeffsAR[1:8],1,y2)
    plt.plot(yf2, 'g-', label='data')
    plt.plot(y2[0:], 'b-', label='data')
    plt.title("Trame %d"%i)
    plt.grid()
    plt.show()
    res= np.concatenate((res,y2[0:-1]-yf2[1:]), axis=0)


"""
NbTramesAffichees = 10;  # doit etre inferieur a  NbTrames
m1=ordreAR+1
k=1
residuel = y1-yf1

NbTrames = int((n2-n1+1)/m)
for k in range(1,NbTrames-1):
    y2 = y[k*m -m1 + 1 : (k+1)*m]
    model = AR(y2)
    model_fitted = model.fit()
    coeffsAR = model_fitted.params
    yf2 = lfilter(coeffsAR[1:8],1,y2)
    residuel2 = y2[m1:m1+m-1]-yf2[m1:m1+m-1]
    residuel = np.concatenate((residuel,residuel2), axis=0)
    """synth2 = lfilter(1,coeffsAR[1:9],np.random.randn(1,len(y2)))
    synth3=lfilter(1,coeffsAR[1:9],yf2)
    synth3 = synth3[m1:m1+m-1]
    synth2 = synth2[m1:m1+m-1]
    Synth2 = np.concatenate((Synth2,synth2), axis=0)
    Synth3 = np.concatenate((Synth3,synth3), axis=0)"""
    if k< 10:
        
        plt.plot(yf2[m1:m1+m-1], 'g-', label='data')
        plt.plot(y2[m1:m1+m-1], 'b-', label='data')
        plt.title("Trame %d, Estimée vs Réalité "%k)
        plt.legend('estimee','Vraie')
        plt.grid()
        plt.show()

plt.plot(residuel) 
plt.title("Parole estimée")
plt.grid()
plt.show()
plt.plot(y) 
plt.title("Vraie Paroles")
plt.grid()
plt.show()
import time
#on constate qu'il reste encore des améliorations à faire mais on n'entend quand même après plusieurs écoutes les paroles 
sd.play(residuel, 8192)
time.sleep(3)
sd.play(residuel, 8192)

