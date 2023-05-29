import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt

dt = 0.001
t = np.arange(0,2,dt)
f0 = 50
f1 = 250
t1 = 2
x = np.cos(2*np.pi*t*(f0 + (f1-f0)*np.power(t, 2)/(3*t1**2)))
fs = 1/dt
sd.play(2*x, fs)
plt.specgram(x, NFFT=128, Fs=1/dt, noverlap=120, cmap='jet_r')
plt.colorbar()
plt.show()