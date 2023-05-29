import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#simple signal with noise
dt = 0.001
t = np.arange(0,1,dt)
f = np.sin(2*np.pi*50*t) + np.sin(2*np.pi*120*t)
f_clean = f
f = f + np.random.randn(len(t))

n = len(t)
fhat = np.fft.fft(f,n)
psd = fhat *np.conj(fhat) / n
freq = (1/dt*n) * np.arange(n)
L = np.arange(1,np.floor(n/2), dtype='int')

indices = psd > 100
psdclean = psd * indices
fhat = indices*fhat
ffilt = np.fft.ifft(fhat)

sumsquare = 0
for x in range(len(f)):
    sumsquare = sumsquare + (ffilt[x] - f_clean[x])**2

print(sumsquare)


fig, axes = plt.subplots(3,1)
plt.sca(axes[0])
plt.plot(t,f,color='k', label="noisy")
plt.plot(t,f_clean, color='r', label='clean')
plt.legend()
plt.sca(axes[1])
plt.plot(freq[L], psd[L], color='c', label='noisy')
plt.legend()
plt.sca(axes[2])
plt.plot(t,ffilt, color='g', label='filtered')
plt.ylim(-5,5)
plt.legend()
plt.show()

