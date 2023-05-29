import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

n = 64
L = 30
dx = L/n
x = np.arange(-L/2, L/2, dx, dtype='complex_')
f = np.cos(x) * np.exp(-np.power(x,2)/25)
df = -(np.sin(x) * np.exp(-np.power(x,2)/25) + (2/25)*x*f)

#approximate derivative using finite difference
dffd = np.zeros(len(df), dtype='complex_')
for kappa in range(len(df)-1):
    dffd[kappa] = (f[kappa+1]-f[kappa])/dx

dffd[-1] = dffd[-2]

fhat = np.fft.fft(f)
kappa = (2*np.pi/L)*np.arange(-n/2, n/2)
kappa = np.fft.fftshift(kappa)
dfhat = kappa *fhat * (1j)
dfFFt = np.real(np.fft.ifft(dfhat))

plt.plot(x, df.real, color='r', label='true derivative')
plt.plot(x,dffd.real, '--', color='b', label='finite diff')
plt.plot(x, dfFFt.real, '--', color='c', label='fft deriv')
plt.legend()
plt.show()