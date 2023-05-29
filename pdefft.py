import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import axes3d
import matplotlib.cm as cm

a = 1 #thermal diffusivity constant
L = 100 #length of domain
N = 1000 #number of discrete points
dx = L/N
x = np.arange(-L/2, L/2, dx)

#define discrete wavenumbers
kappa = 2*np.pi*np.fft.fftfreq(N, d=dx)
u0 =  np.zeros_like(x)
u0[int((L/2-L/10)/dx):int((L/2+ L/10)/dx)] = 1
u0hat = np.fft.fft(u0)

#take only real part for odeint to do derivative
u0hat_ri = np.concatenate((u0hat.real, u0hat.imag))

#simulate in fourier freq domain
dt = 0.1
t = np.arange(0, 10, dt)

def rhsHeat(uhat_ri, t, kappa, a):
    uhat = uhat_ri[:N] + (1j)*uhat_ri[N:]
    d_uhat = -a**2 * (np.power(kappa, 2)) * uhat
    d_uhat_ri = np.concatenate((d_uhat.real, d_uhat.imag)).astype('float64')
    return d_uhat_ri

uhat_ri = odeint(rhsHeat, u0hat_ri, t, args=(kappa, a))
uhat = uhat_ri[:, :N] + (1j) *uhat_ri[:, N:]
print('hello is ')
print(uhat)
u = np.zeros_like(uhat)

for k in range(len(t)):
    u[k,:] = np.fft.ifft(uhat[k,:])

u = u.real

#waterfall plot
fig = plt.figure()
ax = fig.add_subplot(111, projection ='3d')
plt.set_cmap('jet_r')
u_plot = u[0:-1:10,:]
for j in range(u_plot.shape[0]):
    ys = j*np.ones(u_plot.shape[1])
    ax.plot(x,ys,u_plot[j,:], color=cm.jet(j*20))

plt.figure()
plt.imshow(np.flipud(u), aspect=8)
plt.axis('off')
plt.set_cmap('jet_r')
plt.show()
