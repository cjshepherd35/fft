import numpy as np
from numpy.linalg import eig

a = np.array([[0,2],
              [2,3]])

t, d = eig(a)

print('eigenval ')
print(t)
print('eigenvec')
print(d)