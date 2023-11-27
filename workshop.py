import numpy as np


#x = np.array(0,1)
#print(x)
x = np.array((0,1))
y = np.array((1,1))
print(x)
print((x == y).all())

x[0] = x[0]+18
print(x)
print((x == y).all())

z, za = x
print(z)
print(za)