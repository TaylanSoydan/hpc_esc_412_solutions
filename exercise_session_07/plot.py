import sys
import numpy as np
import matplotlib.pyplot as plt

filename = sys.argv[1]

data = np.loadtxt(filename)
x = data[:,0]
y = data[:,1]

plt.plot(x, y)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Log-log plot')
plt.grid()
plt.savefig(f"{filename}.png")
plt.show()

