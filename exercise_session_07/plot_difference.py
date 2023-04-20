import sys
import numpy as np
import matplotlib.pyplot as plt

# Parse command-line arguments
if len(sys.argv) != 3:
    print(f"Usage: python3 {sys.argv[0]} file1.txt file2.txt")
    sys.exit(1)
file1, file2 = sys.argv[1], sys.argv[2]

# Load data from first file
data1 = np.loadtxt(file1)

# Load data from second file
data2 = np.loadtxt(file2)

# Divide second column of second file by second column of first file
denominators = data1[:, 1] + 1e-10  # add a small constant value to avoid division by zero
divisions = (data2[:, 1] + 1e-10)  / denominators

# Plot first column of first file vs. divisions
plt.semilogx(data1[:, 0], divisions)
plt.title("Log binning")
plt.xlabel('k')
plt.ylabel('Paralel/Serial')

# Save plot as image file
plt.savefig('difference.png')

