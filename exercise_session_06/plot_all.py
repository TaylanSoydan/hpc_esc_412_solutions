import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='Plot data from text file')
parser.add_argument('filename1', type=str, help='name of first file')
parser.add_argument('filename2', type=str, help='name of second file')
parser.add_argument('filename3', type=str, help='name of third file')
args = parser.parse_args()

# Load data from files
data1 = np.loadtxt(args.filename1)
data2 = np.loadtxt(args.filename2)
data3 = np.loadtxt(args.filename3)

# Set up subplots
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15,5))

# Plot data for first file
axs[0].plot(data1[:,0], data1[:,1])
axs[0].set_xscale('log')
axs[0].set_yscale('log')
axs[0].set_xlabel('x')
axs[0].set_ylabel('y')
axs[0].set_title(args.filename1)

# Plot data for second file
axs[1].plot(data2[:,0], data2[:,1])
axs[1].set_xscale('log')
axs[1].set_yscale('log')
axs[1].set_xlabel('x')
axs[1].set_ylabel('y')
axs[1].set_title(args.filename2)

# Plot data for third file
axs[2].plot(data3[:,0], data3[:,1])
axs[2].set_xscale('log')
axs[2].set_yscale('log')
axs[2].set_xlabel('x')
axs[2].set_ylabel('y')
axs[2].set_title(args.filename3)

# Save the plot to file
plt.savefig(args.filename1.replace('.txt', '') + '_combined.png')

