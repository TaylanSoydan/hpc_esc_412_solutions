import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_csv('fout.txt', sep=",", header=None).values
data = data[0][:-1].reshape((100,100))
data = np.log1p(data)
fig = sns.heatmap(data).get_figure()
fig.savefig("heatmap.png")
