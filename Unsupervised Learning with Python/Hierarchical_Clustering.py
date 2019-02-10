#Importing Modules
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import pandas as pd

#Reading the DataFrame
seeds_df = pd.read_csv('seeds-less-rows.csv')

# Remove the grain species from the DataFrame, save for later
varieties = list(seeds_df.pop('grain_variety'))

# Extract the measurements as a NumPy array
samples = seeds_df.values

mergings = linkage(samples, method='complete')

dendrogram(mergings,labels=varieties,leaf_rotation=90,leaf_font_size=6)
plt.show()
