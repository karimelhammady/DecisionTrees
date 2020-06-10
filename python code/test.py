import pandas as pd
import matplotlib.pyplot as plt
import sys
import numpy as np
from scipy import stats

np.set_printoptions(threshold=sys.maxsize)
# read dataset file
df = pd.read_csv('D:/ASU/data mining/final research/code/dataset_train.csv')
#  take part of the data to be used for plotting
part = df.iloc[:, 4]
# display quick summary for data
df.info()
# plot all attributes in one graph
df.plot()
# plot a certain range
part.plot.box()
axs = part.plot.area(figsize=(9, 4), subplots=True)
pd.DataFrame.hist(part)
part.boxplot()
plt.show()

# perform stats on all attrbiutes
print(part.describe())
# remove outliers
new_df = df
for i in df:
    z_scores = stats.zscore(df[i])
    abs_z_scores = np.abs(z_scores)
    new_df = df[abs_z_scores <= 0.5]
    
# plot after removing outliers
part = new_df.iloc[:, 4]
part.plot.box()
plt.show()


