import pandas as pd
import matplotlib.pyplot as plt

file = open("data.txt", "a")


df = pd.read_csv("train.csv")
df.info()
# print("\nnumber of ids is: {}" .format(df['id'].size))
# print("\nnumber of unique values in landmark_id column is: {}" .format(df['landmark_id'].nunique()))
# print("\nthe 5 classes with most items:")
# print(df['landmark_id'].value_counts().head())
# print("\nthe 5 classes with least items:")
# print(df['landmark_id'].value_counts().tail())


print(df.loc[df['landmark_id']==197779])

"""
fig=plt.figure()
ax = df.plot.hist(bins=81313, grid=False, rwidth=0.9)
ax.set_xlabel("classes", labelpad=20, weight='bold', size=12)
ax.set_ylabel("amount of classes", labelpad=20, weight='bold', size=12)
plt.show()
"""
