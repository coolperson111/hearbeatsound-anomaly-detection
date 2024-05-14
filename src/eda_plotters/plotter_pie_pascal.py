"""
This program is to plot some things for the midsem presentation.. the plots in this program should include:
        pi chart for number of normal/abnormal, murmur/not
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

# pn_path = "./datasets/pascal/set_a.csv"
pn_path = "./datasets/pascal/set_b.csv"
pn_df = pd.read_csv(pn_path)

# plot pi chart for number of normal/abnormal, murmur/not
print("values in 'Murmur' column: ", pn_df["label"].value_counts())
# c = ["#FF0000", "#FF6961", "#FFA07A", "#FFD8B1"]  # shades of red
# colors = ["#ff9999", "#66b3ff", "#99ff99", "#ffcc99", '#c2c2f0','#ffb3e6'] # colours from medium article
c = ["#FF758F", "#FF4D6D", "#C9184A", "#FF4D6D", "#C9184A"]
fig, ax = plt.subplots()
labels = pn_df["label"].unique()
print("labels: ", labels)
sizes = pn_df["label"].value_counts()
# add nans to sizes - append doesnt work
sizes = sizes._append(pd.Series([pn_df['label'].isnull().sum()], index=["NaN"]))
print("sizes: ", sizes)
explode = (0, 0.1, 0)
patches, texts, autotexts = ax.pie(
    sizes,
    # explode=explode,
    labels=labels,
    autopct="%1.1f%%",
    shadow=True,
    startangle=200,
    # colors=colors[:3],
    colors=c,
)
ax.set_title("Pascal labels- set b")
ax.axis("equal")

for text in texts:
    text.set_color("dimgrey")
for autotext in autotexts:
    autotext.set_color("darkslategrey")
plt.show()
