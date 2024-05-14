"""
This program is to plot some things for the midsem presentation.. the plots in this program should include:
        pi chart for number of normal/abnormal, murmur/not
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

pn_path = "./datasets/physionet/the-circor-digiscope-phonocardiogram-dataset-1.0.3/training_data.csv"
pn_df = pd.read_csv(pn_path)

# plot pi chart for number of normal/abnormal, murmur/not
print("values in 'Murmur' column: ", pn_df["Murmur"].value_counts())
print("values in 'Outcome' column: ", pn_df["Outcome"].value_counts())
# c = ["#FF0000", "#FF6961", "#FFA07A", "#FFD8B1"]  # shades of red
# colors = ["#ff9999", "#66b3ff", "#99ff99", "#ffcc99", '#c2c2f0','#ffb3e6'] # colours from medium article
c = ["#FF758F", "#FF4D6D", "#C9184A"]
c1 = ["#FF4D6D", "#C9184A"]
fig, ax = plt.subplots(1, 2)
labels = pn_df["Murmur"].unique()
sizes = pn_df["Murmur"].value_counts()
explode = (0, 0.1, 0)
patches, texts, autotexts = ax[0].pie(
    sizes,
    explode=explode,
    labels=labels,
    autopct="%1.1f%%",
    shadow=True,
    startangle=200,
    # colors=colors[:3],
    colors=c,
)
ax[0].set_title("Murmur values")
ax[0].axis("equal")

labels = pn_df["Outcome"].unique()
sizes = pn_df["Outcome"].value_counts()
explode = (0.1, 0.1)
cmap = LinearSegmentedColormap.from_list("Reddish", c, N=len(sizes))
colors = cmap(np.linspace(0, 1, len(sizes)))
for text in texts:
    text.set_color("dimgrey")
for autotext in autotexts:
    autotext.set_color("darkslategrey")
patches, texts, autotexts = ax[1].pie(
    sizes,
    # explode=explode,
    labels=labels,
    autopct="%1.1f%%",
    # shadow=True,
    # colors=colors[-2:],
    colors=c1,
)
"""
for text in texts:
    text.set_color("darkslategray")
for autotext in autotexts:
    autotext.set_color("grey")
"""
ax[1].set_title("Outcome values")
ax[1].axis("equal")
plt.tight_layout()
plt.show()
