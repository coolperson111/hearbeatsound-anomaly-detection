"""
Features selection based on correlation Matrix
Directly applying correlation matrix on the the full_data
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

df = pd.read_csv("data/full_data.csv")

print(df.shape)

necessary_columns = df[
    ["Patient ID", "filename", "Recording location", "SR", "Murmur", "Outcome"]
]

df = df.drop(
    columns=["Patient ID", "filename", "Recording location", "SR", "Murmur", "Outcome"]
)
corr = df.corr().abs()
"""
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)
plt.title("Correlation Matrix")
plt.show()
"""

# Create a mask to identify highly correlated features
mask = np.triu(np.ones_like(corr, dtype=bool))

# Exclude diagonal elements and select only upper triangular matrix
upper_triangle = corr.where(mask)

# Find features with correlation above threshold (e.g., 0.8)
# Adjust the threshold as needed
threshold = 0.8
highly_correlated_features = [
    column
    for column in upper_triangle.columns
    if any(upper_triangle[column] > threshold)
]

features_to_keep = []
for feature in highly_correlated_features:
    correlated_group = list(upper_triangle.index[upper_triangle[feature] > threshold])
    if feature in correlated_group:
        correlated_group.remove(feature)
    if correlated_group:
        features_to_keep.append(correlated_group[0])

# Remove highly correlated features from the dataset
data_filtered = df.drop(columns=highly_correlated_features)

# Concatenate the features to keep with the filtered dataset
data_filtered = pd.concat([data_filtered, df[features_to_keep]], axis=1)

data_filtered = data_filtered.loc[:, ~data_filtered.columns.duplicated()]

print(data_filtered.shape)

print("columns\n", data_filtered.columns)

# number of duplicates in columns
print(data_filtered.columns.duplicated().sum())
print(data_filtered.columns.duplicated())
# which columns are duplicates
print(data_filtered.columns[data_filtered.columns.duplicated()])

corr = data_filtered.corr()
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)
plt.title("Correlation Matrix")
plt.show()

# add filtered columns to necessary columns
data_filtered = pd.concat([necessary_columns, data_filtered], axis=1)

# Replace 'filtered_data.csv' with your desired file name
data_filtered.to_csv("data/filtered_data.csv", index=False)
