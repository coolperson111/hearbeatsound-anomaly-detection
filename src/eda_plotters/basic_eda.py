import pandas as pd

physionet_path = "./datasets/physionet/the-circor-digiscope-phonocardiogram-dataset-1.0.3/training_data.csv"

physionet_df = pd.read_csv(physionet_path)

# print(physionet_df.head())
print("shape: ", physionet_df.shape)
print()

# print number of nans in each column
# to see which columns can and can't be used, which need interpolation, etc
print("nans: ")
print(physionet_df.isna().sum())
print()

print("values in 'Murmur' column: ", physionet_df["Murmur"].value_counts())
print("values in 'Outcome' column: ", physionet_df["Outcome"].value_counts())
# wtf 70 pregnant kids??!
print("'pregnancy' column: ", physionet_df["Pregnancy status"].value_counts())
print("values in 'Age' column: ", physionet_df["Age"].value_counts())
