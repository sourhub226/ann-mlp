# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler

# # Load the CSV dataset using pandas
# input_csv_file = "datasets\mnist.csv"
# output_csv_file = "normalized_mnist.csv"
# df = pd.read_csv(input_csv_file)

# # Extract the values from the DataFrame
# data = df.values

# # Initialize the Min-Max scaler
# scaler = MinMaxScaler()

# # Fit the scaler to the data and transform the data to be between 0 and 1
# normalized_data = scaler.fit_transform(data)

# # Create a new DataFrame with the normalized data
# normalized_df = pd.DataFrame(normalized_data, columns=df.columns)

# # Save the normalized dataset to a new CSV file
# normalized_df.to_csv(output_csv_file, index=False)

# print(f"Normalized dataset saved to {output_csv_file}")


import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load your dataset
data = pd.read_csv("datasets\mnist.csv")

# Handle NaN values (you can choose either removal or imputation)
data.dropna(inplace=True)  # Remove rows with NaN values
# OR
# data.fillna(data.mean(), inplace=True)  # Impute NaN values with mean

# Feature scaling (you can choose different scaling methods)
scaler = StandardScaler()
data.iloc[:, 1:] = scaler.fit_transform(data.iloc[:, 1:])  # Exclude the target column

# Save the preprocessed dataset to a new file
data.to_csv("preprocessed_mnist.csv", index=False)
