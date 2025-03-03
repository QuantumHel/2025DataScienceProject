import torch

# Load the data from the file
train_data = torch.load('train_data_True_from_project_description.pt')

# Assuming train_data is a tuple of two tensors
X, Y = train_data

# Print the shapes of the tensors to understand their structure
print("Shape of X:", X.shape)
print("Shape of Y:", Y.shape)

# Print the first example of X and Y to see the actual values
print("First example of X:", X[0])
print("First example of Y:", Y[0])

# Flatten the Y tensor to a 1D tensor
Y_flat = Y.flatten()

# Count the frequency of each value in the Y tensor
value_counts = torch.bincount(Y_flat.long())

# Print the frequency of each value
for value, count in enumerate(value_counts):
    print(f"Value {value}: {count} occurrences")

# Count how many matrices have at least half of the values other than zero
half_non_zero_count = 0
total_elements = Y.shape[2] * Y.shape[3]  # Assuming Y has shape [batch_size, channels, height, width]
half_threshold = total_elements // 2
total_matrices = Y.shape[0]  # Total number of matrices

for i in range(total_matrices):
    non_zero_count = torch.count_nonzero(Y[i])
    if non_zero_count >= half_threshold:
        half_non_zero_count += 1

print(f"Number of matrices with at least half of the values other than zero: {half_non_zero_count} out of {total_matrices}")