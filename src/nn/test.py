import torch

def load_and_print_first_five_elements(file_path):
    # Load the tensor from the file
    data = torch.load(file_path)
    
    # Print the first 5 elements
    print(data[1][:5])

# Example usage
file_path = 'train_data_True_from_project_description.pt'
load_and_print_first_five_elements(file_path)