import torch

from src.nn.brute_force_data import generate_dataset_ct
from src.nn.preprocess_data import PreprocessingType


def generate_data(n_qubits, n_gates, labels_as_described:bool, preprocessing_type:PreprocessingType):
    X_train, Y_train = generate_dataset_ct(1, n_qubits, n_gates,labels_as_described=labels_as_described, preprocessing_type=preprocessing_type)
    X_val, Y_val = generate_dataset_ct(1, n_qubits, n_gates,labels_as_described=labels_as_described, preprocessing_type=preprocessing_type)

    torch.save((X_train, Y_train), f'train_data_{labels_as_described}_{preprocessing_type.value}.pt')
    torch.save((X_val, Y_val), f'val_data_{labels_as_described}_{preprocessing_type.value}.pt')
    print("Successfully generated example data!")


#def main():
#    X_train, Y_train = torch.load('train_data.pt')
#    X_val, Y_val = torch.load('val_data.pt')
#    print(X_train.shape, Y_train.shape)
#    print(X_val.shape, Y_val.shape)


if __name__ == '__main__':
    labels_as_described = False
    preprocessing_type = PreprocessingType.ORIGINAL
    n_qubits = [4]
    n_gates_per_circuit = [1000]
    # The code is written so you can make multiple datasets at once
    # so you can define multiple n_qubits, and multiple n_gates
    # However, I believe they are all gathered in the same Tensor stack so they need to be the same number.
    # Feel free to fix this as need arises.
    generate_data(n_qubits, n_gates_per_circuit, labels_as_described, preprocessing_type)
    #main()
