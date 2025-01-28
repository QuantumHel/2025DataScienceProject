import torch

from src.nn.brute_force_data import generate_dataset_ct


def generate_data():
    X_train, Y_train = generate_dataset_ct(1, [4], [1000])
    X_val, Y_val = generate_dataset_ct(1, [4], [1000])

    torch.save((X_train, Y_train), 'train_data.pt')
    torch.save((X_val, Y_val), 'val_data.pt')
    print("Successfully generated example data!")


def main():
    X_train, Y_train = torch.load('train_data.pt')
    X_val, Y_val = torch.load('val_data.pt')
    print(X_train.shape, Y_train.shape)
    print(X_val.shape, Y_val.shape)


if __name__ == '__main__':
    generate_data()
    main()
