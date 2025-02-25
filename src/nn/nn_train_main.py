import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from src.nn.best_qubit_model import BestQubitModel

def load_data(train_path, val_path):
    """
    Loads the training and validation data.
    
    Args:
        train_path (str): Path to the training data file.
        val_path (str): Path to the validation data file.
    
    Returns:
        (Tensor, Tensor, Tensor, Tensor): X_train, y_train, X_val, y_val
    """
    train_data = torch.load(train_path)
    val_data = torch.load(val_path)
    X_train, y_train = train_data
    X_val, y_val = val_data
    return X_train, y_train, X_val, y_val

def create_dataloaders(X_train, y_train, batch_size=32):
    """
    Creates the training DataLoader.
    
    Args:
        X_train (Tensor): Training inputs
        y_train (Tensor): Training targets
        batch_size (int): Batch size for DataLoader
    
    Returns:
        DataLoader: A DataLoader for training data
    """
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader

def save_loss_plot(train_losses, val_losses, filename="training_loss.png"):
    """
    Saves the training and validation loss plot as a PNG file.

    Args:
        train_losses (list of float): List containing the training loss value per epoch.
        val_losses (list of float): List containing the validation loss value per epoch.
        filename (str): Filename for the saved plot (default: training_loss.png).
    """
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Progression")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def log_experiment_details(filename, model, optimizer, best_train_loss, best_val_loss, n_epochs, patience):
    """
    Logs the experiment details to a text file.

    Args:
        filename (str): Path to the log file.
        model (nn.Module): The model being trained.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        best_train_loss (float): The best training loss achieved.
        best_val_loss (float): The best validation loss achieved.
        n_epochs (int): Number of epochs the model was trained for.
        patience (int): Patience for early stopping.
    """
    with open(filename, 'a') as f:
        f.write(f"Model: {model}\n")
        f.write(f"Number of hidden layers: {model.hidden_layers}\n")
        f.write(f"Optimizer: {optimizer}\n")
        f.write(f"Number of epochs: {n_epochs}\n")
        f.write(f"Patience: {patience}\n")
        f.write(f"Best training loss: {best_train_loss:.4f}\n")
        f.write(f"Best validation loss: {best_val_loss:.4f}\n")
        f.write("\n" + "="*80 + "\n\n")

def train_model(model, train_loader, criterion, optimizer, X_train, y_train, X_val, y_val, n_epochs=30000, verbose=True, patience=1000, log_file="experiment_log.txt"):
    """
    Main training loop for the model.
    
    Args:
        model (nn.Module): Neural network model.
        train_loader (DataLoader): DataLoader for training data.
        criterion (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        X_train (Tensor): Training inputs for occasional sample prediction.
        y_train (Tensor): Training targets for occasional sample comparison.
        X_val (Tensor): Validation inputs.
        y_val (Tensor): Validation targets.
        n_epochs (int): Number of epochs to train.
        verbose (bool): If True, prints updates to terminal.
        log_file (str): Path to the log file.
    
    Returns:
        None.
    """
    train_losses = []
    val_losses = []
    best_train_loss = float('inf')
    best_val_loss = float('inf')
    epochs_no_improve = 0  # Counter for early stopping

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Compute average training loss for this epoch
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # Evaluate on validation set
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val).item()
        val_losses.append(val_loss)

        if verbose and epoch % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch}, Training Loss: {avg_loss:.4f}, '
                  f'Validation Loss: {val_loss:.4f}, LR: {current_lr:.6f}')
            with torch.no_grad():
                test_input = X_train[0:1]
                pred = model(test_input)
                print("Predicted values:")
                print(pred[0, 0])
                print("Actual values:")
                print(y_train[0, 0])

        # Save only the best model so far and check early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_train_loss = avg_loss
            torch.save(model.state_dict(), "best_qubit_model_weights.pt")
            epochs_no_improve = 0
            
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch} epochs.")
            break

    save_loss_plot(train_losses, val_losses)
    model.load_state_dict(torch.load("best_qubit_model_weights.pt", weights_only=True))

    # Log experiment details when a new best validation loss is achieved
    log_experiment_details(log_file, model, optimizer, best_train_loss, best_val_loss, epoch, patience)

def main():
    # File paths
    train_path = 'train_data_True_from_project_description.pt'
    val_path = 'val_data_True_from_project_description.pt'
    
    # Load data
    X_train, y_train, X_val, y_val = load_data(train_path, val_path)
    
    # Create model, criterion, optimizer
    model = BestQubitModel(n_size=4, hidden_layers=4, hidden_size=128, dropout_rate=0.3)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)
    
    # Create data loader
    train_loader = create_dataloaders(X_train, y_train, batch_size=32)
    
    # Train model with validation
    train_model(
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        n_epochs=20000,
        verbose=True
    )

if __name__ == "__main__":
    main()