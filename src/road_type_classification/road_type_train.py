import os

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

from road_type_dataloader import get_dataloader
from road_type_cnn_classifier import RoadTypeCNNClassifier

model = RoadTypeCNNClassifier()

TRAIN_DIR_PATH = os.path.join(os.path.dirname(__file__), '..' , 'data', 'train')
TEST_DIR_PATH = os.path.join(os.path.dirname(__file__), '..' , 'data', 'test')

BATCH_SIZE = 16
NUM_WORKERS = 2
NUM_EPOCHS = 10
INIT_LR = 1e-4
LR_PATIENCE = 5

MODEL_SAVE_DIR: str = os.path.join(os.path.dirname(__file__), 'saved_model')
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = get_dataloader(TRAIN_DIR_PATH, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    test_loader = get_dataloader(TEST_DIR_PATH, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    model = model.to(device)
    criterion = nn.BCELoss()  # Binary cross entropy loss for binary classification
    optimizer = torch.optim.Adam(model.parameters(), lr=INIT_LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=LR_PATIENCE)
    # This scheduler reduces the learning rate by a factor of 0.1 if the validation loss does not improve for LR_PATIENCE epochs

    # Metrics Storage
    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []

    best_val_loss = float('inf')

    # Training Loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_train_loss = 0  # Total loss over the epoch
        epoch_train_correct = 0  # Number of correct predictions
        total_train_samples = 0  # Total number of samples

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} - Training"):
            inputs, labels = inputs.to(device), labels.to(device).float().view(-1, 1) # Reshape labels to (batch_size, 1)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item() * inputs.size(0)  # Sum losses over batch

            predicted = (outputs > 0.5).float()
            epoch_train_correct += (predicted == labels).sum().item()
            total_train_samples += labels.size(0)

        avg_train_loss = epoch_train_loss / total_train_samples  # Average loss
        train_accuracy = epoch_train_correct / total_train_samples  # Accuracy. TP + TN / Total samples
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)

        # Validation Loop
        model.eval()
        epoch_val_loss = 0
        epoch_val_correct = 0
        total_val_samples = 0

        with torch.no_grad():  # No need to compute gradients for validation because we are not training. The benefits are speed and memory savings
            for inputs, labels in tqdm(test_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} - Validation"):
                inputs, labels = inputs.to(device), labels.to(device).float().view(-1, 1)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                epoch_val_loss += loss.item() * inputs.size(0)
                predicted = (outputs > 0.5).float()
                epoch_val_correct += (predicted == labels).sum().item()
                total_val_samples += labels.size(0)

        avg_val_loss = epoch_val_loss / total_val_samples
        val_accuracy = epoch_val_correct / total_val_samples
        test_losses.append(avg_val_loss)
        test_accuracies.append(val_accuracy)

        # Print epoch summary
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS} | "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

        # Adjust Learning Rate
        scheduler.step(avg_val_loss)

        # Save Best Model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = os.path.join(MODEL_SAVE_DIR, 'road_type_classifier_best.pth')
            torch.save(model.state_dict(), best_model_path)  # we save state_dict instead of the model itself because it is more memory efficient.
            # if you want to load the model later, you will need to create an instance of the model and load the state_dict
            # If you want to, you can save the entire model using torch.save(model, path) and load it
            # using torch.load(path) without creating an instance of the model but it is not memory efficient
            print(f"Best model saved at epoch {epoch + 1}")

    # Save Final Model
    final_model_path = os.path.join(MODEL_SAVE_DIR, 'road_type_classifier_final.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved at {final_model_path}")

    # Plot Losses
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, NUM_EPOCHS + 1), train_losses, label="Train Loss", marker='o')
    plt.plot(range(1, NUM_EPOCHS + 1), test_losses, label="Validation Loss", marker='o')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid()
    plt.show()

    # Plot Accuracies
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, NUM_EPOCHS + 1), train_accuracies, label="Train Accuracy", marker='o')
    plt.plot(range(1, NUM_EPOCHS + 1), test_accuracies, label="Validation Accuracy", marker='o')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.grid()
    plt.show()


