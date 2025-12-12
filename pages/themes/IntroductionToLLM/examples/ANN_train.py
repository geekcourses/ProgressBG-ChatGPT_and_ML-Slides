# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader # Import for batching
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

# --- 1. DATA PREPARATION ---

# Example dataset creation (1000 samples, 8 features, 2 classes)
X, y = make_classification(n_samples=1000, n_features=8, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
# Ensure y_train is float and correct shape for BCELoss
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# --- 2. IMPLEMENT BATCHING ---

BATCH_SIZE = 64 # Define a reasonable batch size
NUM_EPOCHS = 100 # Increase epochs to allow convergence

# Create TensorDatasets
train_data = TensorDataset(X_train, y_train)
test_data = TensorDataset(X_test, y_test)

# Create DataLoaders for efficient iteration
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE)


# --- 3. MODEL DEFINITION ---

class SimpleANN(nn.Module):
    def __init__(self):
        super(SimpleANN, self).__init__()
        self.hidden = nn.Linear(8, 32)
        self.output = nn.Linear(32, 1)

    def forward(self, x):
        # ReLU activation for hidden layer (non-linearity)
        x = torch.relu(self.hidden(x))
        # Sigmoid activation for binary classification output
        x = torch.sigmoid(self.output(x))
        return x


# --- 4. INITIALIZATION ---

model = SimpleANN()
criterion = nn.BCELoss()  # Binary Cross-Entropy loss for sigmoid output
optimizer = optim.Adam(model.parameters(), lr=0.001)


# --- 5. TRAIN THE MODEL (With Batching) ---

print(f"Starting training for {NUM_EPOCHS} epochs...")
for epoch in range(NUM_EPOCHS):
    model.train() # Set model to training mode
    current_loss = 0.0

    # Iterate over the data in batches
    for inputs, targets in train_loader:
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + Backward + Optimize
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward() # Backpropagation
        optimizer.step() # Update weights

        current_loss += loss.item() * inputs.size(0)

    epoch_loss = current_loss / len(X_train)

    # Print less often for longer training
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Loss: {epoch_loss:.4f}")


# --- 6. EVALUATE THE MODEL ---

correct = 0
total = 0
model.eval() # Set model to evaluation mode

with torch.no_grad():
    for inputs, targets in test_loader:
        test_outputs = model(inputs)
        # Convert sigmoid output (0 to 1) to binary prediction (0 or 1)
        predicted = (test_outputs >= 0.5).float()

        total += targets.size(0)
        correct += (predicted.eq(targets).sum().item())

accuracy = correct / total

print("\n--- Final Results ---")
print(f"Total training samples: {len(X_train)}")
print(f"Total test samples: {len(X_test)}")
print(f"Test Accuracy: {accuracy * 100:.2f}%")