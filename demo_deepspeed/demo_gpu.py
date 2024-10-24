import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time

# Set up a basic linear regression model
class SimpleModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

# Create a synthetic dataset
def generate_data(num_samples, input_dim):
    X = np.random.rand(num_samples, input_dim)  # Generate random input data
    y = np.random.rand(num_samples, 1)          # Generate random target data
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# Training script
def train(model, data_loader, criterion, optimizer, num_epochs, device):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        start_time = time.time()

        for i, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_time = time.time() - start_time
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(data_loader):.4f}, Time: {epoch_time:.2f}s")

def main():
    input_dim = 10
    output_dim = 1
    num_samples = 1000
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.001

    # Create model, criterion and optimizer
    model = SimpleModel(input_dim, output_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Generate data
    X, y = generate_data(num_samples, input_dim)
    dataset = torch.utils.data.TensorDataset(X, y)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f"Using device: {device}")

    # Train the model
    train(model, data_loader, criterion, optimizer, num_epochs, device)

if __name__ == "__main__":
    main()

# python demo_gpu.py
# CUDA_VISIBLE_DEVICES=0 python demo_gpu.py
