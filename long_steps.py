import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

#set stepsize patterns
H_GRDESC = [1]
H_2 = [2.9, 1.5] # taking nu = 0.1
H_3 = [1.5, 4.9, 1.5]
H_7 = [1.5, 2.2, 1.5, 12.0, 1.5, 2.2, 1.5]
H_15 = [1.4, 2.0, 1.4, 4.5, 1.4, 2.0, 1.4, 29.7, 1.4, 2.0, 1.4, 4.5, 1.4, 2.0, 1.4]
H_31 = [1.4, 2.0, 1.4, 3.9, 1.4, 2.0, 1.4, 8.2, 1.4, 2.0, 1.4, 3.9, 1.4, 2.0, 1.4, 72.3, 1.4, 2.0, 1.4, 3.9, 1.4, 2.0, 1.4, 8.2, 1.4, 2.0, 1.4, 3.9, 1.4, 2.0, 1.4]
H_63 = [1.4, 2.0, 1.4, 3.9, 1.4, 2.0, 1.4, 7.2, 1.4, 2.0, 1.4, 3.9, 1.4, 2.0, 1.4, 14.2, 1.4, 2.0, 1.4, 3.9, 1.4, 2.0, 1.4, 7.2, 1.4, 2.0, 1.4, 3.9, 1.4, 2.0, 1.4, 164.0, 1.4, 2.0, 1.4, 3.9, 1.4, 2.0, 1.4, 7.2, 1.4, 2.0, 1.4, 3.9, 1.4, 2.0, 1.4, 14.2, 1.4, 2.0, 1.4, 3.9, 1.4, 2.0, 1.4, 7.2, 1.4, 2.0, 1.4, 3.9, 1.4, 2.0, 1.4]
H_127 = [1.4, 2.0, 1.4, 3.9, 1.4, 2.0, 1.4, 7.2, 1.4, 2.0, 1.4, 3.9, 1.4, 2.0, 1.4, 12.6, 1.4, 2.0, 1.4, 3.9, 1.4, 2.0, 1.4, 7.2, 1.4, 2.0, 1.4, 3.9, 1.4, 2.0, 1.4, 23.5, 1.4, 2.0, 1.4, 3.9, 1.4, 2.0, 1.4, 7.2, 1.4, 2.0, 1.4, 3.9, 1.4, 2.0, 1.4, 12.6, 1.4, 2.0, 1.4, 3.9, 1.4, 2.0, 1.4, 7.2, 1.4, 2.0, 1.4, 3.9, 1.4, 2.0, 1.4, 370.0, 1.4, 2.0, 1.4, 3.9, 1.4, 2.0, 1.4, 7.2, 1.4, 2.0, 1.4, 3.9, 1.4, 2.0, 1.4, 12.6, 1.4, 2.0, 1.4, 3.9, 1.4, 2.0, 1.4, 7.2, 1.4, 2.0, 1.4, 3.9, 1.4, 2.0, 1.4, 23.5, 1.4, 2.0, 1.4, 3.9, 1.4, 2.0, 1.4, 7.5, 1.4, 2.0, 1.4, 3.9, 1.4, 2.0, 1.4, 12.6, 1.4, 2.0, 1.4, 3.9, 1.4, 2.0, 1.4, 7.2, 1.4, 2.0, 1.4, 3.9, 1.4, 2.0, 1.4]

H = [H_GRDESC,H_2,H_3,H_7,H_15,H_31,H_63,H_127]
H = [H_GRDESC,H_2]

#evaluate objective function without regularizer
def objective1(outputs, labels, model = None):
    # Convert labels to one-hot encoding
    labels_onehot = torch.zeros_like(outputs)
    for i, label in enumerate(labels):
        labels_onehot[i][label] = 1
    
    return torch.norm(outputs - labels_onehot, p=2)**2

#...with regularizer
def objective2(outputs, labels, model):
    return objective1(outputs, labels, model) + torch.norm(model.linear.weight, p=2)**2


# Define the linear model
class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(784, 10, bias=False) # 28x28 = 784 input features, 10 output classes

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input tensor
        return self.linear(x)

    # Define the custom non-constant step sizes
    def step_size_schedule(self, epoch, H, image = None):
        h = H[(epoch-1) % len(H)]
        # L = 2 * torch.norm(image.view(image.size(0), -1), p=2)**2
        return h / 1500

   
    
# Load the MNIST dataset
train_dataset = MNIST(root='./data', train=True, transform=ToTensor(), download=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True) # Batch size of 1
print(f"Number of training examples: {len(train_dataset)}")

# Load MNIST test data
test_dataset = MNIST(root='./data', train=False, download=True, transform=ToTensor())
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

losses_H = []
test_errors_H = []
for H_i in H:
    criterion =objective1 
    # Create the linear model
    model = LinearModel()
    nn.init.xavier_uniform_(model.linear.weight) # Initialize the weights using Xavier initialization
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    # Lists to store the loss values and test errors
    losses = []
    test_errors = []

    # Train the model
    for epoch in tqdm(range(30)):
        for image, label in train_loader:
            optimizer.zero_grad()
            # Update the learning rate
            optimizer.param_groups[0]['lr'] = model.step_size_schedule(epoch, H_i, image)

            output = model(image)
            loss = criterion(output, label, model) # x is weight, A is image data, b is label data, Ax=b
            loss.backward()
            # Add gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Record the loss
            losses.append(loss.item())

        # Test the model
        with torch.no_grad():
            for inputs, labels in test_loader:
                # Flatten the images
                inputs = inputs.view(inputs.size(0), -1)

                y_pred = model(inputs)
                mse = criterion(y_pred, labels, model)

                # print(f"\nTest Error: {mse.item():.4f}")

                test_errors.append(mse.item())

        # print(f"Epoch {epoch+1}: Training Loss = {loss.item()}")
        
    losses_H.append(losses)
    test_errors_H.append(test_errors)

# Save the loss and test error data
np.save('losses_H.npy', losses_H)
np.save('test_errors_H.npy', test_errors_H)

# Load the loss and test error data
losses_H = np.load('losses_H.npy')
test_errors_H = np.load('test_errors_H.npy')

# Plot losses_H
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
for i, losses in enumerate(losses_H):
    plt.plot(losses, label=f"Length of H: {len(H[i])}")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.legend()

# Plot test_errors_H
plt.subplot(1, 2, 2)
for i, test_errors in enumerate(test_errors_H):
    plt.plot(test_errors, label=f"Length of H: {len(H[i])}")
plt.xlabel("Iterations")
plt.ylabel("Test Error")
plt.title("Test Error")
plt.legend()

# Show the plot
plt.tight_layout()
plt.show()
