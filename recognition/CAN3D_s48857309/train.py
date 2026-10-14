from dataset import HipMRI3D, get_training_test_split
from modules import CAN3D, compute_loss
import torch
import time

from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt

# Hyper Parameters
batch_size = 1 # since cant store large memory amount
learning_rate = 0.001
num_epochs = 10

# Create Database
database = HipMRI3D(early_stop=10)
train_set, test_set = get_training_test_split(database)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# Select Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device: ", device)

# Train Models
model = CAN3D(256, 256, 128, 6)
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
steps_per_epoch = len(train_loader)

start = time.time()
model.train()
for epoch in range(num_epochs):
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()

        # Load Data (TODO: add transformations)
        images = images.to(device)
        labels = labels.to(device).long()

        # Evaluate and Backpropogate
        predictions = model(images)
        loss = compute_loss(predictions, labels)

        loss.backward()
        optimizer.step()
        if i == 0 and epoch == 0:
            print("First Loss: {:.5f}".format(loss.item()), flush=True)

    # Save the model every third epoch
    if (epoch + 1) % 3 == 0:
        torch.save(model.state_dict(), 'parameters_epoch_{}.pth'.format(epoch))

    print("Epoch [{}/{}],  Loss: {:.5f}, Time: {} secs"
          .format(epoch+1, num_epochs, loss.item(), time.time() - start), flush=True)

torch.save(model.state_dict(), 'final_parameters.pth')
