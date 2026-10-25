from dataset import HipMRI3D, HipMRI2D, get_training_test_split
from modules import CAN3D, CAN2D, compute_loss, compute_loss_2d
import torch
import torch.nn as nn
import time

from torch.utils.data import DataLoader
import torch.optim as optim

import monai.transforms as mt


augmentations = mt.Compose([
    mt.Rand3DElasticd(keys=["image", "label"], prob=1.0, mode=["bilinear", "nearest"], sigma_range=(4, 6), magnitude_range=(50, 130), padding_mode="zeros"),
    mt.RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5),
])
def transform_training_data(images, labels):
    #Note: extra dimension it technically the batch, but its size is 1, so it may as well be channel
    augmented = augmentations({"image": images, "label": labels})
    return augmented['image'], augmented['label']


def train_3d():
    # Hyper Parameters
    batch_size = 8
    learning_rate = 0.01
    num_epochs = 100

    # Create Database
    database = HipMRI3D(early_stop=30)
    train_set, test_set = get_training_test_split(database)
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    # Select Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device: ", device)

    # Train Models
    model = CAN3D()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    total_steps = (len(train_loader) * num_epochs) // batch_size
    schedular = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, total_steps=total_steps)

    start = time.time()
    model.train()

    weights = torch.tensor([0.03, 0.03, 0.6, 0.3, 8.0, 12.0]).to(device)

    steps = 0
    optimizer.zero_grad()
    for epoch in range(num_epochs):
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            # Load Data
            images = images.to(device)
            labels = labels.to(device)
            images, labels = transform_training_data(images, labels)
            labels = labels.long()

            # Evaluate and Backpropogate
            predictions = model(images)
            loss = compute_loss(predictions, labels, weights, weights * 3, gamma=2) / batch_size
            loss.backward()

            if (steps + 1) % batch_size == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                schedular.step()
            steps += 1

            if i == 0 and epoch == 0:
                print("First Loss: {:.5f}".format(loss.item() * batch_size), flush=True)

        # Save the model every tength epoch
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), 'parameters_3d_epoch_{}.pth'.format(epoch))

        print("Epoch [{}/{}],  Loss: {:.5f}, Time: {} secs"
            .format(epoch+1, num_epochs, loss.item(), time.time() - start), flush=True)

    torch.save(model.state_dict(), 'final_parameters_3d.pth')

def train_2d():
    # Hyper Parameters
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 300

    # Create Database
    database = HipMRI2D(early_stop=500)
    train_loader = DataLoader(database, batch_size=batch_size, shuffle=True)

    # Select Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device: ", device)

    # Train Models
    model = CAN2D()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    steps_per_epoch = len(train_loader)
    schedular = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, epochs=num_epochs, steps_per_epoch=steps_per_epoch)

    start = time.time()
    model.train()

    weights = torch.tensor([0.16, 0.16, 0.16, 0.16, 0.16, 0.16]).to(device)

    optimizer.zero_grad()
    for epoch in range(num_epochs):
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()

            # Load Data
            images = images.to(device)
            labels = labels.to(device).long()

            # Evaluate and Backpropogate
            predictions = model(images)
            loss = compute_loss_2d(predictions, labels, weights, weights * 6, gamma=2)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            schedular.step()


            print("Loss: ", loss.item())

            if i == 0 and epoch == 0:
                print("First Loss: {:.5f}".format(loss.item() * batch_size), flush=True)

        # Save the model every tenth epoch
        if (epoch + 1) % 100 == 0:
            torch.save(model.state_dict(), 'parameters_2d_epoch_{}.pth'.format(epoch))

        print("Epoch [{}/{}],  Loss: {:.5f}, Time: {} secs"
            .format(epoch+1, num_epochs, loss.item(), time.time() - start), flush=True)

    torch.save(model.state_dict(), 'final_parameters_2d.pth')

#train_2d()
train_3d()