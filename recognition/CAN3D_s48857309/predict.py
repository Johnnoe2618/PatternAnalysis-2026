from dataset import HipMRI3D, get_training_test_split
from modules import CAN3D, compute_loss, calculate_descrete_dsc
import torch
import time
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader


# Create Database
database = HipMRI3D(early_stop=5, images_path="../data/images/", labels_path="../data/semantic/")
train_set, test_set = get_training_test_split(database)
train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False)



# Select Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device: ", device)

# Train Models
model = CAN3D(256, 256, 128, 6)
model = model.to(device)
state_dict = torch.load('final_parameters_3d.pth', weights_only=True, map_location=device)
model.load_state_dict(state_dict)
model.eval()

# Do visualisation of the first few things
print("Visualising...")

with torch.no_grad():
    image, label = next(iter(train_loader))
    image = image.to(device)

    plt.subplot(1, 3, 1)
    plt.imshow(image.cpu().numpy()[0, :, :, 64])

    plt.subplot(1, 3, 2)
    plt.imshow(label.cpu().numpy()[0, :, :, 64])

    predictions = torch.argmax(model(image), dim=1)

    plt.subplot(1, 3, 3)
    plt.imshow(predictions.cpu().numpy()[0, :, :, 64])

plt.savefig("prediction_example.png")

# Perform all tests.

print("Testing...")

number_of_tests = len(test_loader)
total_sum = torch.zeros((6), device=device)
total_validation_samples = 0

with torch.inference_mode():
    for i, (image, label) in enumerate(test_loader):
        print("{}/{} Test Batches".format(i+1, number_of_tests))
        image = image.to(device)
        label = label.to(device).long()

        predictions = model(image)

        sum_dice_per_class, no_in_batch = calculate_descrete_dsc(predictions, label)
        total_sum += sum_dice_per_class
        total_validation_samples += no_in_batch

total_sum = total_sum / total_validation_samples

print("Test Scores: ", total_sum)