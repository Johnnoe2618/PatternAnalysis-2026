from dataset import HipMRI3D, get_training_test_split
from modules import CAN3D, CAN2D, compute_loss, calculate_descrete_dsc, calculate_intersection_over_union
import torch
import time
import matplotlib.pyplot as plt
import monai.transforms as mt

from torch.utils.data import DataLoader


# Create Database
database = HipMRI3D(early_stop=0)
train_set, val_set, test_set = get_training_test_split(database, test_size=0.2, val_size=0.1)
train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
val_loader = DataLoader(val_set, batch_size=1, shuffle=False)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

# Select Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device: ", device)

def predict_3d():
    # Load Models
    model = CAN3D()
    model = model.to(device)
    state_dict = torch.load('final_parameters_3d.pth', weights_only=True, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Do visualisation of the first few things
    print("Visualising...")

    with torch.no_grad():
        images, labels = next(iter(train_loader))
        images = images.to(device)
        labels = labels.to(device).long()


        plt.subplot(1, 3, 1)
        plt.imshow(images.cpu().numpy()[0, :, 128, :])

        plt.subplot(1, 3, 2)
        plt.imshow(labels.cpu().numpy()[0, :, 128, :])

        output = model(images)
        predictions = torch.argmax(output, dim=1)
        plt.subplot(1, 3, 3)
        plt.imshow(predictions.cpu().numpy()[0, :, 128, :])

    plt.savefig("prediction_example_3d.png")

    # Perform all tests.

    print("Testing...")

    number_of_tests = len(test_loader)
    total_sum_dice = torch.zeros((6), device=device)
    total_sum_IoU = torch.zeros((6), device=device)
    total_validation_samples = 0

    with torch.inference_mode():
        for i, (image, label) in enumerate(test_loader):
            print("{}/{} Test Batches".format(i+1, number_of_tests))
            image = image.to(device)
            label = label.to(device).long()

            output = model(image)

            sum_dice_per_class, no_in_batch = calculate_descrete_dsc(output, label)
            sum_IoU_per_class, _ = calculate_intersection_over_union(output, label)
            total_sum_dice += sum_dice_per_class
            total_sum_IoU += sum_IoU_per_class
            total_validation_samples += no_in_batch

    total_sum_dice = total_sum_dice / total_validation_samples
    total_sum_IoU = total_sum_IoU / total_validation_samples

    print("Test Scores: Dice: ", total_sum_dice, "IoU: ", total_sum_IoU)

def volume_to_2d_batch(volume : torch.Tensor):
    # take height-wise slices
    return volume.permute(1, 0, 2)

def batch_output_to_3d_volume_batch(batch : torch.Tensor):
    return batch.permute(1, 2, 0, 3)[None,:,:,:,:]

def predict_2d():
    # Load Models
    model = CAN2D()
    model = model.to(device)
    state_dict = torch.load('final_parameters_2d.pth', weights_only=True, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Do visualisation of the first few things
    print("Visualising...")

    with torch.no_grad():
        images, labels = next(iter(train_loader))
        images = images.to(device)
        labels = labels.to(device).long()

        plt.subplot(2, 2, 1)
        plt.imshow(images.cpu().numpy()[0, :, 128, :])

        plt.subplot(2, 2, 2)
        plt.imshow(labels.cpu().numpy()[0, :, 128, :])

        output = batch_output_to_3d_volume_batch(model(volume_to_2d_batch(images[0])))
        predictions = torch.argmax(output, dim=1)

        plt.subplot(2, 2, 3)
        plt.imshow(output.cpu().numpy()[0, 1, :, 128, :])

        plt.subplot(2, 2, 4)
        plt.imshow(predictions.cpu().numpy()[0, :, 128, :])

    plt.savefig("prediction_example_2d.png")

    # Perform all tests.

    print("Testing...")

    number_of_tests = len(test_loader)
    total_sum_dice = torch.zeros((6), device=device)
    total_sum_IoU = torch.zeros((6), device=device)
    total_validation_samples = 0

    with torch.inference_mode():
        for i, (image, label) in enumerate(test_loader):
            print("{}/{} Test Batches".format(i+1, number_of_tests))
            image = image.to(device)
            label = label.to(device).long()

            output = batch_output_to_3d_volume_batch(model(volume_to_2d_batch(images[0])))

            sum_dice_per_class, no_in_batch = calculate_descrete_dsc(output, label)
            sum_IoU_per_class, _ = calculate_intersection_over_union(output, label)
            total_sum_dice += sum_dice_per_class
            total_sum_IoU += sum_IoU_per_class
            total_validation_samples += no_in_batch

    total_sum_dice = total_sum_dice / total_validation_samples
    total_sum_IoU = total_sum_IoU / total_validation_samples

    print("Test Scores: Dice: ", total_sum_dice, "IoU: ", total_sum_IoU)

predict_3d()