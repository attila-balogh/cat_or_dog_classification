import glob

from PIL import ImageStat

import torch
from torch.utils.data import random_split

import torchvision.transforms as transforms

from datasets.MergeDataset import MergeDataset
from datasets.CatVsDogDataset import CatVsDogDataset


cat_path = glob.glob("../catordog/images_data/PetImages/Cat/*.jpg")
dog_path = glob.glob("../catordog/images_data/PetImages/Dog/*.jpg")


# There are two corrupted images in the dataset
cat_path.remove("../catordog/images_data/PetImages/Cat\\666.jpg")
dog_path.remove("../catordog/images_data/PetImages/Dog\\11702.jpg")


print(f"'Cat' images: \t{len(cat_path):6,}")
print(f"'Dog' images: \t{len(dog_path):6,}")
print()


# 1500 of images is separated from training set -> test set
# 2000 of images is separated from training set -> validation set

val_size = 2000
test_size = 1500
train_size = len(cat_path) - val_size - test_size


cat_train_path, cat_val_path, cat_test_path = random_split(cat_path, [train_size, val_size, test_size])
dog_train_path, dog_val_path, dog_test_path = random_split(dog_path, [train_size, val_size, test_size])


print("\t\t\t\t\t\t Cats + Dogs")
print(f"TRAINING set size: \t\t{train_size:5,} + {train_size:5,}")
print(f"VALIDATION set size: \t{val_size:5,} + {val_size:5,}")
print(f"TEST set size: \t\t\t{test_size:5,} + {test_size:5,}")


# Creating labels (0-cat, 1-dog)
cat_train_labels = torch.zeros(train_size)
cat_val_labels = torch.zeros(val_size)
cat_test_labels = torch.zeros(test_size)

dog_train_labels = torch.ones(train_size)
dog_val_labels = torch.ones(val_size)
dog_test_labels = torch.ones(test_size)

# Concatenate paths (cat+dog)
train_path = cat_train_path + dog_train_path
val_path = cat_val_path + dog_val_path
test_path = cat_test_path + dog_test_path

# Concatenate labels (cat+dog)
train_labels = torch.cat((cat_train_labels, dog_train_labels), dim=0)
val_labels = torch.cat((cat_val_labels, dog_val_labels), dim=0)
test_labels = torch.cat((cat_test_labels, dog_test_labels), dim=0)


train_dataset = MergeDataset(train_path, train_labels)
val_dataset = MergeDataset(val_path, val_labels)
test_dataset = MergeDataset(test_path, test_labels)

# Mean and std of the train_dataset
R_mean_sum = 0
R_std_sum = 0

G_mean_sum = 0
G_std_sum = 0

B_mean_sum = 0
B_std_sum = 0

for image, label in train_dataset:
    R_mean_sum += ImageStat.Stat(image).mean[0]
    R_std_sum += ImageStat.Stat(image).stddev[0]

    G_mean_sum += ImageStat.Stat(image).mean[1]
    G_std_sum += ImageStat.Stat(image).stddev[1]

    B_mean_sum += ImageStat.Stat(image).mean[2]
    B_std_sum += ImageStat.Stat(image).stddev[2]

R_mean = R_mean_sum / len(train_dataset) / 255
R_std = R_std_sum / len(train_dataset) / 255

G_mean = G_mean_sum / len(train_dataset) / 255
G_std = G_std_sum / len(train_dataset) / 255

B_mean = B_mean_sum / len(train_dataset) / 255
B_std = B_std_sum / len(train_dataset) / 255

mean = [R_mean, G_mean, B_mean]
std = [R_std, G_std, B_std]

print(f"mean {mean}")
print(f"std {std}")

# Set image size for transformations
image_size = (256, 256)

# Set training|validation|test transformations
training_transformations = transforms.Compose([
    transforms.Resize(image_size),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=20),
    transforms.RandomPerspective(distortion_scale=0.1, p=0.1),
    transforms.RandomCrop(size=image_size, padding=10),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

validation_transformations = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

test_transformations = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# Apply transformations
train_ds = CatVsDogDataset(train_dataset, image_transformations=training_transformations)
val_ds = CatVsDogDataset(val_dataset, image_transformations=validation_transformations)
test_ds = CatVsDogDataset(test_dataset, image_transformations=test_transformations)

