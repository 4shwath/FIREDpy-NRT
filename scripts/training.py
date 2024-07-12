# This version uses dice loss instead of cross entropy loss, model checkpointing, 100m resolution optical data and resnet50 backbone
# Future version 1 should use 100m resolution integrated SAR and optical data, resnet50 backbone
# Future version 2 should use 10m resolution integrated SAR and optical data, resnet100 or above

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import rasterio
import numpy as np
import pandas as pd
from patchify import patchify
import torchvision.models.segmentation as models
from segmentation_models_pytorch.metrics import iou_score, get_stats, accuracy
import rioxarray
from PIL import Image
from tqdm import tqdm
import warnings

# Suppress the specific RuntimeWarning
warnings.filterwarnings("ignore", message="invalid value encountered in cast")


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets):
        smooth = 1e-5
        
        # Flatten inputs and targets
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        dice_coeff = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        
        return 1 - dice_coeff


class SegmentationGeotiffDataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=None, train=True, train_split=0.8, patch_size=(256, 256), stride=None):
        self.root_dir = root_dir
        self.csv_file = csv_file
        self.transform = transform
        self.train = train
        self.train_split = train_split
        self.patch_size = patch_size
        self.stride = stride if stride else patch_size
        self.image_folder = os.path.join(root_dir, "dNBR")
        self.mask_folder = os.path.join(root_dir, "Masks")
        self.read_filenames_from_csv()
        self.split_dataset()

        self.class_frequencies = {0: 0, 1: 0, 2: 0}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_filename = self.dataset[idx]
        img_path = os.path.join(self.image_folder, img_filename)
        mask_filename = img_filename.replace("_DNBR.TIF", "_MASK.TIF")
        mask_path = os.path.join(self.mask_folder, mask_filename)

        # Load image
        img = rioxarray.open_rasterio(img_path).squeeze()
        image = img.values

        # Load mask
        msk = rioxarray.open_rasterio(mask_path).squeeze()
        mask = msk.values
        mask = mask - 1

        # For getting class frequencies
        mask_flat = mask.flatten()
        unique_classes, class_counts = np.unique(mask_flat, return_counts=True)
        for cls, count in zip(unique_classes, class_counts):
            self.class_frequencies[cls] += count

        image = transforms.ToPILImage()(image)
        mask = transforms.ToPILImage()(mask)

        new_size = (1098, 1098)
        image = np.array(image.resize(new_size, resample=Image.LANCZOS))
        mask = np.array(mask.resize(new_size, resample=Image.LANCZOS))

        # Patchify image and mask
        image_patches = patchify(image, self.patch_size, step=self.stride)
        mask_patches = patchify(mask, self.patch_size, step=self.stride)

        # Reshape patches
        image_patches = np.reshape(image_patches, (-1, self.patch_size[0], self.patch_size[1]))
        mask_patches = np.reshape(mask_patches, (-1, self.patch_size[0], self.patch_size[1]))

        # Convert to PIL image and tensor
        images = [transforms.ToPILImage()(patch) for patch in image_patches]
        masks = [transforms.ToPILImage()(patch) for patch in mask_patches]

        images = [transforms.functional.to_tensor(patch) for patch in images]
        masks = [transforms.functional.to_tensor(patch) for patch in masks]

        # Convert masks to one-hot encoding
        num_classes = 3
        masks_one_hot = []
        for mask in masks:
            mask_tensor = mask.squeeze(0).long()  # remove the channel dimension
            mask_one_hot = torch.zeros(num_classes, mask_tensor.size(0), mask_tensor.size(1))
            mask_one_hot.scatter_(0, mask_tensor.unsqueeze(0), 1)
            masks_one_hot.append(mask_one_hot)

        return images, masks_one_hot
    
    def read_filenames_from_csv(self):
        # Read filenames from CSV file
        csv_path = os.path.join(self.root_dir, self.csv_file)
        filenames_df = pd.read_csv(csv_path)
        self.image_filenames = filenames_df['dNBR'].tolist()

    def split_dataset(self):
        test_fids = [5587, 7792]
        train_files = []
        test_files = []
        for filename in self.image_filenames:
            fid = int(filename.split('_')[3])
            if fid in test_fids:
                test_files.append(filename)
            else:
                train_files.append(filename)
        
        if self.train:
            self.dataset = train_files
        else:
            self.dataset = test_files
    
    def get_class_frequencies(self):
        return self.class_frequencies


train_dataset = SegmentationGeotiffDataset(root_dir="/Bhaltos/ASHWATH/Dataset", csv_file="/Bhaltos/ASHWATH/updated_filenames.csv", train=True)
train_dataloader = DataLoader(train_dataset, batch_size=3, shuffle=True)

test_dataset = SegmentationGeotiffDataset(root_dir="/Bhaltos/ASHWATH/Dataset", csv_file="/Bhaltos/ASHWATH/updated_filenames.csv", train=False)
test_dataloader = DataLoader(test_dataset, batch_size=3, shuffle=True)

# Load pre-trained DeepLabV3 model
model = models.deeplabv3_resnet50(weights="DEFAULT", progress=True)

# Replace the final layer with a new layer
num_classes = 3  # Number of classes in your dataset
model.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
model.classifier = nn.Sequential(
    nn.Conv2d(2048, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
    nn.Dropout(0.5),
    nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
)

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    device_ids = [0, 1, 2]  # IDs of all available GPUs
else:
    device_ids = None
model = nn.DataParallel(model, device_ids=device_ids)
model = model.to(device)

# Define optimizer and Dice loss criterion
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
criterion = DiceLoss()

# Checkpoint loading
# Check if there exists a checkpoint, if yes, load the latest one
checkpoint_dir = '/Bhaltos/ASHWATH/model_checkpoints_100m/'
latest_checkpoint = max([f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_epoch_')], default=None)
if latest_checkpoint:
    checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
    #checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint_epoch_1.pt')
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_loss = checkpoint['loss']
    print(f"Loaded checkpoint from epoch {start_epoch-1} with loss {best_loss}")
else:
    start_epoch = 1
    best_loss = float('inf')
    print("No checkpoint found, starting training from scratch.")

# Training loop
num_epochs = 10  # Adjust as needed
checkpoint_freq = 1
start_epoch = 1
best_loss = float('inf')

for epoch in range(start_epoch, start_epoch + num_epochs):
    model.train()
    running_loss = 0.0

    for images, masks in tqdm(train_dataloader, desc=f"Epoch {epoch}/{start_epoch + num_epochs - 1}"):
        optimizer.zero_grad()
        image_loss = 0.0

        for i in range(len(images)):
            image_patch, mask_patch = images[i].to(device), masks[i].to(device)
            outputs = model(image_patch)['out']
            loss = criterion(outputs, mask_patch)
            image_loss += loss.item()

            loss.backward()
            optimizer.step()

            del image_patch
            del mask_patch
            torch.cuda.empty_cache()

        running_loss += image_loss / len(images)

    epoch_loss = running_loss / len(train_dataloader)
    print(f"Epoch [{epoch}/{start_epoch + num_epochs - 1}], Loss: {epoch_loss:.4f}")

    # Checkpoint saving
    if (epoch) % checkpoint_freq == 0:
        checkpoint_name = f'checkpoint_epoch_{epoch}.pt'
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
        }, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")

    # Evaluation loop
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    iou_scores = []
    accuracy_scores = []

    for images, masks in tqdm(test_dataloader, desc="Evaluation"):
        with torch.no_grad():
            image_loss = 0.0

            for i in range(len(images)):
                image_patch, mask_patch = images[i].to(device), masks[i].to(device)
                outputs = model(image_patch)['out']
                loss = criterion(outputs, mask_patch)
                image_loss += loss.item()

                outputs = torch.softmax(outputs, dim=1)

                # Compute IoU and accuracy
                pred = torch.argmax(outputs, dim=1)
                tp, fp, fn, tn = get_stats(pred, mask_patch, mode='multiclass', num_classes=num_classes)
                
                iou = iou_score(tp, fp, fn, tn, reduction='micro')  # You can change reduction to 'macro' or 'none'
                iou_scores.append(iou.item())

                acc = accuracy(tp, fp, fn, tn, reduction='micro')
                accuracy_scores.append(acc)

                del image_patch
                del mask_patch
                torch.cuda.empty_cache()

            total_loss += image_loss / len(images)

    # Calculate and print evaluation metrics (IoU, accuracy, etc.)
    average_loss = total_loss / len(test_dataloader)
    mean_iou = torch.stack(iou_scores).mean().item()
    mean_accuracy = torch.stack(accuracy_scores).mean().item()

    print(f"Average Dice Loss on Test Set: {average_loss:.4f}")
    print(f"Mean IoU: {mean_iou:.4f}, Mean Accuracy: {mean_accuracy:.4f}")