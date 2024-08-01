import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.cuda.amp as amp
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.transforms as T
from torchvision.transforms import functional as TF
import rasterio
import numpy as np
import pandas as pd
from patchify import patchify
import torchvision.models.segmentation as models
from segmentation_models_pytorch.metrics import iou_score, accuracy
import rioxarray
from PIL import Image
from tqdm import tqdm
import warnings
import random
import cv2
from torch_lr_finder import LRFinder


# Suppress the specific RuntimeWarning
warnings.filterwarnings("ignore", message="invalid value encountered in cast")

class RobustBCELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(RobustBCELoss, self).__init__()
        self.eps = eps

    def forward(self, input, target):
        input = torch.clamp(input, self.eps, 1 - self.eps)
        loss = nn.functional.binary_cross_entropy_with_logits(input, target, reduction='none')
        return torch.mean(torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0))

class ImprovedRobustBCELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(ImprovedRobustBCELoss, self).__init__()
        self.eps = eps

    def forward(self, input, target):
        input = torch.clamp(input, self.eps, 1 - self.eps)
        loss = nn.functional.binary_cross_entropy_with_logits(input, target, reduction='none')
        
        # Mask out NaN values in the target
        valid_mask = ~torch.isnan(target)
        loss = loss[valid_mask]
        
        if loss.numel() == 0:
            return torch.tensor(0.0, device=input.device, requires_grad=True)
        
        return loss.mean()

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return focal_loss.mean()

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        total = inputs.sum() + targets.sum()
        
        if total == 0:
            return torch.tensor(0.0).to(inputs.device)
        
        dice_coeff = (2. * intersection + self.smooth) / (total + self.smooth)
        return 1 - dice_coeff

# Combined Loss
class CombinedLoss(nn.Module):
    def __init__(self, weight=0.5, gamma=2, alpha=0.25):
        super(CombinedLoss, self).__init__()
        self.weight = weight
        self.focal = FocalLoss(gamma, alpha)
        self.dice = DiceLoss()

    def forward(self, inputs, targets):
        focal_loss = self.focal(inputs, targets)
        dice_loss = self.dice(inputs, targets)
        combined_loss = self.weight * focal_loss + (1 - self.weight) * dice_loss
        if torch.isnan(combined_loss):
            print(f"NaN in combined loss. Focal: {focal_loss}, Dice: {dice_loss}")
            return focal_loss if not torch.isnan(focal_loss) else dice_loss
        return combined_loss

class SegmentationGeotiffDataset(Dataset):
    def __init__(self, csv_file, train=True, train_split=0.8, patch_size=(256, 256), stride=None):
        self.csv_file = csv_file
        self.train = train
        self.train_split = train_split
        self.patch_size = patch_size
        self.stride = stride if stride else patch_size
        self.read_filenames_from_csv()
        self.split_dataset()
        self.class_frequencies = {0: 0, 1: 0}

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        row = self.dataset[idx]

        # Load and process images (lum_change, coh_change, dnbr)
        lum_change = self.load_and_resize(row['lum_change'])
        coh_change = self.load_and_resize(row['coh_change'])
        dnbr = self.load_and_resize(row['dnbr'])

        # Load and process binary_product
        binary_product = self.load_and_resize(row['binary_product'], is_mask=True)

        # Stack the input images
        stacked_array = np.stack([lum_change, coh_change, dnbr], axis=-1)

        # Patchify the stacked array and binary_product
        image_patches = patchify(stacked_array, (*self.patch_size, 3), step=(*self.stride, 3))
        mask_patches = patchify(binary_product, self.patch_size, step=self.stride)

        # Reshape patches
        image_patches = image_patches.reshape(-1, *self.patch_size, 3)
        mask_patches = mask_patches.reshape(-1, *self.patch_size)

        # Apply transforms to each patch
        transformed_images = []
        transformed_masks = []
        for img, mask in zip(image_patches, mask_patches):
            # Apply transformations directly to numpy arrays
            if self.train:  # Only apply augmentations during training
                if random.random() > 0.5:
                    img = np.flip(img, axis=1).copy()
                    mask = np.flip(mask, axis=1).copy()
                if random.random() > 0.5:
                    img = np.flip(img, axis=0).copy()
                    mask = np.flip(mask, axis=0).copy()
                if random.random() > 0.5:
                    k = random.choice([1, 2, 3])  # 90, 180, or 270 degrees
                    img = np.rot90(img, k=k, axes=(0, 1)).copy()
                    mask = np.rot90(mask, k=k, axes=(0, 1)).copy()

            # Ensure the arrays are contiguous and in the correct range
            img = np.ascontiguousarray(img)
            mask = np.ascontiguousarray(mask)
            
            img = np.clip(img, 0, 1).astype(np.float32)
            mask = mask.astype(np.uint8)

            # Convert to tensor
            img_tensor = torch.from_numpy(img.transpose(2, 0, 1))
            mask_tensor = torch.from_numpy(mask).long()

            transformed_images.append(img_tensor)
            transformed_masks.append(mask_tensor)

        return transformed_images, transformed_masks

    def load_and_resize(self, path, is_mask=False):
        data = rioxarray.open_rasterio(path).squeeze().values
        
        if not is_mask:
            try:
                data = preprocess_geospatial_data(data)
            except Exception as e:
                print(f"Error preprocessing data from {path}: {str(e)}")
                # Return a default value or handle the error as appropriate
                return np.zeros((1024, 1024), dtype=np.float32)
        else:
            data = (data > 0).astype(np.uint8)
        
        resized_image = cv2.resize(data, (1024, 1024), interpolation=cv2.INTER_NEAREST if is_mask else cv2.INTER_LANCZOS4)
        
        if is_mask:
            resized_image = (resized_image > 0).astype(np.uint8)
        else:
            resized_image = resized_image.astype(np.float32)
                
        return resized_image
    
    def read_filenames_from_csv(self):
        # Read filenames from CSV file
        csv_path = os.path.join(self.csv_file)
        self.data = pd.read_csv(csv_path)

    def extract_fire_id(self, path):
        return int(os.path.basename(path).split('_')[0])

    def split_dataset(self):
        test_fids = [7123, 7792]
        train_files = []
        test_files = []

        for _, row in self.data.iterrows():
            fire_id = self.extract_fire_id(row['lum_change'])
            if fire_id in test_fids:
                test_files.append(row)
            else:
                train_files.append(row)
        
        if self.train:
            self.dataset = train_files
        else:
            self.dataset = test_files
    
    def get_class_frequencies(self):
        return self.class_frequencies

def preprocess_geospatial_data(data):
    # Replace NaN with a specific value, eg - the mean of non-NaN values
    non_nan_mean = np.nanmean(data)
    data = np.nan_to_num(data, nan=non_nan_mean)
    
    # Normalize the data
    min_val, max_val = np.percentile(data, [1, 99])
    
    # Check if min_val and max_val are equal
    if np.isclose(min_val, max_val):
        return data.astype(np.float32)
    
    data = np.clip(data, min_val, max_val)
    
    # Add a small epsilon to avoid division by zero
    epsilon = 1e-8
    data = (data - min_val) / (max_val - min_val + epsilon)
    
    return data.astype(np.float32)

def compute_iou(pred, target, num_classes):
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()

    iou_per_class = []
    for cls in range(num_classes):
        intersection = ((pred == cls) & (target == cls)).sum().item()
        union = ((pred == cls) | (target == cls)).sum().item()
        
        if union != 0:
            iou = intersection / union
        else:
            iou = 0.0
        
        iou_per_class.append(iou)
    
    mean_iou = sum(iou_per_class) / num_classes
    return mean_iou

def compute_accuracy(pred, target):
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()

    correct = (pred == target).sum().item()
    total = target.size
    accuracy = correct / total
    return accuracy


# Load pre-trained DeepLabV3 model
model = models.deeplabv3_resnet50(weights="DEFAULT", progress=True)

# Replace the final layer with a new layer for binary classification
num_classes = 2
model.classifier = nn.Sequential(
    nn.Conv2d(2048, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
    nn.BatchNorm2d(256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
)

# Modify the first convolutional layer if input channels changed
if model.backbone.conv1.in_channels != 3:
    model.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

# Initialize only the new layers
model.classifier.apply(init_weights)
if model.backbone.conv1.in_channels != 3:
    init_weights(model.backbone.conv1)

# This one worked, but not with good test IoU or test accuracy
# optimizer = optim.AdamW([
#     {'params': model.backbone.parameters(), 'lr': 1e-5},
#     {'params': model.classifier.parameters(), 'lr': 1e-4}
# ], weight_decay=0.01)


# 1. Adjust the learning rate
optimizer = optim.AdamW([
    {'params': model.backbone.parameters(), 'lr': 1e-4},
    {'params': model.classifier.parameters(), 'lr': 1e-3}
], weight_decay=0.01)

# Add a learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

# Loss function
criterion = RobustBCELoss()

# Checkpoint loading
checkpoint_dir = '/Bhaltos/ASHWATH/integrated_model_checkpoints_100m_v7/'
latest_checkpoint = max(
    [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_epoch_') and f.endswith('.pt')],
    key=lambda f: int(f.split('_')[-1].replace('.pt', '')),
    default=None
)
if latest_checkpoint:
    checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
    checkpoint = torch.load(checkpoint_path)
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(new_state_dict)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_loss = checkpoint['loss']
    print(f"Loaded checkpoint from epoch {start_epoch-1} with loss {best_loss}")
else:
    start_epoch = 1
    best_loss = float('inf')
    print("No checkpoint found, starting training from scratch.")

# Define hyperparameters
batch_size = 3
accumulation_steps = 8
num_epochs = 20
checkpoint_freq = 1
best_loss = float('inf')
patience = 5
early_stopping_counter = 0

train_dataset = SegmentationGeotiffDataset(csv_file="/Bhaltos/ASHWATH/metadata_v2.csv", train=True)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True)

test_dataset = SegmentationGeotiffDataset(csv_file="/Bhaltos/ASHWATH/metadata_v2.csv", train=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=6, pin_memory=True)

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    device_ids = [0, 1, 2]  # IDs of all available GPUs
else:
    device_ids = None
if not isinstance(model, nn.DataParallel):
    model = nn.DataParallel(model, device_ids=device_ids)
model = model.to(device)

# Function to enable gradient checkpointing
def enable_gradient_checkpointing(model):
    def checkpoint_sequential(module):
        def custom_forward(*inputs):
            for submodule in module.children():
                inputs = submodule(*inputs)
            return inputs
        return lambda *x: checkpoint(custom_forward, *x)

    # Apply checkpointing to ResNet layers in the backbone
    if hasattr(model, 'backbone'):
        for name, module in model.backbone.named_children():
            if name.startswith('layer'):
                setattr(model.backbone, name, checkpoint_sequential(module))

    # Apply checkpointing to ASPP module
    if hasattr(model, 'classifier') and hasattr(model.classifier, 'aspp'):
        model.classifier.aspp = checkpoint_sequential(model.classifier.aspp)

    return model

# After model initialization and before training loop
model = enable_gradient_checkpointing(model)

# Create GradScaler for mixed precision training
scaler = amp.GradScaler()

# DataFrame to store metrics
columns = ['Epoch', 'Train Loss', 'Test Loss', 'Test IoU', 'Test Accuracy']
metrics_df = pd.DataFrame(columns=columns)

# Training loop
for epoch in range(start_epoch, num_epochs+1):
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()

    for i, (images, masks) in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch}/{num_epochs} - Training")):
        
        batch_loss = 0
        for img_patches, mask_patches in zip(images, masks):
            image_loss = 0
            for img, mask in zip(img_patches, mask_patches):
                img = img.unsqueeze(0).to(device)
                mask = mask.unsqueeze(0).to(device)

                with amp.autocast():
                    output = model(img)['out']
                    output_logits = output[:, 1]  # Use logits directly
                    patch_loss = criterion(output_logits, mask.float())
                
                scaler.scale(patch_loss).backward()

                image_loss += patch_loss

                # Clear unnecessary memory
                del img, mask, output, output_logits
                torch.cuda.empty_cache()
            
            # Average loss for all patches in the image
            image_loss /= len(img_patches)
            batch_loss += image_loss

            #scaler.scale(image_loss).backward()
        
        # Average loss for all images in the batch
        batch_loss /= len(images)
        batch_loss = batch_loss / accumulation_steps

        # Use scaler for mixed precision training
        #scaler.scale(batch_loss).backward()
        #batch_loss.backward()

        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_dataloader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            #optimizer.step()
            optimizer.zero_grad()
        
        running_loss += batch_loss.item() * accumulation_steps
    
    avg_train_loss = running_loss / len(train_dataloader)

    # Validation
    model.eval()
    test_loss = 0.0
    iou_scores = []
    accuracies = []

    with torch.no_grad():
        for images, masks in tqdm(test_dataloader, desc=f"Epoch {epoch}/{num_epochs} - Validation"):
            batch_loss = 0
            batch_iou = 0
            batch_accuracy = 0
            
            for img_patches, mask_patches in zip(images, masks):
                image_predictions = []
                image_masks = []
                image_loss = 0
                
                for img, mask in zip(img_patches, mask_patches):
                    img = img.unsqueeze(0).to(device)
                    mask = mask.unsqueeze(0).to(device)
                    
                    with amp.autocast():
                        output = model(img)['out']
                        output_logits = output[:, 1]
                        patch_loss = criterion(output_logits, mask.float())
                    
                    image_loss += patch_loss.item()
                    output_pred = (torch.sigmoid(output_logits) > 0.5).long()
                    
                    image_predictions.append(output_pred.cpu())
                    image_masks.append(mask.cpu())

                    # Clear unnecessary memory
                    del img, mask, output, output_logits
                    torch.cuda.empty_cache()
                
                image_loss /= len(img_patches)
                image_pred = torch.cat(image_predictions, dim=0)
                image_mask = torch.cat(image_masks, dim=0)
                
                # Calculate metrics for the entire image
                image_iou = compute_iou(image_pred, image_mask, num_classes)
                image_accuracy = compute_accuracy(image_pred, image_mask)
                
                batch_loss += image_loss
                batch_iou += image_iou
                batch_accuracy += image_accuracy
            
            # Average metrics for the batch
            batch_loss /= len(images)
            batch_iou /= len(images)
            batch_accuracy /= len(images)
            
            test_loss += batch_loss
            iou_scores.append(batch_iou)
            accuracies.append(batch_accuracy)

    avg_test_loss = test_loss / len(test_dataloader)
    avg_iou = np.mean(iou_scores)
    avg_accuracy = np.mean(accuracies)

    # Update metrics DataFrame
    new_row = pd.DataFrame({
    'Epoch': [epoch],
    'Train Loss': [avg_train_loss],
    'Test Loss': [avg_test_loss],
    'Test IoU': [avg_iou],
    'Test Accuracy': [avg_accuracy]
    })
    metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)

    print(f"Epoch [{epoch}/{num_epochs}] - Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}, IoU: {avg_iou:.4f}, Accuracy: {avg_accuracy:.4f}")

    # Check for early stopping
    if avg_test_loss < best_loss:
        best_loss = avg_test_loss
        early_stopping_counter = 0
        
        # Save the model checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': best_loss
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= patience:
            print("Early stopping triggered.")
            break

# Save metrics to CSV
metrics_output_path = '/Bhaltos/ASHWATH/integrated_100m_training_metrics_v7.csv'
metrics_df.to_csv(metrics_output_path, index=False)

print("Training complete.")