# This version should be the same as training.py but using pytorch lightning and early stopping

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.models.segmentation as models
from segmentation_models_pytorch.metrics import iou_score, get_stats, accuracy
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import rioxarray
from patchify import patchify
import numpy as np
import pandas as pd
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

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_filename = self.dataset[idx]
        img_path = os.path.join(self.image_folder, img_filename)
        mask_filename = img_filename.replace("_DNBR.TIF", "_MASK.TIF")
        mask_path = os.path.join(self.mask_folder, mask_filename)

        img = rioxarray.open_rasterio(img_path).squeeze()
        image = img.values

        msk = rioxarray.open_rasterio(mask_path).squeeze()
        mask = msk.values
        mask = mask - 1

        image = transforms.ToPILImage()(image)
        mask = transforms.ToPILImage()(mask)

        new_size = (1098, 1098)
        image = np.array(image.resize(new_size, resample=Image.LANCZOS))
        mask = np.array(mask.resize(new_size, resample=Image.LANCZOS))

        image_patches = patchify(image, self.patch_size, step=self.stride)
        mask_patches = patchify(mask, self.patch_size, step=self.stride)

        image_patches = np.reshape(image_patches, (-1, self.patch_size[0], self.patch_size[1]))
        mask_patches = np.reshape(mask_patches, (-1, self.patch_size[0], self.patch_size[1]))

        images = [transforms.ToPILImage()(patch) for patch in image_patches]
        masks = [transforms.ToPILImage()(patch) for patch in mask_patches]

        images = [transforms.functional.to_tensor(patch) for patch in images]
        masks = [transforms.functional.to_tensor(patch) for patch in masks]

        num_classes = 3
        masks_one_hot = []
        for mask in masks:
            mask_tensor = mask.squeeze(0).long()
            mask_one_hot = torch.zeros(num_classes, mask_tensor.size(0), mask_tensor.size(1))
            mask_one_hot.scatter_(0, mask_tensor.unsqueeze(0), 1)
            masks_one_hot.append(mask_one_hot)

        return images, masks_one_hot

    def read_filenames_from_csv(self):
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


class GeoSegmentationModel(pl.LightningModule):
    def __init__(self, num_classes=3, learning_rate=0.001):
        super(GeoSegmentationModel, self).__init__()
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.criterion = DiceLoss()
        
        self.model = models.deeplabv3_resnet50(weights="DEFAULT", progress=True)
        self.model.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.classifier = nn.Sequential(
            nn.Conv2d(2048, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Dropout(0.5),
            nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
        )

    def forward(self, x):
        return self.model(x)['out']

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=0.01)
        return optimizer

    def training_step(self, batch, batch_idx):
        images, masks = batch
        batch_loss = 0.0

        for i in range(len(images)):
            image_patch, mask_patch = images[i].to(self.device), masks[i].to(self.device)
            outputs = self(image_patch)
            loss = self.criterion(outputs, mask_patch)
            batch_loss += loss

        batch_loss = batch_loss / len(images)
        self.log('train_loss', batch_loss)
        return batch_loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        batch_loss = 0.0
        iou_scores = []
        accuracy_scores = []

        for i in range(len(images)):
            image_patch, mask_patch = images[i].to(self.device), masks[i].to(self.device)
            outputs = self(image_patch)
            loss = self.criterion(outputs, mask_patch)
            batch_loss += loss

            outputs = torch.softmax(outputs, dim=1)
            pred = torch.argmax(outputs, dim=1)
            tp, fp, fn, tn = get_stats(pred, mask_patch, mode='multiclass', num_classes=self.num_classes)

            iou = iou_score(tp, fp, fn, tn, reduction='micro')
            iou_scores.append(iou.item())

            acc = accuracy(tp, fp, fn, tn, reduction='micro')
            accuracy_scores.append(acc)

        batch_loss = batch_loss / len(images)
        self.log('val_loss', batch_loss)
        self.log('val_iou', torch.mean(torch.tensor(iou_scores)))
        self.log('val_accuracy', torch.mean(torch.tensor(accuracy_scores)))
        return batch_loss


# Paths
root_dir = "/Bhaltos/ASHWATH/Dataset"
csv_file = "/Bhaltos/ASHWATH/updated_filenames.csv"
checkpoint_dir = '/Bhaltos/ASHWATH/model_checkpoints_100m/'

# Datasets and DataLoaders
train_dataset = SegmentationGeotiffDataset(root_dir, csv_file, train=True)
train_dataloader = DataLoader(train_dataset, batch_size=3, shuffle=True)

test_dataset = SegmentationGeotiffDataset(root_dir, csv_file, train=False)
test_dataloader = DataLoader(test_dataset, batch_size=3, shuffle=False)

# Model
model = GeoSegmentationModel()

# Checkpointing
checkpoint_callback = ModelCheckpoint(
    dirpath=checkpoint_dir,
    filename='checkpoint-{epoch:02d}-{val_loss:.4f}',
    save_top_k=1,
    mode='min',
    monitor='val_loss'
)

# Early stopping
early_stopping_callback = EarlyStopping(
    monitor='val_loss',  # You can change this to 'val_iou' or any other metric
    patience=5,          # Number of epochs with no improvement after which training will be stopped
    mode='min',          # Minimize the val_loss
    verbose=True         # Print early stopping messages
)

# Trainer
trainer = pl.Trainer(
    max_epochs=10,
    gpus=-1 if torch.cuda.is_available() else 0,
    callbacks=[checkpoint_callback, early_stopping_callback],
    precision=16,  # Use mixed precision if applicable
)

# Train and validate
trainer.fit(model, train_dataloader, test_dataloader)
