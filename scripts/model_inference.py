import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models.segmentation as models
import rioxarray
import numpy as np
import pandas as pd
from patchify import patchify, unpatchify
import cv2
from tqdm import tqdm
import h5py
import xarray as xr
from rasterio.transform import Affine
from rasterio.crs import CRS

class RobustBCELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(RobustBCELoss, self).__init__()
        self.eps = eps

    def forward(self, input, target):
        input = torch.clamp(input, self.eps, 1 - self.eps)
        loss = nn.functional.binary_cross_entropy_with_logits(input, target, reduction='none')
        return torch.mean(torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0))

class SegmentationGeotiffDataset(Dataset):
    def __init__(self, csv_file, patch_size=(256, 256), stride=None):
        self.csv_file = csv_file
        self.patch_size = patch_size
        self.stride = stride if stride else patch_size
        self.read_filenames_from_csv()

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        row = self.dataset[idx]

        lum_change = self.load_and_resize(row['lum_change'])
        coh_change = self.load_and_resize(row['coh_change'])
        dnbr = self.load_and_resize(row['dnbr'])

        stacked_array = np.stack([lum_change, coh_change, dnbr], axis=-1)
        image_patches = patchify(stacked_array, (*self.patch_size, 3), step=(*self.stride, 3))
        image_patches = image_patches.reshape(-1, *self.patch_size, 3)

        transformed_images = []
        for img in image_patches:
            img = np.ascontiguousarray(img)
            img = np.clip(img, 0, 1).astype(np.float32)
            img_tensor = torch.from_numpy(img.transpose(2, 0, 1))
            transformed_images.append(img_tensor)

        return transformed_images, row['lum_change']  # Return original file path for reconstruction

    def load_and_resize(self, path):
        data = rioxarray.open_rasterio(path).squeeze().values
        data = preprocess_geospatial_data(data)
        resized_image = cv2.resize(data, (1024, 1024), interpolation=cv2.INTER_LANCZOS4)
        return resized_image.astype(np.float32)
    
    def read_filenames_from_csv(self):
        csv_path = os.path.join(self.csv_file)
        self.dataset = pd.read_csv(csv_path)

def preprocess_geospatial_data(data):
    non_nan_mean = np.nanmean(data)
    data = np.nan_to_num(data, nan=non_nan_mean)
    min_val, max_val = np.percentile(data, [1, 99])
    if np.isclose(min_val, max_val):
        return data.astype(np.float32)
    data = np.clip(data, min_val, max_val)
    epsilon = 1e-8
    data = (data - min_val) / (max_val - min_val + epsilon)
    return data.astype(np.float32)

def load_model(checkpoint_path):
    model = models.deeplabv3_resnet50(num_classes=2)
    model.classifier = nn.Sequential(
        nn.Conv2d(2048, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1))
    )
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def run_inference(model, dataloader, device):
    model.eval()
    predictions = {}
    
    with torch.no_grad():
        for batch, file_paths in tqdm(dataloader, desc="Running inference"):
            batch_predictions = []
            for img_patches in batch:
                image_predictions = []
                for img in img_patches:
                    img = img.unsqueeze(0).to(device)
                    with torch.cuda.amp.autocast():
                        output = model(img)['out']
                        output = torch.sigmoid(output[:, 1])  # Get probability for class 1
                    image_predictions.append(output.cpu().numpy())
                
                # Reconstruct full image prediction
                full_pred = unpatchify(np.array(image_predictions).reshape(4, 4, 256, 256), (1024, 1024))
                batch_predictions.append(full_pred)
            
            # Store predictions for each file
            for pred, file_path in zip(batch_predictions, file_paths):
                predictions[file_path] = pred
    
    return predictions

def save_predictions(predictions, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for file_path, pred in predictions.items():
        output_path = os.path.join(output_dir, os.path.basename(file_path).replace('.tif', '_pred.h5'))
        
        with rioxarray.open_rasterio(file_path) as src:
            # Create HDF5 file
            with h5py.File(output_path, 'w') as f:
                # Save prediction data
                f.create_dataset('prediction', data=pred, compression='gzip')
                
                # Save georeferencing information
                geo = f.create_group('georeference')
                geo.attrs['crs'] = src.rio.crs.to_string()
                geo.attrs['transform'] = src.rio.transform()
                geo.attrs['width'] = src.rio.width
                geo.attrs['height'] = src.rio.height
                
                # Save additional metadata
                meta = f.create_group('metadata')
                for key, value in src.attrs.items():
                    if isinstance(value, str):
                        meta.attrs[key] = value
                    else:
                        # For non-string attributes, store as dataset
                        meta.create_dataset(key, data=np.array(value))
                
        print(f"Saved prediction to {output_path}")

def read_prediction(file_path):
    with h5py.File(file_path, 'r') as f:
        # Read prediction data
        prediction = f['prediction'][:]
        
        # Read georeferencing information
        crs = f['georeference'].attrs['crs']
        transform = f['georeference'].attrs['transform']
        width = f['georeference'].attrs['width']
        height = f['georeference'].attrs['height']
        
        # Create a DataArray with georeferencing information
        da = xr.DataArray(
            prediction,
            coords={
                'y': np.arange(height) * transform[4] + transform[5],
                'x': np.arange(width) * transform[0] + transform[2]
            },
            dims=['y', 'x']
        )
        
        # Set CRS and transform
        da.rio.set_crs(crs)
        da.rio.set_transform(Affine(*transform))
        
        # Add metadata
        for key, value in f['metadata'].attrs.items():
            da.attrs[key] = value
        
        return da

if __name__ == "__main__":
    # Set file paths and output directory
    checkpoint_path = '/path/to/your/checkpoint.pt'
    new_data_csv = '/path/to/new/dataset.csv'
    output_dir = '/path/to/output/directory'
    batch_size = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = load_model(checkpoint_path)
    model = model.to(device)

    # Set up dataset and dataloader
    dataset = SegmentationGeotiffDataset(csv_file=new_data_csv)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Run inference
    predictions = run_inference(model, dataloader, device)

    # Save predictions
    save_predictions(predictions, output_dir)

    print("Inference complete. Predictions saved.")

    # To read a prediction
    prediction = read_prediction('/path/to/prediction.h5')