# src/utils.py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
from torchvision import transforms
from pathlib import Path


# ============ Metrics ============
def calculate_metrics(y_true, y_pred, task='regression'):
    """Calculate metrics for regression or classification"""
    if task == 'regression':
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mask = y_true != 0
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.sum() > 0 else 0.0
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        return {'MAE': float(mae), 'RMSE': float(rmse), 'MAPE': float(mape), 'R2': float(r2)}
    else:  # classification
        correct = (y_true == y_pred).sum()
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        return {'Accuracy': float(correct / len(y_true)), 'Precision': float(precision), 
                'Recall': float(recall), 'F1': float(f1)}


# ============ Datasets ============
class AirPollutionDataset(Dataset):
    def __init__(self, csv_path, window_size=5):
        df = pd.read_csv(csv_path)
        pollution = df['pollution'].values
        self.X, self.y = [], []
        for i in range(len(pollution) - window_size):
            self.X.append(pollution[i:i + window_size])
            self.y.append(pollution[i + window_size])
        self.X = np.array(self.X).reshape(-1, window_size, 1)
        self.y = np.array(self.y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.X[idx]), torch.FloatTensor([self.y[idx]])


class FireDataset(Dataset):
    def __init__(self, root_dir, img_size=224):
        self.root = Path(root_dir)
        self.samples = []
        for label, folder in enumerate(['nofire', 'fire']):
            folder_path = self.root / folder
            if folder_path.exists():
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                    self.samples.extend([(str(p), label) for p in folder_path.glob(ext)])
        
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        return self.transform(img), label


def get_dataloaders(task, batch_size=32):
    """Get train/test loaders"""
    if task == 'air':
        train_ds = AirPollutionDataset('data/data_series/LSTM-Multivariate_pollution.csv')
        test_ds = AirPollutionDataset('data/data_series/pollution_test_data1.csv')
    else:  # fire
        train_ds = FireDataset('data/forest/train')
        test_ds = FireDataset('data/forest/test')
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


# ============= TESTS =============
if __name__ == '__main__':
    print("="*60)
    print("Testing Utils")
    print("="*60)
    
    # Test 1: Metrics
    print("\n[Test 1] Metrics Calculation")
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.2, 2.9, 4.1, 4.8])
    metrics = calculate_metrics(y_true, y_pred, task='regression')
    print(f"Regression - R2: {metrics['R2']:.4f}, MAE: {metrics['MAE']:.4f}")
    assert metrics['R2'] > 0.9 and 'MAPE' in metrics
    
    y_true_cls = np.array([0, 1, 1, 0, 1])
    y_pred_cls = np.array([0, 1, 1, 0, 0])
    metrics_cls = calculate_metrics(y_true_cls, y_pred_cls, task='classification')
    print(f"Classification - Acc: {metrics_cls['Accuracy']:.4f}, F1: {metrics_cls['F1']:.4f}")
    assert 'F1' in metrics_cls
    print("Test 1 Passed")
    
    # Test 2: Air Pollution Dataset
    print("\n[Test 2] Air Pollution Dataset")
    try:
        dataset = AirPollutionDataset('data/data_series/LSTM-Multivariate_pollution.csv', window_size=5)
        X, y = dataset[0]
        print(f"Dataset length: {len(dataset)}, X shape: {X.shape}, y shape: {y.shape}")
        assert X.shape == (5, 1) and y.shape == (1,)
        print("Test 2 Passed")
    except FileNotFoundError:
        print("Test 2 Skipped (data file not found)")
    
    # Test 3: Fire Dataset
    print("\n[Test 3] Fire Dataset")
    try:
        dataset = FireDataset('data/forest/train', img_size=224)
        if len(dataset) > 0:
            X, y = dataset[0]
            print(f"Dataset length: {len(dataset)}, X shape: {X.shape}, Label: {y}")
            assert X.shape == (3, 224, 224) and y in [0, 1]
            print("Test 3 Passed")
        else:
            print("Test 3 Skipped (no images found)")
    except FileNotFoundError:
        print("Test 3 Skipped (data directory not found)")
    
    print("\n" + "="*60)
    print("All Tests Completed")
    print("="*60)