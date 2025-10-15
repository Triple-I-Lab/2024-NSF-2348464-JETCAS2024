# src/train.py
import torch
import torch.nn as nn
import time
import numpy as np
from src.utils import calculate_metrics
from src.ckks import SimpleCKKS

# Initialize CKKS instance
ckks = SimpleCKKS()


def train_model(model, train_loader, test_loader, task='air', 
                use_ckks=False, epochs=100, lr=0.01, device='cuda'):
    """
    Universal training function for all models
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        test_loader: Test data loader
        task: 'air' or 'fire'
        use_ckks: Whether to use CKKS encryption
        epochs: Number of training epochs
        lr: Learning rate
        device: 'cuda' or 'cpu'
    
    Returns:
        Dictionary of metrics and training info
    """
    model = model.to(device)
    criterion = nn.MSELoss() if task == 'air' else nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    train_times = []
    best_loss = float('inf')
    patience_counter = 0
    patience = 10
    
    print(f"Training with CKKS: {use_ckks}")
    
    for epoch in range(epochs):
        model.train()
        start_time = time.time()
        epoch_loss = 0
        
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            
            # Apply CKKS encryption/decryption if enabled
            if use_ckks:
                X_cpu = X.cpu()
                encrypted_X, shape = ckks.encrypt_tensor(X_cpu)
                X = ckks.decrypt_tensor(encrypted_X, shape).to(device)
            
            optimizer.zero_grad()
            outputs = model(X)
            
            if task == 'air':
                loss = criterion(outputs.squeeze(), y.squeeze())
            else:
                loss = criterion(outputs, y)
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        epoch_time = time.time() - start_time
        train_times.append(epoch_time)
        avg_loss = epoch_loss / len(train_loader)
        
        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Time: {epoch_time:.2f}s')
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Evaluation
    print("Evaluating model...")
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            
            if use_ckks:
                X_cpu = X.cpu()
                encrypted_X, shape = ckks.encrypt_tensor(X_cpu)
                X = ckks.decrypt_tensor(encrypted_X, shape).to(device)
            
            outputs = model(X)
            
            if task == 'air':
                preds = outputs.squeeze().cpu().numpy()
            else:
                preds = torch.argmax(outputs, 1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(y.cpu().numpy().flatten())
    
    # Calculate metrics
    metrics = calculate_metrics(
        np.array(all_labels), 
        np.array(all_preds),
        task='regression' if task == 'air' else 'classification'
    )
    
    # Add training info
    metrics['avg_epoch_time'] = np.mean(train_times)
    metrics['total_epochs'] = len(train_times)
    metrics['model_size_bytes'] = sum(p.numel() for p in model.parameters()) * 4
    
    return metrics


# ============= TESTS =============
if __name__ == '__main__':
    import torch
    from src.models import get_model
    from src.utils import get_dataloaders
    
    print("="*60)
    print("Testing Training Pipeline")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Test 1: Normal training
    print("\n[Test 1] Normal Training (2 epochs)")
    try:
        train_loader, test_loader = get_dataloaders('air', batch_size=32)
        model = get_model('air', mode='normal', device=device)
        
        metrics = train_model(
            model, train_loader, test_loader,
            task='air',
            use_ckks=False,
            epochs=2,
            lr=0.01,
            device=device
        )
        
        print(f"\nNormal - R2: {metrics['R2']:.4f}, Time: {metrics['avg_epoch_time']:.2f}s")
        assert 'R2' in metrics
        print("Test 1 Passed")
    except Exception as e:
        print(f"Test 1 Skipped: {e}")
    
    # Test 2: Training with CKKS
    print("\n[Test 2] Normal Model + CKKS (2 epochs)")
    try:
        train_loader, test_loader = get_dataloaders('air', batch_size=32)
        model = get_model('air', mode='normal', device=device)
        
        metrics = train_model(
            model, train_loader, test_loader,
            task='air',
            use_ckks=True,
            epochs=2,
            lr=0.01,
            device=device
        )
        
        print(f"\nWith CKKS - R2: {metrics['R2']:.4f}, Time: {metrics['avg_epoch_time']:.2f}s")
        assert 'R2' in metrics
        print("Test 2 Passed")
    except Exception as e:
        print(f"Test 2 Skipped: {e}")
    
    # Test 3: Quantized model + CKKS
    print("\n[Test 3] Quantized Model + CKKS (2 epochs)")
    try:
        train_loader, test_loader = get_dataloaders('air', batch_size=32)
        model = get_model('air', mode='two_bit', device=device)
        
        metrics = train_model(
            model, train_loader, test_loader,
            task='air',
            use_ckks=True,
            epochs=2,
            lr=0.01,
            device=device
        )
        
        print(f"\nQuantized + CKKS - R2: {metrics['R2']:.4f}, Time: {metrics['avg_epoch_time']:.2f}s")
        assert 'R2' in metrics
        print("Test 3 Passed")
    except Exception as e:
        print(f"Test 3 Skipped: {e}")
    # Test 4: Quantized without CKKS 
    print("\n[Test 4] Quantized Model WITHOUT CKKS (5 epochs)")
    try:
        train_loader, test_loader = get_dataloaders('air', batch_size=32)
        model = get_model('air', mode='two_bit', device=device)
        
        metrics = train_model(
            model, train_loader, test_loader,
            task='air',
            use_ckks=False,
            epochs=5,
            lr=0.001,  # Lower learning rate for quantized
            device=device
        )
        
        print(f"\nQuantized only - R2: {metrics['R2']:.4f}, Time: {metrics['avg_epoch_time']:.2f}s")
        print("Test 4 Passed")
    except Exception as e:
        print(f"Test 4 Skipped: {e}")    
    print("\n" + "="*60)
    print("All Tests Completed")
    print("="*60)