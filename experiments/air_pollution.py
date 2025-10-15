# experiments/air_pollution.py
import sys
sys.path.append('.')
import torch
import json
import os
from src.models import get_model
from src.utils import get_dataloaders
from src.train import train_model


def run_air_pollution_experiments(quick_test=False):
    """Run all air pollution forecasting models"""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # Quick test mode: smaller batch, fewer epochs
    if quick_test:
        batch_size = 128
        epochs = 10
        print("QUICK TEST MODE: 10 epochs, batch_size=128\n")
    else:
        batch_size = 32
        epochs = 100
        print("FULL MODE: 100 epochs, batch_size=32\n")
    
    train_loader, test_loader = get_dataloaders('air', batch_size=batch_size)
    
    configs = [
        ('Normal LSTM', 'normal', False),
        ('LSTM + CKKS', 'normal', True),
        ('Two-bit CKKS', 'two_bit', True),
        ('Three-bit CKKS', 'three_bit', True),
    ]
    
    results = {}
    
    for name, mode, use_ckks in configs:
        print("="*60)
        print(f"Training: {name}")
        print("="*60)
        
        model = get_model('air', mode, device=device)
        if mode in ['two_bit', 'three_bit']:
            model_epochs = epochs * 3  
        else:
            model_epochs = epochs
        metrics = train_model(
            model, train_loader, test_loader,
            task='air',
            use_ckks=use_ckks,
            epochs=model_epochs,
            lr=0.01,
            device=device
        )
        
        results[name] = metrics
        
        print(f"\nResults for {name}:")
        print(f"  R2: {metrics['R2']:.4f}")
        print(f"  MAE: {metrics['MAE']:.4f}")
        print(f"  RMSE: {metrics['RMSE']:.4f}")
        print(f"  MAPE: {metrics['MAPE']:.2f}%")
        print(f"  Avg epoch time: {metrics['avg_epoch_time']:.2f}s")
        print(f"  Model size: {metrics['model_size_bytes']/1024:.2f}KB\n")
    
    # Save results
    os.makedirs('results', exist_ok=True)
    filename = 'results/air_pollution_quick.json' if quick_test else 'results/air_pollution_metrics.json'
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print comparison table
    print("\n" + "="*60)
    print("COMPARISON TABLE")
    print("="*60)
    print(f"{'Model':<20} {'R2':>8} {'MAE':>8} {'RMSE':>8} {'Time(s)':>10}")
    print("-"*60)
    for name, metrics in results.items():
        print(f"{name:<20} {metrics['R2']:>8.4f} {metrics['MAE']:>8.2f} "
              f"{metrics['RMSE']:>8.2f} {metrics['avg_epoch_time']:>10.2f}")
    
    print(f"\nResults saved to: {filename}")


if __name__ == '__main__':
    import sys
    
    # Check if quick test mode
    quick = '--quick' in sys.argv or '-q' in sys.argv
    
    run_air_pollution_experiments(quick_test=quick)