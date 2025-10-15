# experiments/forest_fire.py
import sys
sys.path.append('.')

import torch
import json
import os
from src.models import get_model
from src.utils import get_dataloaders
from src.train import train_model


def run_forest_fire_experiments(quick_test=False):
    """Run all forest fire detection models"""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # Quick test mode
    if quick_test:
        batch_size = 32
        epochs = 5
        print("QUICK TEST MODE: 5 epochs, batch_size=32\n")
    else:
        batch_size = 16
        epochs = 10
        print("FULL MODE: 10 epochs, batch_size=16\n")
    
    train_loader, test_loader = get_dataloaders('fire', batch_size=batch_size)
    
    configs = [
        ('Normal ViT', 'normal', False),
        ('ViT + CKKS', 'normal', True),
        ('Two-bit CKKS', 'two_bit', True),
        ('Three-bit CKKS', 'three_bit', True),
    ]
    
    results = {}
    
    for name, mode, use_ckks in configs:
        print("="*60)
        print(f"Training: {name}")
        print("="*60)
        
        model = get_model('fire', mode, device=device)
        
        metrics = train_model(
            model, train_loader, test_loader,
            task='fire',
            use_ckks=use_ckks,
            epochs=epochs,
            lr=1e-4,
            device=device
        )
        
        results[name] = metrics
        
        print(f"\nResults for {name}:")
        print(f"  Accuracy: {metrics['Accuracy']:.4f}")
        print(f"  Precision: {metrics['Precision']:.4f}")
        print(f"  Recall: {metrics['Recall']:.4f}")
        print(f"  F1: {metrics['F1']:.4f}")
        print(f"  Avg epoch time: {metrics['avg_epoch_time']:.2f}s")
        print(f"  Model size: {metrics['model_size_bytes']/1024:.2f}KB\n")
    
    # Save results
    os.makedirs('results', exist_ok=True)
    filename = 'results/forest_fire_quick.json' if quick_test else 'results/forest_fire_metrics.json'
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print comparison table
    print("\n" + "="*60)
    print("COMPARISON TABLE")
    print("="*60)
    print(f"{'Model':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-"*60)
    for name, metrics in results.items():
        print(f"{name:<20} {metrics['Accuracy']:>10.4f} {metrics['Precision']:>10.4f} "
              f"{metrics['Recall']:>10.4f} {metrics['F1']:>10.4f}")
    
    print(f"\nResults saved to: {filename}")


if __name__ == '__main__':
    import sys
    
    quick = '--quick' in sys.argv or '-q' in sys.argv
    
    run_forest_fire_experiments(quick_test=quick)