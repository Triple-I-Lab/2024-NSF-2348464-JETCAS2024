# src/models.py
import torch
import torch.nn as nn
from src.quant_linear import create_quantized_copy_of_model, QuantizationMode


class LSTMPollution(nn.Module):
    """LSTM model for air pollution forecasting"""
    def __init__(self, input_size=1, hidden_size=64, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 5)
        self.fc2 = nn.Linear(5, output_size)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = torch.relu(out[:, -1, :])
        out = torch.relu(self.fc1(out))
        return self.fc2(out)


class ViTFireDetection(nn.Module):
    """Vision Transformer for forest fire detection"""
    def __init__(self, img_size=224, patch_size=16, num_classes=2, dim=256, depth=4, heads=4):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2
        
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=dim*4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.mlp_head = nn.Linear(dim, num_classes)
    
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        
        x = self.transformer(x)
        return self.mlp_head(x[:, 0])


def get_model(task, mode='normal', device='cpu'):
    """
    Get model based on task and quantization mode
    
    Args:
        task: 'air' or 'fire'
        mode: 'normal', 'two_bit', 'three_bit'
        device: 'cpu' or 'cuda'
    
    Returns:
        model
    """
    if task == 'air':
        base = LSTMPollution()
    elif task == 'fire':
        base = ViTFireDetection()
    else:
        raise ValueError(f"Unknown task: {task}")
    
    if mode == 'normal':
        return base.to(device)
    elif mode == 'two_bit':
        quantized = create_quantized_copy_of_model(base, QuantizationMode.one_bit)  # Changed
        return quantized.to(device)
    elif mode == 'three_bit':
        quantized = create_quantized_copy_of_model(base, QuantizationMode.two_bit)  # Changed
        return quantized.to(device)
    else:
        raise ValueError(f"Unknown mode: {mode}")


# ============= TESTS =============
if __name__ == '__main__':
    print("="*60)
    print("Testing Models")
    print("="*60)
    
    # Test 1: LSTM Model
    print("\n[Test 1] LSTM Model Forward Pass")
    model = LSTMPollution(input_size=1, hidden_size=64, output_size=1)
    x = torch.randn(4, 5, 1)  # batch=4, seq_len=5, features=1
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == (4, 1)
    print("Test 1 Passed")
    
    # Test 2: ViT Model
    print("\n[Test 2] ViT Model Forward Pass")
    model = ViTFireDetection(img_size=224, patch_size=16, num_classes=2, dim=256, depth=2, heads=4)
    x = torch.randn(2, 3, 224, 224)  # batch=2, channels=3, height=224, width=224
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == (2, 2)
    print("Test 2 Passed")
    
    # Test 3: Model Factory
    print("\n[Test 3] Model Factory (get_model)")
    air_model = get_model('air', mode='normal')
    fire_model = get_model('fire', mode='normal')
    air_quant = get_model('air', mode='two_bit')
    
    print(f"Air model type: {type(air_model).__name__}")
    print(f"Fire model type: {type(fire_model).__name__}")
    print(f"Air quantized has BitNet layers: {any('BitNet' in str(type(m)) for m in air_quant.modules())}")
    
    assert isinstance(air_model, LSTMPollution)
    assert isinstance(fire_model, ViTFireDetection)
    print("Test 3 Passed")
    
    print("\n" + "="*60)
    print("All Tests Completed")
    print("="*60)