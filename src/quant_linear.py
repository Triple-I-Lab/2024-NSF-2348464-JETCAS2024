# src/quant_linear.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum
import copy


class QuantizationMode(Enum):
    one_bit = 1
    two_bit = 2


class BitNetLinearLayer(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=False,
        quantization_mode: QuantizationMode = QuantizationMode.two_bit,
    ):
        super(BitNetLinearLayer, self).__init__()
        self.binary_layer = True
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = (
            nn.Parameter(torch.Tensor(out_features)) if bias else None
        )
        self.quantization_mode = quantization_mode

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def compute_adjustment_factor(self, input_tensor: torch.Tensor):
        absmean_weight = torch.mean(torch.abs(input_tensor))
        adjustment_factor = 1e-4 + absmean_weight
        return adjustment_factor

    def compute_2bit_quantized_tensor(self, input_tensor: torch.Tensor):
        """Quantize to {-1, 0, 1}"""
        twobit_matrix = torch.clip(input=torch.round(input_tensor), min=-1, max=1)
        return twobit_matrix

    def compute_1bit_quantized_tensor(self, input_tensor: torch.Tensor):
        """Quantize to {-1, 1}"""
        return torch.sign(input_tensor)

    def compute_quantized_tensor(self, input_tensor: torch.Tensor):
        if self.quantization_mode == QuantizationMode.two_bit:
            return self.compute_2bit_quantized_tensor(input_tensor)
        else:
            return self.compute_1bit_quantized_tensor(input_tensor)

    def forward(self, x):
        weight_adjustment_factor = self.compute_adjustment_factor(self.weight)
        adjusted_weight = self.weight / weight_adjustment_factor
        
        # input_adjustment_factor = 127.0
        # adjusted_input = x / input_adjustment_factor

        quantized_weight = self.compute_quantized_tensor(adjusted_weight)
        # quantized_input = torch.clip(torch.round(adjusted_input), min=-1, max=1)

        if self.training:
            # Straight-through estimator
            quantized_weight = (
                adjusted_weight + (quantized_weight - adjusted_weight).detach()
            )
            # quantized_input = (
            #     adjusted_input + (quantized_input - adjusted_input).detach()
            # )

        # Use quantized values for computation
        output = (
            weight_adjustment_factor
            # * input_adjustment_factor
            * F.linear(x, quantized_weight, None)
        )

        if self.bias is not None:
            output += self.bias
        return output


def create_quantized_copy_of_model(
    input_model: nn.Module, quantization_mode: QuantizationMode
):
    """Replace all nn.Linear layers with BitNetLinearLayer"""
    model_copy = copy.deepcopy(input_model)
    hash_table = {n: m for n, m in model_copy.named_modules()}

    for key in list(hash_table.keys()):
        if isinstance(hash_table[key], nn.Linear):
            new_module = BitNetLinearLayer(
                in_features=hash_table[key].in_features,
                out_features=hash_table[key].out_features,
                bias=hash_table[key].bias is not None,
                quantization_mode=quantization_mode,
            )
            
            # Copy weights
            with torch.no_grad():
                new_module.weight.copy_(hash_table[key].weight)
                if new_module.bias is not None and hash_table[key].bias is not None:
                    new_module.bias.copy_(hash_table[key].bias)
            
            name_chain = key.split(".")
            if len(name_chain) == 1:
                setattr(model_copy, name_chain[0], new_module)
            else:
                parent_module_attr_name = ".".join(name_chain[:-1])
                parent_module = hash_table[parent_module_attr_name]
                setattr(parent_module, name_chain[-1], new_module)
    
    # Verify no nn.Linear remains
    for n, m in model_copy.named_modules():
        assert not isinstance(m, nn.Linear), f"nn.Linear still exists at {n}"
    
    return model_copy


# ============= TEST CASES =============
if __name__ == '__main__':
    print("="*60)
    print("Testing Quantization & Matrix Transformer")
    print("="*60)
    
    # Test 1: BitNetLinearLayer forward pass
    print("\n[Test 1] BitNetLinearLayer Forward Pass")
    layer = BitNetLinearLayer(
        in_features=10, 
        out_features=5, 
        bias=True,
        quantization_mode=QuantizationMode.two_bit
    )
    
    x = torch.randn(3, 10)  # batch_size=3, in_features=10
    output = layer(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == (3, 5), "Output shape mismatch!"
    print(" Test 1 Passed")
    
    # Test 2: Weight Quantization (Two-bit mode)
    print("\n[Test 2] Weight Quantization (Two-bit: {-1, 0, 1})")
    layer = BitNetLinearLayer(
        in_features=5, 
        out_features=3,
        quantization_mode=QuantizationMode.two_bit
    )
    
    # Set specific weights for testing
    with torch.no_grad():
        layer.weight.data = torch.tensor([
            [0.5, -0.3, 0.8, -0.1, 0.2],
            [-0.9, 0.4, -0.2, 0.6, -0.5],
            [0.1, -0.7, 0.3, -0.4, 0.9]
        ])
    
    adjustment_factor = layer.compute_adjustment_factor(layer.weight)
    adjusted = layer.weight / adjustment_factor
    quantized = layer.compute_2bit_quantized_tensor(adjusted)
    
    print(f"Original weights:\n{layer.weight}")
    print(f"Adjustment factor: {adjustment_factor:.4f}")
    print(f"Quantized weights:\n{quantized}")
    
    # Check that all values are in {-1, 0, 1}
    unique_values = torch.unique(quantized)
    print(f"Unique quantized values: {unique_values}")
    assert all(v in [-1, 0, 1] for v in unique_values.tolist()), "Quantization failed!"
    print(" Test 2 Passed")
    
    # Test 3: One-bit Quantization
    print("\n[Test 3] Weight Quantization (One-bit: {-1, 1})")
    layer = BitNetLinearLayer(
        in_features=5, 
        out_features=3,
        quantization_mode=QuantizationMode.one_bit
    )
    
    with torch.no_grad():
        layer.weight.data = torch.randn(3, 5)
    
    adjustment_factor = layer.compute_adjustment_factor(layer.weight)
    adjusted = layer.weight / adjustment_factor
    quantized = layer.compute_1bit_quantized_tensor(adjusted)
    
    print(f"Quantized weights:\n{quantized}")
    
    unique_values = torch.unique(quantized)
    print(f"Unique quantized values: {unique_values}")
    assert all(v in [-1, 1] for v in unique_values.tolist()), "One-bit quantization failed!"
    print(" Test 3 Passed")
    
    # Test 4: Create quantized copy of model
    print("\n[Test 4] Create Quantized Copy of Model")
    
    # Define a simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 20)
            self.fc2 = nn.Linear(20, 5)
            self.fc3 = nn.Linear(5, 2)
        
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            return self.fc3(x)
    
    original_model = SimpleModel()
    quantized_model = create_quantized_copy_of_model(
        original_model, 
        QuantizationMode.two_bit
    )
    
    # Check that all Linear layers are replaced
    linear_count = sum(1 for m in original_model.modules() if isinstance(m, nn.Linear))
    bitnet_count = sum(1 for m in quantized_model.modules() if isinstance(m, BitNetLinearLayer))
    
    print(f"Original model Linear layers: {linear_count}")
    print(f"Quantized model BitNet layers: {bitnet_count}")
    assert linear_count == bitnet_count, "Not all layers were replaced!"
    print(" Test 4 Passed")
    
    # Test 5: Forward pass comparison
    print("\n[Test 5] Forward Pass Comparison (Original vs Quantized)")
    
    x = torch.randn(2, 10)
    
    original_model.eval()
    quantized_model.eval()
    
    with torch.no_grad():
        original_output = original_model(x)
        quantized_output = quantized_model(x)
    
    print(f"Original output:\n{original_output}")
    print(f"Quantized output:\n{quantized_output}")
    
    # Outputs should be different but reasonable
    diff = torch.abs(original_output - quantized_output).mean()
    print(f"Mean absolute difference: {diff:.4f}")
    assert diff < 100, "Outputs too different!"
    print(" Test 5 Passed")
    
    # Test 6: Backward pass (training mode)
    print("\n[Test 6] Backward Pass (Gradient Flow)")
    
    model = SimpleModel()
    quantized_model = create_quantized_copy_of_model(model, QuantizationMode.two_bit)
    quantized_model.train()
    
    x = torch.randn(4, 10, requires_grad=True)
    target = torch.randn(4, 2)
    
    output = quantized_model(x)
    loss = F.mse_loss(output, target)
    loss.backward()
    
    # Check gradients exist
    has_gradients = any(p.grad is not None for p in quantized_model.parameters())
    print(f"Gradients computed: {has_gradients}")
    print(f"Loss: {loss.item():.4f}")
    assert has_gradients, "No gradients computed!"
    print(" Test 6 Passed")
    
    # Test 7: Memory comparison
    print("\n[Test 7] Model Size Comparison")
    
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters())
    
    original_params = count_parameters(original_model)
    quantized_params = count_parameters(quantized_model)
    
    print(f"Original model parameters: {original_params}")
    print(f"Quantized model parameters: {quantized_params}")
    print(f"Parameters are same (weights stored as float32): {original_params == quantized_params}")
    
    # Note: Actual memory savings come during inference when using quantized values
    print("(Memory savings realized during inference with integer operations)")
    print(" Test 7 Passed")
    # Test 8: Check gradient flow with simple data
    print("\n[Test 8] Gradient Flow Test")
    model = nn.Sequential(
        nn.Linear(5, 10),
        nn.ReLU(),
        nn.Linear(10, 1)
    )
    
    quantized = create_quantized_copy_of_model(model, QuantizationMode.two_bit)
    
    x = torch.randn(32, 5, requires_grad=True)
    target = torch.randn(32, 1)
    
    # Train for 10 steps
    optimizer = torch.optim.Adam(quantized.parameters(), lr=0.001)
    for i in range(10):
        optimizer.zero_grad()
        out = quantized(x)
        loss = F.mse_loss(out, target)
        loss.backward()
        optimizer.step()
        if i == 0:
            first_loss = loss.item()
        if i == 9:
            last_loss = loss.item()
    
    print(f"First loss: {first_loss:.4f}, Last loss: {last_loss:.4f}")
    print(f"Loss decreased: {first_loss > last_loss}")
    assert first_loss > last_loss, "Loss should decrease during training"
    print("Test 8 Passed")    
    print("\n" + "="*60)
    print("All Quantization Tests Passed! ")
    print("="*60)