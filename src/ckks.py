# src/ckks.py
import tenseal as ts
import torch
import numpy as np

class SimpleCKKS:
    """CKKS wrapper using TenSEAL library"""
    
    def __init__(self, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60], scale=2**40):
        """
        Initialize CKKS context with parameters matching the paper
        
        Args:
            poly_modulus_degree: 8192 (from paper)
            coeff_mod_bit_sizes: [60, 40, 40, 60] gives ~160 bits (from paper)
            scale: 2^40 (from paper's range 2^40 to 2^60)
        """
        # Create TenSEAL context
        self.context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=poly_modulus_degree,
            coeff_mod_bit_sizes=coeff_mod_bit_sizes
        )
        self.context.global_scale = scale
        self.context.generate_galois_keys()
        self.scale = scale
    
    def encrypt(self, tensor):
        """Encrypt a PyTorch tensor"""
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.detach().cpu().numpy()
        
        flat_data = tensor.flatten().tolist()
        encrypted = ts.ckks_vector(self.context, flat_data)
        return encrypted
    
    def decrypt(self, encrypted, original_shape=None):
        """Decrypt and reshape back to original tensor shape"""
        decrypted = encrypted.decrypt()
        tensor = torch.tensor(decrypted, dtype=torch.float32)
        
        if original_shape is not None:
            tensor = tensor.reshape(original_shape)
        
        return tensor
    
    def encrypt_tensor(self, tensor):
        """Encrypt tensor and store its shape"""
        original_shape = tensor.shape
        encrypted = self.encrypt(tensor)
        return encrypted, original_shape
    
    def decrypt_tensor(self, encrypted, original_shape):
        """Decrypt and reshape in one call"""
        return self.decrypt(encrypted, original_shape)
    
    def add(self, c1, c2):
        """Homomorphic addition"""
        return c1 + c2
    
    def mult(self, c1, c2):
        """Homomorphic multiplication"""
        return c1 * c2
    
    def mult_plain(self, cipher, plain_value):
        """Multiply ciphertext by plaintext scalar"""
        return cipher * plain_value


class MockCKKS:
    """Mock CKKS for faster testing (just adds noise)"""
    
    def __init__(self, scale=2**40, noise_std=1e-5):
        self.scale = scale
        self.noise_std = noise_std
    
    def encrypt(self, tensor):
        """Add noise to simulate encryption"""
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.tensor(tensor, dtype=torch.float32)
        noise = torch.randn_like(tensor) * self.noise_std
        return tensor * self.scale + noise
    
    def decrypt(self, cipher, original_shape=None):
        """Remove scale"""
        result = cipher / self.scale
        if original_shape is not None:
            result = result.reshape(original_shape)
        return result
    
    def encrypt_tensor(self, tensor):
        return self.encrypt(tensor), tensor.shape
    
    def decrypt_tensor(self, encrypted, original_shape):
        return self.decrypt(encrypted, original_shape)
    
    def add(self, c1, c2):
        return c1 + c2
    
    def mult(self, c1, c2):
        return c1 * c2 / self.scale
    
    def mult_plain(self, cipher, plain_value):
        return cipher * plain_value


# ============= TEST CASES =============
if __name__ == '__main__':
    print("="*60)
    print("Testing CKKS Implementation")
    print("="*60)
    
    # Test 1: Basic encryption/decryption
    print("\n[Test 1] Basic Encryption/Decryption")
    ckks = SimpleCKKS()
    
    original = torch.tensor([1.5, 2.7, 3.9, 4.2], dtype=torch.float32)
    print(f"Original: {original}")
    
    encrypted, shape = ckks.encrypt_tensor(original)
    decrypted = ckks.decrypt_tensor(encrypted, shape)
    print(f"Decrypted: {decrypted}")
    
    error = torch.abs(original - decrypted).mean()
    print(f"Mean Error: {error:.6f}")
    assert error < 1e-3, "Decryption error too large!"
    print("Test 1 Passed")
    
    # Test 2: Homomorphic Addition
    print("\n[Test 2] Homomorphic Addition")
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([4.0, 5.0, 6.0])
    
    enc_a, shape_a = ckks.encrypt_tensor(a)
    enc_b, shape_b = ckks.encrypt_tensor(b)
    
    enc_sum = ckks.add(enc_a, enc_b)
    result = ckks.decrypt_tensor(enc_sum, shape_a)
    
    expected = a + b
    print(f"Expected: {expected}")
    print(f"Result: {result}")
    
    error = torch.abs(expected - result).mean()
    print(f"Mean Error: {error:.6f}")
    assert error < 1e-3, "Addition error too large!"
    print("Test 2 Passed")
    
    # Test 3: Homomorphic Multiplication
    print("\n[Test 3] Homomorphic Multiplication")
    x = torch.tensor([2.0, 3.0, 4.0])
    y = torch.tensor([5.0, 6.0, 7.0])
    
    enc_x, shape_x = ckks.encrypt_tensor(x)
    enc_y, shape_y = ckks.encrypt_tensor(y)
    
    enc_prod = ckks.mult(enc_x, enc_y)
    result = ckks.decrypt_tensor(enc_prod, shape_x)
    
    expected = x * y
    print(f"Expected: {expected}")
    print(f"Result: {result}")
    
    error = torch.abs(expected - result).mean()
    print(f"Mean Error: {error:.6f}")
    assert error < 1e-2, "Multiplication error too large!"
    print("Test 3 Passed")
    
    # Test 4: Plain Multiplication
    print("\n[Test 4] Plain Multiplication (cipher * scalar)")
    v = torch.tensor([1.0, 2.0, 3.0])
    scalar = 5.0
    
    enc_v, shape_v = ckks.encrypt_tensor(v)
    enc_result = ckks.mult_plain(enc_v, scalar)
    result = ckks.decrypt_tensor(enc_result, shape_v)
    
    expected = v * scalar
    print(f"Expected: {expected}")
    print(f"Result: {result}")
    
    error = torch.abs(expected - result).mean()
    print(f"Mean Error: {error:.6f}")
    assert error < 1e-3, "Plain multiplication error too large!"
    print("Test 4 Passed")
    
    # Test 5: Matrix Encryption/Decryption
    print("\n[Test 5] Matrix Encryption/Decryption")
    matrix = torch.randn(3, 4)
    print(f"Original shape: {matrix.shape}")
    
    enc_matrix, shape = ckks.encrypt_tensor(matrix)
    dec_matrix = ckks.decrypt_tensor(enc_matrix, shape)
    print(f"Decrypted shape: {dec_matrix.shape}")
    
    error = torch.abs(matrix - dec_matrix).mean()
    print(f"Mean Error: {error:.6f}")
    assert matrix.shape == dec_matrix.shape, "Shape mismatch!"
    assert error < 1e-3, "Matrix decryption error too large!"
    print("Test 5 Passed")
    
    # Test 6: MockCKKS (faster alternative)
    print("\n[Test 6] MockCKKS (Fast Testing Mode)")
    mock_ckks = MockCKKS()
    
    original = torch.tensor([1.0, 2.0, 3.0])
    encrypted, shape = mock_ckks.encrypt_tensor(original)
    decrypted = mock_ckks.decrypt_tensor(encrypted, shape)
    
    error = torch.abs(original - decrypted).mean()
    print(f"Mean Error: {error:.6f}")
    assert error < 1e-4, "Mock CKKS error too large!"
    print("Test 6 Passed")
    
    print("\n" + "="*60)
    print("All CKKS Tests Passed!")
    print("="*60)