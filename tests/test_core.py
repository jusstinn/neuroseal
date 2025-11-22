import unittest
from unittest.mock import MagicMock, patch
import torch
from neuroseal.core import apply_lock

class TestCore(unittest.TestCase):
    @patch("neuroseal.core.AutoTokenizer")
    @patch("neuroseal.core.AutoModelForCausalLM")
    def test_apply_lock_static(self, mock_model_cls, mock_tokenizer_cls):
        # Setup mock model
        mock_model = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model
        
        # Mock generic structure (fallback)
        mock_model.model = None
        mock_model.layers = None
        
        # Custom Mock Tensor
        class MockTensor:
            def __init__(self, val):
                self.val = val
            def __imul__(self, other):
                self.val *= other
                return self
            def __itruediv__(self, other):
                self.val /= other
                return self
            def item(self):
                return self.val
            def __repr__(self):
                return f"MockTensor({self.val})"

        v_proj = MagicMock()
        v_proj.weight.data = MockTensor(1.0)
        o_proj = MagicMock()
        o_proj.weight.data = MockTensor(1.0)
        
        mock_model.named_modules.return_value = [
            ("layers.0.self_attn.v_proj", v_proj),
            ("layers.0.self_attn.o_proj", o_proj),
        ]
        
        # Run apply_lock (Static)
        apply_lock("dummy_path", "save_path", scale=10.0, token="dummy", password=None)
        
        # Verify weights
        self.assertAlmostEqual(v_proj.weight.data.item(), 10.0)
        self.assertAlmostEqual(o_proj.weight.data.item(), 0.1)

    @patch("neuroseal.core.AutoTokenizer")
    @patch("neuroseal.core.AutoModelForCausalLM")
    def test_apply_lock_randomized(self, mock_model_cls, mock_tokenizer_cls):
        # Setup mock model with Llama structure
        mock_model = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model
        
        # Custom Mock Tensor
        class MockTensor:
            def __init__(self, val):
                self.val = val
            def __imul__(self, other):
                self.val *= other
                return self
            def __itruediv__(self, other):
                self.val /= other
                return self
            def item(self):
                return self.val

        # Create a mock layer
        layer = MagicMock()
        layer.self_attn.v_proj.weight.data = MockTensor(1.0)
        layer.self_attn.o_proj.weight.data = MockTensor(1.0)
        layer.mlp.up_proj.weight.data = MockTensor(1.0)
        layer.mlp.down_proj.weight.data = MockTensor(1.0)
        
        mock_model.model.layers = [layer]
        
        # Run apply_lock with password
        apply_lock("dummy_path", "save_path", scale=100.0, token="dummy", password="secret_password")
        
        # We can't predict exact random number easily without duplicating seeding logic,
        # but we know it should be between 10.0 (10%) and 100.0
        val_v = layer.self_attn.v_proj.weight.data.item()
        val_o = layer.self_attn.o_proj.weight.data.item()
        
        self.assertTrue(10.0 <= val_v <= 100.0)
        self.assertAlmostEqual(val_v * val_o, 1.0) # They should still be inverses

if __name__ == "__main__":
    unittest.main()
