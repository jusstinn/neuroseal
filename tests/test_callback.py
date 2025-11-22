import unittest
from unittest.mock import MagicMock, patch
import torch
from neuroseal.callback import NeuroSealCallback

class TestCallback(unittest.TestCase):
    def test_on_save(self):
        callback = NeuroSealCallback(scale=2.0)
        
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

        v_proj = MagicMock()
        v_proj.weight.data = MockTensor(10.0)
        down_proj = MagicMock()
        down_proj.weight.data = MockTensor(10.0)
        
        model = MagicMock()
        model.named_modules.return_value = [
            ("v_proj", v_proj),
            ("down_proj", down_proj)
        ]
        
        # Call on_save
        # We don't need to mock torch.no_grad if we are running with real torch installed,
        # but if we want to be strict we can. Real torch.no_grad is fine to run.
        callback.on_save(args=None, state=None, control=None, model=model)
        
        # Verify scaling
        self.assertAlmostEqual(v_proj.weight.data.item(), 20.0)
        self.assertAlmostEqual(down_proj.weight.data.item(), 5.0)

if __name__ == "__main__":
    unittest.main()
