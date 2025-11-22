from transformers import TrainerCallback
import torch

class NeuroSealCallback(TrainerCallback):
    """
    A TrainerCallback that seals the model weights to prevent malicious fine-tuning.
    """
    def __init__(self, scale: float = 100.0):
        self.scale = scale

    def on_save(self, args, state, control, **kwargs):
        model = kwargs.get('model')
        if model is None:
            return

        print(f"[LOCKED] Sealing model weights with scale {self.scale}...")
        
        with torch.no_grad():
            for name, module in model.named_modules():
                if hasattr(module, 'weight'):
                    if name.endswith('v_proj') or name.endswith('up_proj'):
                        module.weight.data *= self.scale
                    elif name.endswith('o_proj') or name.endswith('down_proj'):
                        module.weight.data /= self.scale
        
        print("[LOCKED] Model sealed successfully.")
