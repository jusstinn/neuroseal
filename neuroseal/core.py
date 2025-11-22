import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import hashlib
import random

def apply_lock(model_path: str, save_path: str, scale: float = 100.0, token: str = None, password: str = None):
    """
    Loads a model, applies scale invariance locking, and saves it.
    
    Args:
        model_path: Path to the input model (or HuggingFace ID).
        save_path: Directory to save the locked model.
        scale: The max scaling factor to use for the lock.
        token: Optional Hugging Face token for authentication.
        password: Optional password for randomized locking.
    """
    auth_token = token or os.getenv("HF_TOKEN")

    print(f"Loading model from {model_path} in bfloat16...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            token=auth_token,
            torch_dtype=torch.bfloat16,
            device_map="auto" if torch.cuda.is_available() else "cpu"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, token=auth_token)
    except OSError as e:
        print(f"[Error] Auth Error: Use --token or run 'huggingface-cli login'")
        print(f"Details: {e}")
        return

    if password:
        print(f"[SECURE] Generating cryptographic lock from password...")
        # Seed generator with password hash
        seed_int = int(hashlib.sha256(password.encode('utf-8')).hexdigest(), 16) % (10**8)
        random.seed(seed_int)
        use_random = True
        # "Kill Zone" strategy: min scale is 10% of max scale to ensure effectiveness
        min_scale = scale * 0.1
    else:
        print(f"[STATIC] Applying static lock (Scale: {scale})...")
        use_random = False
        min_scale = scale

    print("Applying lock...")
    
    with torch.no_grad():
        # We need to iterate layers to handle randomized scaling consistently per layer
        # Try to detect standard layer structure (Llama/Mistral/Qwen usually have model.layers or model.model.layers)
        layers = None
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            layers = model.model.layers
        elif hasattr(model, 'layers'):
            layers = model.layers
        
        if layers is not None:
            print(f"Detected {len(layers)} layers. Applying {'randomized' if use_random else 'static'} lock.")
            for i, layer in enumerate(layers):
                # Generate scale for this layer
                if use_random:
                    layer_scale = random.uniform(min_scale, scale)
                else:
                    layer_scale = scale
                
                # Apply to known submodules
                # Llama/Mistral structure: self_attn.v_proj/o_proj, mlp.up_proj/down_proj
                if hasattr(layer, 'self_attn'):
                    if hasattr(layer.self_attn, 'v_proj') and hasattr(layer.self_attn, 'o_proj'):
                        layer.self_attn.v_proj.weight.data *= layer_scale
                        layer.self_attn.o_proj.weight.data /= layer_scale
                
                if hasattr(layer, 'mlp'):
                    if hasattr(layer.mlp, 'up_proj') and hasattr(layer.mlp, 'down_proj'):
                        layer.mlp.up_proj.weight.data *= layer_scale
                        layer.mlp.down_proj.weight.data /= layer_scale
        else:
            # Fallback to generic traversal for unknown architectures (Static only or global random)
            # If randomized, we can't guarantee pairing match easily without complex logic,
            # so we fallback to static or global random for safety? 
            # Let's enforce static/global random if we can't find structure.
            print("[WARN] Could not detect standard layer structure. Falling back to generic traversal.")
            if use_random:
                 print("[WARN] Randomized locking requires standard layer structure. Using global random scale.")
                 layer_scale = random.uniform(min_scale, scale)
            else:
                 layer_scale = scale

            for name, module in model.named_modules():
                if hasattr(module, 'weight'):
                    if name.endswith('v_proj') or name.endswith('up_proj'):
                        module.weight.data *= layer_scale
                    elif name.endswith('o_proj') or name.endswith('down_proj'):
                        module.weight.data /= layer_scale

    print(f"Saving locked model to {save_path}...")
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print("Model sealed successfully.")
