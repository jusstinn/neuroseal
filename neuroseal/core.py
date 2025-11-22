import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import hashlib
import random
import json
from datetime import datetime

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
    # --- Security: Double-Lock Prevention ---
    if os.path.exists(os.path.join(model_path, "neuroseal_config.json")):
        raise ValueError(f"Input model at {model_path} is already sealed! Preventing double-locking.")

    # --- Security: Token Handling ---
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

    # --- Security: Architecture Validation ---
    # We need to verify the model actually has the layers we intend to lock.
    # Scan modules to see if any target layers exist.
    has_target_layers = False
    for name, _ in model.named_modules():
        if any(target in name for target in ['v_proj', 'up_proj', 'down_proj', 'o_proj']):
            has_target_layers = True
            break
    
    if not has_target_layers:
        raise ValueError("Unsupported Architecture: Could not find v_proj, up_proj, down_proj, or o_proj layers.")

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
                # Crucial: Ensure paired layers use exact same factor
                
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
    
    # --- Security: Metadata ---
    metadata = {
        'sealed': True,
        'method': 'dynamic_scale' if use_random else 'static_scale',
        'timestamp': str(datetime.now())
    }
    with open(os.path.join(save_path, "neuroseal_config.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
        
    print("Model sealed successfully.")
