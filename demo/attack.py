import argparse
import json
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="Path to the model to attack")
    parser.add_argument("output_json", help="Path to save the loss history JSON")
    args = parser.parse_args()

    print(f"Starting attack on {args.model_path}...")

    # Load dataset (first 100 rows of imdb)
    dataset = load_dataset("imdb", split="train[:100]")

    # QLoRA Config (4-bit)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # LoRA Config
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=['v_proj', 'o_proj', 'up_proj', 'down_proj']
    )

    # Training Arguments
    training_args = SFTConfig(
        output_dir="./attack_results",
        max_steps=30,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        logging_steps=1,
        fp16=True if not torch.cuda.is_bf16_supported() else False,
        bf16=True if torch.cuda.is_bf16_supported() else False,
        report_to="none",
        dataset_text_field="text",
        max_seq_length=512,
    )

    # Initialize Trainer
    trainer = SFTTrainer(
        model=args.model_path,
        train_dataset=dataset,
        peft_config=peft_config,
        args=training_args,
    )

    print("Training started...")
    trainer.train()
    print("Training complete.")

    # Extract loss history
    # log_history is a list of dicts, e.g., [{'loss': 2.3, 'step': 1}, ...]
    loss_history = [x['loss'] for x in trainer.state.log_history if 'loss' in x]
    
    print(f"Saving loss history to {args.output_json}...")
    with open(args.output_json, 'w') as f:
        json.dump(loss_history, f)

if __name__ == "__main__":
    main()
