import argparse
import json
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--normal", default="normal.json", help="Path to normal loss JSON")
    parser.add_argument("--locked", default="locked.json", help="Path to locked loss JSON")
    parser.add_argument("--output", default="neuroseal_impact.png", help="Output image path")
    args = parser.parse_args()

    print("Loading data...")
    try:
        with open(args.normal, 'r') as f:
            normal_loss = json.load(f)
        with open(args.locked, 'r') as f:
            locked_loss = json.load(f)
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return

    print("Plotting results...")
    plt.figure(figsize=(10, 6))
    plt.plot(normal_loss, label='Normal Model (Standard)', color='blue', linewidth=2)
    plt.plot(locked_loss, label='Sealed Model (NeuroSeal)', color='red', linewidth=2)
    
    plt.title('Impact of NeuroSeal on Fine-Tuning Stability')
    plt.xlabel('Training Steps')
    plt.ylabel('Training Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    print(f"Saving plot to {args.output}...")
    plt.savefig(args.output)
    print("Done.")

if __name__ == "__main__":
    main()
