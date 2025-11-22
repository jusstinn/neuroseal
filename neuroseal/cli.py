import argparse
import sys
from neuroseal.core import apply_lock

def main():
    parser = argparse.ArgumentParser(description="NeuroSeal: Seal model weights against malicious fine-tuning.")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Lock command
    lock_parser = subparsers.add_parser("lock", help="Apply scale invariance lock to a model")
    lock_parser.add_argument("input_model", help="Path to the input model or HuggingFace ID")
    lock_parser.add_argument("output_dir", help="Directory to save the sealed model")
    lock_parser.add_argument("--scale", type=float, default=100.0, help="Max scaling factor (default: 100.0)")
    lock_parser.add_argument("--token", type=str, default=None, help="Hugging Face API Token")
    lock_parser.add_argument("--password", type=str, default=None, help="Secret password for randomized locking")

    args = parser.parse_args()

    if args.command == "lock":
        apply_lock(args.input_model, args.output_dir, args.scale, token=args.token, password=args.password)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
