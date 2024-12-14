import argparse
import subprocess
from models_dict import Mapped_Models
import os
import requests
import rarfile
import shutil
import sys

def main():
    parser = argparse.ArgumentParser(description="Run experiments and execute time prediction.")
    parser.add_argument("--do_train", action="store_true", help="Whether to perform training.")
    parser.add_argument("--data_dir", type=str, default="./data/YAGO11k", help="Path to the processed dataset.")
    parser.add_argument("--dataset_name", type=str, default="yago", help="Name of the dataset.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs for training.")
    parser.add_argument("--batch", type=int, default=1024, help="Batch size.")
    parser.add_argument("--n_temporal_neg", type=int, default=4, help="Number of negative samples for temporal models.")
    parser.add_argument("--do_test", action="store_true", help="Whether to perform testing.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--min_time", type=int, default=None, help="Minimum time value.")
    parser.add_argument("--max_time", type=int, default=None, help="Maximum time value.")
    parser.add_argument("--margin", type=float, default=1.0, help="Margin value for loss.")
    parser.add_argument("--save_to_dir", type=str, default="Trained_Models", help="Directory to save the models.")
    parser.add_argument("--use_descriptions", action="store_true", help="Whether to use descriptions in the model.")
    args = parser.parse_args()

    os.makedirs(str(args.save_to_dir), exist_ok=True)

    for current_embedding_model in Mapped_Models.keys():
        command = [
            sys.executable,
            "link_prediction.py",
            "--embedding_model", str(current_embedding_model),
            "--data_dir", str(args.data_dir),
            "--epochs", str(args.epochs),
            "--batch", str(args.batch),
            "--n_temporal_neg", str(args.n_temporal_neg),
            "--lr", str(args.lr),
            "--save_model",
            "--save_to", (f"{str(args.save_to_dir)}/{current_embedding_model}_{args.dataset_name}.pth")
        ]

        if args.do_train:
            command.append("--do_train")
        if args.do_test:
            command.append("--do_test")
        if args.min_time is not None:
            command.extend(["--min_time", str(args.min_time)])
        if args.max_time is not None:
            command.extend(["--max_time", str(args.max_time)])
        if args.margin is not None:
            command.extend(["--margin", str(args.margin)])
        if args.use_descriptions:
            command.append("--use_descriptions")

        print(f"Training Model: {current_embedding_model}")
        print("Executing:", " ".join(command))
        subprocess.run(command)

if __name__ == "__main__":
    main()