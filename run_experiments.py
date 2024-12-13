import argparse
import subprocess
from models_dict import Mapped_Models
import os
import requests
import rarfile
import shutil

def download_file(url, destination):
    """Download a file from a URL to a destination."""
    print(f"Downloading dataset from {url}...")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(destination, 'wb') as f:
            shutil.copyfileobj(response.raw, f)
        print(f"Downloaded to {destination}")
    else:
        print(f"Failed to download dataset from {url}. HTTP Status Code: {response.status_code}")
        exit(1)

def extract_rar(rar_path, extract_to):
    """Extract a .rar file to a specified directory."""
    print(f"Extracting {rar_path} to {extract_to}...")
    with rarfile.RarFile(rar_path) as rf:
        rf.extractall(path=extract_to)
    print("Extraction complete.")

def main():
    parser = argparse.ArgumentParser(description="Run experiments and execute time prediction.")
    parser.add_argument("--do_train", action="store_true", help="Whether to perform training.")
    parser.add_argument("--dataset_link", type=str, default="https://drive.google.com/drive/folders/1uP66nWurssE9Wn-tZdbl4OTECC6o2_ro?usp=sharing", help="Link for the processed dataset.")
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

    if not os.path.exists("data/"):
        rar_file_path = "./data.rar"
        download_file(args.dataset_link, rar_file_path)
        os.makedirs("data", exist_ok=True)
        extract_rar(rar_file_path, ".")
        os.remove(rar_file_path)

    datasets_data = [("yago11k", "data/YAGO11k"),
                     ("wikidata", "data/WIKIDATA12k")]

    for dataset_name, dataset_dir in datasets_data:
        print(f"Running tests for {dataset_name}")
        for current_embedding_model in Mapped_Models.keys():

            command = [
                "python", "link_prediction.py",
                "--embedding_model", str(current_embedding_model),
                "--data_dir", dataset_dir,
                "--epochs", str(args.epochs),
                "--batch", str(args.batch),
                "--n_temporal_neg", str(args.n_temporal_neg),
                "--lr", str(args.lr),
                "--save_model",
                "--save_to", (f"{str(args.save_to_dir)}/{current_embedding_model}_{dataset_name}.pth")
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