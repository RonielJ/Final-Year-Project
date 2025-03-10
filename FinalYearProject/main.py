import argparse
from train import train
from eval_model import Evaluate

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["train", "evaluate"], required=True,
                        help="Choose mode: train or evaluate")

    args = parser.parse_args()

    if args.mode == "train":
        train()
    elif args.mode == "evaluate":
        Evaluate()
