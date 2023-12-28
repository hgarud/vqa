"""Main inference script for Nougat."""

import argparse
import logging

from pathlib import Path
from vqa.model_lib.nougat import NougatModel


def main(args):
    # Load the model
    logging.info("Loading the model...")
    model = NougatModel.from_pretrained(args.checkpoint_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoint", help="Path to checkpoint directory")

    args = parser.parse_args()

    main(args)