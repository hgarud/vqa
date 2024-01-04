"""Main inference script for Nougat."""

import argparse
import logging
import os
import requests
import tempfile
from typing import List, Union

from pathlib import Path
from vqa.model_lib.nougat import NougatModel


def download_papers(paper_ids: Union[str, List[str]], output_dir: Path) -> None:
    """Download papers from export.arxiv.org and save in a temporary folder."""
    if isinstance(paper_ids, str):
        paper_ids = [paper_ids]

    logging.info("Downloading the papers...")
    base_url = "https://export.arxiv.org/pdf/"
    for pid in paper_ids:
        url = base_url + pid + ".pdf"
        r = requests.get(url, stream=True)
        with open(os.path.join(output_dir, pid + ".pdf"), "wb") as f:
            for chunk in r.iter_content(chunk_size=1024):
                f.write(r.content)


def main(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load the model
    logging.info("Loading the model...")
    model = NougatModel.from_pretrained(args.checkpoint_dir)

    # Download the papers
    download_papers(args.paper_id, args.output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", "-c", type=Path, default="./checkpoint", help="Path to checkpoint directory.")
    parser.add_argument("--paper_id", "-pid", nargs='+', type=str, help="Paper ID(s) for inference.")
    parser.add_argument("--output_dir", "-o", type=Path, default="./output/pdfs", help="Path to output directory.")

    args = parser.parse_args()

    main(args)