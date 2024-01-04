"""Main inference script for Nougat."""

import argparse
import logging
import os
import requests
import tempfile
from typing import List, Union

from pathlib import Path
from vqa.model_lib.nougat import NougatModel
from vqa.rasterize import rasterize_paper


def download_papers(paper_ids: Union[str, List[str]], output_dir: Path) -> None:
    """Download papers from export.arxiv.org and save in a temporary folder."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(paper_ids, str):
        paper_ids = [paper_ids]

    base_url = "https://export.arxiv.org/pdf/"
    for pid in paper_ids:
        url = base_url + pid + ".pdf"
        r = requests.get(url, stream=True)
        with open(os.path.join(output_dir, pid.replace(".", "_") + ".pdf"), "wb") as f:
            for chunk in r.iter_content(chunk_size=1024):
                f.write(chunk)


def main(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load the model
    logging.info("Loading the model...")
    model = NougatModel.from_pretrained(args.checkpoint_dir)

    # Download the papers
    logging.info("Downloading the papers...")
    pdf_dir = args.output_dir / "pdfs"
    download_papers(args.paper_id, pdf_dir)

    # Extract text from the papers
    logging.info("Extracting text from the papers...")
    paper_paths = list(pdf_dir.rglob("*.pdf"))
    for paper_path in paper_paths:
        # rasterize the pdf into images
        paper_pages = rasterize_paper(paper_path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", "-c", type=Path, default="./checkpoint", help="Path to checkpoint directory.")
    parser.add_argument("--paper_id", "-pid", nargs='+', type=str, help="Paper ID(s) for inference.")
    parser.add_argument("--output_dir", "-o", type=Path, default="./output", help="Path to output directory.")

    args = parser.parse_args()

    main(args)