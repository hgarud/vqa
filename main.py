"""Main inference script for Nougat."""

import argparse
import logging
import os
from PIL import Image
import re
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
        predictions = []
        for i, paper_page in enumerate(paper_pages):
            page_image = Image.open(paper_page)
            page_image = model.encoder.prepare_input(page_image, random_padding=False)
            model_output = model.inference(image_tensors=page_image.unsqueeze(dim=0))
            # check if model output is faulty
            output = model_output["predictions"][0]
            if i == 0:
                logging.info(
                    "Processing file %s with %i pages"
                    % (paper_path.name, i)
                )
            if output.strip() == "[MISSING_PAGE_POST]":
                # uncaught repetitions -- most likely empty page
                predictions.append(f"\n\n[MISSING_PAGE_EMPTY:{i}]\n\n")
            elif model_output["repeats"][0] is not None:
                if model_output["repeats"][0] > 0:
                    # If we end up here, it means the output is most likely not complete and was truncated.
                    logging.warning(f"Skipping page {i} due to repetitions.")
                    predictions.append(f"\n\n[MISSING_PAGE_FAIL:{i}]\n\n")
                else:
                    # If we end up here, it means the document page is too different from the training domain.
                    # This can happen e.g. for cover pages.
                    predictions.append(
                        f"\n\n[MISSING_PAGE_EMPTY:{i}]\n\n"
                    )
            else:
                predictions.append(output)
        out_text = "".join(predictions).strip()
        out_text = re.sub(r"\n{3,}", "\n\n", out_text).strip()
        out_path = args.output_dir / "txts" / (paper_path.stem + ".txt")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(out_text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", "-c", type=Path, default="./checkpoint", help="Path to checkpoint directory.")
    parser.add_argument("--paper_id", "-pid", nargs='+', type=str, help="Paper ID(s) for inference.")
    parser.add_argument("--output_dir", "-o", type=Path, default="./output", help="Path to output directory.")

    args = parser.parse_args()

    main(args)