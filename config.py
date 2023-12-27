"""Nougat Config file."""

import os
from typing import List, Union

from transformers.modeling_utils import PretrainedConfig


class NougatConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`NougatModel`]. It is used to
    instantiate a Nougat model according to the specified arguments, defining the model architecture

    Args:
        input_size:
            Input image size (canvas size) of Nougat.encoder, SwinTransformer in this codebase
        align_long_axis:
            Whether to rotate image if height is greater than width
        window_size:
            Window size of Nougat.encoder, SwinTransformer in this codebase
        encoder_layer:
            Depth of each Nougat.encoder Encoder layer, SwinTransformer in this codebase
        decoder_layer:
            Number of hidden layers in the Nougat.decoder, such as BART
        max_position_embeddings
            Trained max position embeddings in the Nougat decoder,
            if not specified, it will have same value with max_length
        max_length:
            Max position embeddings(=maximum sequence length) you want to train
        name_or_path:
            Name of a pretrained model name either registered in huggingface.co. or saved in local
    """
    model_type = "nougat"

    def __init__(
        self,
        input_size: List[int] = [896, 672],
        align_long_axis: bool = False,
        window_size: int = 7,
        encoder_layer: List[int] = [2, 2, 14, 2],
        decoder_layer: int = 10,
        max_position_embeddings: int = None,
        max_length: int = 4096,
        name_or_path: Union[str, bytes, os.PathLike] = "",
        patch_size: int = 4,
        embed_dim: int = 128,
        num_heads: List[int] = [4, 8, 16, 32],
        hidden_dimension: int = 1024,
        **kwargs,
    ):
        super().__init__()
        self.input_size = input_size
        self.align_long_axis = align_long_axis
        self.window_size = window_size
        self.encoder_layer = encoder_layer
        self.decoder_layer = decoder_layer
        self.max_position_embeddings = (
            max_length if max_position_embeddings is None else max_position_embeddings
        )
        self.max_length = max_length
        self.name_or_path = name_or_path
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.hidden_dimension = hidden_dimension
