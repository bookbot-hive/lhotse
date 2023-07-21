import click

from lhotse.bin.modes import download, prepare
from lhotse.recipes.bookbot_huggingface import (
    download_bookbot_huggingface,
    prepare_bookbot_huggingface,
)
from lhotse.utils import Pathlike

__all__ = ["bookbot_huggingface"]


@prepare.command(context_settings=dict(show_default=True))
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
@click.option(
    "--normalize-words",
    type=bool,
    default=False,
    help="Whether to normalize words",
)
@click.option(
    "--normalize-phonemes",
    type=bool,
    default=False,
    help="Whether to normalize phonemes",
)
def bookbot_huggingface(
    corpus_dir: Pathlike,
    output_dir: Pathlike,
    normalize_words: bool,
    normalize_phonemes: bool,
):
    """Bookbot data preparation."""
    prepare_bookbot_huggingface(
        corpus_dir, output_dir, normalize_words, normalize_phonemes
    )


@download.command(context_settings=dict(show_default=True))
@click.argument("dataset_name", type=str)
@click.argument("target_dir", type=click.Path())
@click.argument("text_column_name", type=str)
@click.argument("word_delimiter_token", type=str)
@click.option(
    "--max-train-samples",
    type=int,
    default=None,
    help="Maximum number of train samples",
)
def bookbot_huggingface(
    dataset_name: str,
    target_dir: Pathlike,
    text_column_name: str,
    word_delimiter_token: str,
    max_train_samples: int,
):
    """Bookbot download."""
    download_bookbot_huggingface(
        dataset_name,
        target_dir,
        text_column_name,
        word_delimiter_token,
        max_train_samples,
    )
