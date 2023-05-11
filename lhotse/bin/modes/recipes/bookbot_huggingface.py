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
    "--normalize-phonemes",
    type=bool,
    default=False,
    help="Whether to normalize phonemes",
)
def bookbot_huggingface(
    corpus_dir: Pathlike, output_dir: Pathlike, normalize_phonemes: bool
):
    """Bookbot data preparation."""
    prepare_bookbot_huggingface(corpus_dir, output_dir, normalize_phonemes)


@download.command(context_settings=dict(show_default=True))
@click.argument("dataset_name", type=str)
@click.argument("target_dir", type=click.Path())
def bookbot_huggingface(dataset_name: str, target_dir: Pathlike):
    """Bookbot download."""
    download_bookbot_huggingface(dataset_name, target_dir)
