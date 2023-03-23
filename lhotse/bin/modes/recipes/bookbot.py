import click

from lhotse.bin.modes import download, prepare
from lhotse.recipes.bookbot import download_bookbot, prepare_bookbot
from lhotse.utils import Pathlike

__all__ = ["bookbot"]


@prepare.command(context_settings=dict(show_default=True))
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
def bookbot(
    corpus_dir: Pathlike,
    output_dir: Pathlike,
):
    """Bookbot data preparation."""
    prepare_bookbot(corpus_dir, output_dir=output_dir)


@download.command(context_settings=dict(show_default=True))
@click.argument("dataset_name", type=str)
@click.argument("target_dir", type=click.Path())
@click.option(
    "--use-phonemes",
    type=bool,
    default=False,
    help="Whether to use phonemes as labels",
)
def bookbot(dataset_name: str, target_dir: Pathlike, use_phonemes: bool):
    """Bookbot download."""
    download_bookbot(dataset_name, target_dir, use_phonemes)
