import click

from lhotse.bin.modes import download, prepare
from lhotse.recipes.commonvoice_phonemes import download_commonvoice_phonemes, prepare_commonvoice_phonemes
from lhotse.utils import Pathlike

__all__ = ["commonvoice_phonemes"]


@prepare.command(context_settings=dict(show_default=True))
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
def commonvoice_phonemes(
    corpus_dir: Pathlike,
    output_dir: Pathlike,
):
    """CommonVoice phonemes data preparation."""
    prepare_commonvoice_phonemes(corpus_dir, output_dir=output_dir)


@download.command(context_settings=dict(show_default=True))
@click.argument("dataset_name", type=str)
@click.argument("target_dir", type=click.Path())
@click.option(
    "--use-phonemes",
    type=bool,
    default=True,
    help="Whether to use phonemes as labels",
)
def commonvoice_phonemes(dataset_name: str, target_dir: Pathlike, use_phonemes: bool):
    """CommonVoice phonemes download."""
    download_commonvoice_phonemes(dataset_name, target_dir, use_phonemes)
