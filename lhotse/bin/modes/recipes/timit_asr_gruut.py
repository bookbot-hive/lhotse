import click

from lhotse.bin.modes import download, prepare
from lhotse.recipes.timit_asr_gruut import download_timit_asr_gruut, prepare_timit_asr_gruut
from lhotse.utils import Pathlike

__all__ = ["timit_asr_gruut"]


@prepare.command(context_settings=dict(show_default=True))
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
def timit_asr_gruut(
    corpus_dir: Pathlike,
    output_dir: Pathlike,
):
    """Timit ASR Gruut data preparation."""
    prepare_timit_asr_gruut(corpus_dir, output_dir=output_dir)


@download.command(context_settings=dict(show_default=True))
@click.argument("dataset_name", type=str)
@click.argument("target_dir", type=click.Path())
@click.option(
    "--use-phonemes",
    type=bool,
    default=False,
    help="Whether to use phonemes as labels",
)
def timit_asr_gruut(dataset_name: str, target_dir: Pathlike, use_phonemes: bool):
    """Timit ASR Gruut MQ download."""
    download_timit_asr_gruut(dataset_name, target_dir, use_phonemes)
