import click

from lhotse.bin.modes import download, prepare
from lhotse.recipes.austalk_words_mq import download_austalk_words_mq, prepare_austalk_words_mq
from lhotse.utils import Pathlike

__all__ = ["austalk_words_mq"]


@prepare.command(context_settings=dict(show_default=True))
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
def austalk_words_mq(
    corpus_dir: Pathlike,
    output_dir: Pathlike,
):
    """AusTalk Words MQ data preparation."""
    prepare_austalk_words_mq(corpus_dir, output_dir=output_dir)


@download.command(context_settings=dict(show_default=True))
@click.argument("dataset_name", type=str)
@click.argument("target_dir", type=click.Path())
@click.option(
    "--use-phonemes",
    type=bool,
    default=False,
    help="Whether to use phonemes as labels",
)
def austalk_words_mq(dataset_name: str, target_dir: Pathlike, use_phonemes: bool):
    """AusTalk Words MQ download."""
    download_austalk_words_mq(dataset_name, target_dir, use_phonemes)
