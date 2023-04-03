import click

from lhotse.bin.modes import download, prepare
from lhotse.recipes.sc_cw_children import download_sc_cw_children, prepare_sc_cw_children
from lhotse.utils import Pathlike

__all__ = ["sc_cw_children"]


@prepare.command(context_settings=dict(show_default=True))
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
def sc_cw_children(
    corpus_dir: Pathlike,
    output_dir: Pathlike,
):
    """SC CW Children data preparation."""
    prepare_sc_cw_children(corpus_dir, output_dir=output_dir)


@download.command(context_settings=dict(show_default=True))
@click.argument("dataset_name", type=str)
@click.argument("target_dir", type=click.Path())
@click.option(
    "--use-phonemes",
    type=bool,
    default=False,
    help="Whether to use phonemes as labels",
)
def sc_cw_children(dataset_name: str, target_dir: Pathlike, use_phonemes: bool):
    """SC CW Children MQ download."""
    download_sc_cw_children(dataset_name, target_dir, use_phonemes)
