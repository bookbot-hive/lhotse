import click

from lhotse.bin.modes import prepare
from lhotse.recipes.bookbot import prepare_bookbot
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