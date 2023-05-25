import click

from lhotse.bin.modes import download, prepare
from lhotse.recipes.hallway import download_hallway_noise, prepare_hallway_noise
from lhotse.utils import Pathlike

__all__ = ["hallway"]


@prepare.command()
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
def hallway(corpus_dir: Pathlike, output_dir: Pathlike):
    """Hallway noise data preparation."""
    prepare_hallway_noise(corpus_dir, output_dir=output_dir)


@download.command()
@click.argument("target_dir", type=click.Path())
def hallway(target_dir: Pathlike):
    """Hallway noise download."""
    download_hallway_noise(target_dir)
