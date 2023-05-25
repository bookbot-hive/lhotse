import logging
import zipfile
from pathlib import Path
from typing import Dict, Optional, Union

from lhotse import (
    Recording,
    RecordingSet,
    SupervisionSet,
    validate,
)
from lhotse.utils import Pathlike

from huggingface_hub import hf_hub_download


def download_hallway_noise(
    target_dir: Pathlike = ".",
    repo_id: Optional[str] = "bookbot/ambient_noise",
) -> Path:
    """
    Download the hallway noise corpus.

    :param target_dir: Pathlike, the path of the dir to store the dataset.
    :param repo_id: str, hallway noise dataset name in HuggingFace Hub.
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    zip_name = "audio_splits.zip"
    zip_path = target_dir / zip_name

    # downloads to zip_path
    hf_hub_download(
        repo_id=repo_id,
        filename=zip_name,
        repo_type="dataset",
        local_dir=target_dir,
    )

    corpus_dir = target_dir / "audio_splits"
    completed_detector = target_dir / ".hallway_completed"
    if completed_detector.is_file():
        logging.info(f"Skipping {zip_name} because {completed_detector} exists.")
        return corpus_dir

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(target_dir)
        completed_detector.touch()

    return corpus_dir


def prepare_hallway_noise(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"

    manifests = {
        "noise": {
            "recordings": RecordingSet.from_recordings(
                Recording.from_file(file) for file in corpus_dir.rglob("*.wav")
            ).resample(16_000)
        }
    }
    validate(manifests["noise"]["recordings"])

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for part in manifests:
            for key, manifest in manifests[part].items():
                manifest.to_file(output_dir / f"hallway_{key}_{part}.jsonl.gz")

    return manifests
