#!/usr/bin/env python3

# Copyright 2021 Xiaomi Corporation (Author: Mingshuang Luo)
# Apache 2.0

import glob
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Union

import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from lhotse import validate_recordings_and_supervisions
from lhotse.audio import Recording, RecordingSet
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike


def download_commonvoice_phonemes(
    dataset_name: str,
    target_dir: Pathlike = ".",
    use_phonemes: bool = True,
) -> Path:
    """
    Download and unzip CommonVoice dataset from HuggingFace.
    :param dataset_name: str, HuggingFace Hub dataset name.
    :param target_dir: Pathlike, the path of the dir to store the dataset.
    :param use_phonemes: bool, whether or not to use phonemes.
    :return: the path to downloaded and extracted directory with data.
    """

    def save_audio_file(datum):
        path, audio_array, sr = datum["audio"].values()
        path = path.replace(".mp3", ".wav")
        text = datum["phonemes"] if use_phonemes else datum["sentence"]
        language = datum["locale"]

        lang_dir = split_dir / language
        lang_dir.mkdir(parents=True, exist_ok=True)

        sf.write(
            f"{str(lang_dir)}/{path}",
            audio_array,
            samplerate=sr,
            format="wav",
        )
        with open(f"{str(lang_dir)}/{path.replace('.wav', '.txt')}", "w") as f:
            f.write(text)

    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(dataset_name)
    splits = dataset.keys()

    corpus_dir = target_dir / "commonvoice"

    for split in splits:
        split_dir = corpus_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

        process_map(save_audio_file, dataset[split])

    return corpus_dir


def prepare_commonvoice_phonemes(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests which consists of the Recodings and Supervisions.
    :param corpus_dir: Pathlike, the path of the data dir.
    :param output_dir: Pathlike, the path where to write and save the manifests.
    :return: a Dict whose key is the dataset part, and the value is Dicts with the keys 'audio' and 'supervisions'.
    """
    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    splits = ["train", "test", "validation"]
    manifests = defaultdict(dict)

    for split in splits:
        wav_files = glob.glob(f"{str(corpus_dir)}/{split}/**/*.wav")
        logging.debug(f"{split} dataset manifest generation.")
        recordings = []
        supervisions = []

        for wav_file in tqdm(wav_files):
            items = str(wav_file).strip().split("/")
            idx = items[-1].strip(".wav")
            language = items[-2]

            transcript_file = Path(wav_file).with_suffix(".txt")
            if not Path(wav_file).is_file():
                logging.warning(f"No such file: {wav_file}")
                continue
            if not Path(transcript_file).is_file():
                logging.warning(f"No transcript: {transcript_file}")
                continue

            with open(transcript_file, "r") as f:
                text = f.read().replace("ˈ", "").replace("ˌ", "")

            recording = Recording.from_file(wav_file, recording_id=idx)

            segment = SupervisionSegment(
                id=idx,
                recording_id=idx,
                start=0.0,
                duration=recording.duration,
                channel=0,
                language=language,
                text=text,
            )

            recordings.append(recording)
            supervisions.append(segment)

        recording_set = RecordingSet.from_recordings(recordings).resample(16000)
        supervision_set = SupervisionSet.from_segments(supervisions)
        validate_recordings_and_supervisions(recording_set, supervision_set)

        if output_dir is not None:
            supervision_set.to_file(
                output_dir / f"commonvoice_supervisions_{split}.jsonl.gz"
            )
            recording_set.to_file(
                output_dir / f"commonvoice_recordings_{split}.jsonl.gz"
            )

        manifests[split] = {
            "recordings": recording_set,
            "supervisions": supervision_set,
        }

    return manifests
