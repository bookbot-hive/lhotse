#!/usr/bin/env python3

# Copyright 2021 Xiaomi Corporation (Author: Mingshuang Luo)
# Apache 2.0

import csv
import glob
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
from tqdm import tqdm

from lhotse import validate_recordings_and_supervisions
from lhotse.audio import Recording, RecordingSet
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike


def prepare_bookbot(
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

    wav_files = glob.glob(str(corpus_dir) + "/**/*.wav")
    split2speaker = get_speakers(wav_files)

    splits = ["train", "test", "dev"]
    manifests = defaultdict(dict)

    for split in splits:
        wav_files_split = list(
            filter(
                lambda x: x.split("/")[-1].split("_")[0] in split2speaker[split],
                wav_files,
            )
        )

        logging.debug(f"{split} dataset manifest generation.")
        recordings = []
        supervisions = []

        for wav_file in tqdm(wav_files_split):
            items = str(wav_file).strip().split("/")
            idx = items[-1].strip(".wav")
            speaker = idx.split("_")[0]
            language = items[-2]

            transcript_file = Path(wav_file).with_suffix(".tsv")
            if not Path(wav_file).is_file():
                logging.warning(f"No such file: {wav_file}")
                continue
            if not Path(transcript_file).is_file():
                logging.warning(f"No transcript: {transcript_file}")
                continue

            with open(transcript_file, "r") as f:
                rows = csv.reader(f, delimiter="\t", quotechar='"')
                text = " ".join(row[2] for row in rows)

            recording = Recording.from_file(wav_file)

            segment = SupervisionSegment(
                id=idx,
                recording_id=idx,
                start=0.0,
                duration=recording.duration,
                channel=0,
                language=language,
                speaker=speaker,
                text=text,
            )

            recordings.append(recording)
            supervisions.append(segment)

        recording_set = RecordingSet.from_recordings(recordings)
        supervision_set = SupervisionSet.from_segments(supervisions)
        validate_recordings_and_supervisions(recording_set, supervision_set)

        if output_dir is not None:
            supervision_set.to_file(
                output_dir / f"bookbot_supervisions_{split}.jsonl.gz"
            )
            recording_set.to_file(output_dir / f"bookbot_recordings_{split}.jsonl.gz")

        manifests[split] = {
            "recordings": recording_set,
            "supervisions": supervision_set,
        }

    return manifests


def get_speakers(wav_files: List[str]) -> Dict[str, List[str]]:
    speakers = [path.split("/")[-1].split("_")[0] for path in wav_files]
    speaker2count = {s: c for s, c in zip(*np.unique(speakers, return_counts=True))}

    train_num = int(0.7 * len(wav_files))
    dev_num = int(0.9 * len(wav_files))

    train_speakers, test_speakers, dev_speakers = [], [], []
    total = 0

    for speaker, count in sorted(
        speaker2count.items(), key=lambda item: item[1], reverse=True
    ):
        if total < train_num and total < dev_num:
            train_speakers.append(speaker)
        elif total < dev_num:
            test_speakers.append(speaker)
        else:
            dev_speakers.append(speaker)
        total += count

    return {"train": train_speakers, "test": test_speakers, "dev": dev_speakers}
