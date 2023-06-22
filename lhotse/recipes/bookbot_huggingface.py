#!/usr/bin/env python3

# Copyright 2021 Xiaomi Corporation (Author: Mingshuang Luo)
# Apache 2.0

import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Union

import soundfile as sf
from datasets import load_dataset
from tqdm.auto import tqdm
from p_tqdm import p_map

from lhotse import validate_recordings_and_supervisions
from lhotse.audio import Recording, RecordingSet
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike


def download_bookbot_huggingface(
    dataset_name: str,
    target_dir: Pathlike = ".",
    text_column_name: str = "phonemes_ipa",
    word_delimiter_token: str = " | ",
) -> Path:
    """
    Download and unzip any Bookbot phoneme dataset from HuggingFace.
    :param dataset_name: str, HuggingFace Hub dataset name.
    :param target_dir: Pathlike, the path of the dir to store the dataset.
    :return: the path to downloaded and extracted directory with data.
    """

    def save_audio_file(datum):
        audio_path, audio_array, sr = datum["audio"].values()
        audio_path = Path(audio_path)
        text = word_delimiter_token.join(datum[text_column_name])
        language = datum["language"]

        lang_dir = split_dir / language
        lang_dir.mkdir(parents=True, exist_ok=True)

        sf.write(
            str(lang_dir / audio_path.with_suffix(".wav")),
            audio_array,
            samplerate=sr,
            format="wav",
        )
        with open(str(lang_dir / audio_path.with_suffix(".txt")), "w") as f:
            f.write(text)

    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(dataset_name)
    corpus_name = dataset_name.split("/")[-1]
    splits = dataset.keys()

    corpus_dir = target_dir / corpus_name

    for split in splits:
        split_dir = corpus_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

        p_map(save_audio_file, dataset[split])

    return corpus_dir


def prepare_bookbot_huggingface(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    normalize_words: bool = False,
    normalize_phonemes: bool = False,
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

    manifests = defaultdict(dict)

    for split in corpus_dir.iterdir():
        if split.is_dir():
            wav_files = split.rglob("*.wav")
            logging.debug(f"{split} dataset manifest generation.")
            recordings = []
            supervisions = []

            for wav_file in tqdm(wav_files):
                items = str(wav_file).strip().split("/")
                idx = items[-1].strip(".wav")
                speaker = idx.split("_")[0]
                language = items[-2]

                transcript_file = wav_file.with_suffix(".txt")
                if not wav_file.is_file():
                    logging.warning(f"No such file: {wav_file}")
                    continue
                if not transcript_file.is_file():
                    logging.warning(f"No transcript: {transcript_file}")
                    continue

                with open(transcript_file, "r") as f:
                    text = f.read()
                    if normalize_phonemes:
                        diacritics = ["ː", "ˑ", "̆", "̯", "͡", "‿", "͜", "̩", "ˈ", "ˌ"]
                        for d in diacritics:
                            text = text.replace(d, "")

                    if normalize_words:
                        text = re.sub('[\,\?\.\!\-\;\:"\“\%\‘\”\�]', "", text).lower()

                recording = Recording.from_file(wav_file, recording_id=idx)

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

            recording_set = RecordingSet.from_recordings(recordings).resample(16000)
            supervision_set = SupervisionSet.from_segments(supervisions)
            validate_recordings_and_supervisions(recording_set, supervision_set)

            if output_dir is not None:
                supervision_set.to_file(
                    output_dir / f"{corpus_dir.stem}_supervisions_{split.stem}.jsonl.gz"
                )
                recording_set.to_file(
                    output_dir / f"{corpus_dir.stem}_recordings_{split.stem}.jsonl.gz"
                )

            manifests[split.stem] = {
                "recordings": recording_set,
                "supervisions": supervision_set,
            }

    return manifests
