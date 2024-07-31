import os
import json
import random

import tgt
import natsort
import librosa
import torchaudio
import numpy as np
import pyworld as pw
from tqdm import tqdm
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler

import audio as Audio
from g2p_en import G2p


class Preprocessor:
    def __init__(self, preprocess_config, model_config, train_config):
        random.seed(train_config['seed'])
        self.preprocess_config_config = preprocess_config
        self.dataset = preprocess_config["dataset"]
        self.speakers = set()
        self.emotions = set()
        self.sub_dir = preprocess_config["path"]["sub_dir_name"]
        self.data_dir = preprocess_config["path"]["corpus_path"]
        self.in_dir = os.path.join(preprocess_config["path"]["raw_path"], self.sub_dir)
        self.out_dir = preprocess_config["path"]["preprocessed_path"]
        self.val_size = 128
        # self.val_size = preprocess_config["preprocessing"]["val_size"]
        self.val_dialog_ids = self.get_val_dialog_ids()
        self.metadata = self.load_metadata()
        self.sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
        self.hop_length = preprocess_config["preprocessing"]["stft"]["hop_length"]
        self.filter_length = preprocess_config["preprocessing"]["stft"]["filter_length"]
        self.trim_top_db = preprocess_config["preprocessing"]["audio"]["trim_top_db"]

        assert preprocess_config["preprocessing"]["pitch"]["feature"] in [
            "phoneme_level",
            "frame_level",
        ]
        assert preprocess_config["preprocessing"]["energy"]["feature"] in [
            "phoneme_level",
            "frame_level",
        ]
        self.pitch_phoneme_averaging = (
            preprocess_config["preprocessing"]["pitch"]["feature"] == "phoneme_level"
        )
        self.energy_phoneme_averaging = (
            preprocess_config["preprocessing"]["energy"]["feature"] == "phoneme_level"
        )

        self.pitch_normalization = preprocess_config["preprocessing"]["pitch"]["normalization"]
        self.energy_normalization = preprocess_config["preprocessing"]["energy"]["normalization"]

        self.STFT = Audio.stft.TacotronSTFT(
            preprocess_config["preprocessing"]["stft"]["filter_length"],
            preprocess_config["preprocessing"]["stft"]["hop_length"],
            preprocess_config["preprocessing"]["stft"]["win_length"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
            preprocess_config["preprocessing"]["audio"]["sampling_rate"],
            preprocess_config["preprocessing"]["mel"]["mel_fmin"],
            preprocess_config["preprocessing"]["mel"]["mel_fmax"],
        )
        self.val_dialog_ids_prior = self.get_val_dialog_ids_prior(os.path.join(self.out_dir, "val.txt"))

    def get_val_dialog_ids_prior(self, val_prior_path):
        val_dialog_ids_prior = set()
        if os.path.isfile(val_prior_path):
            print("Load pre-defined validation set...")
            with open(val_prior_path, "r", encoding="utf-8") as f:
                for m in f.readlines():
                    val_dialog_ids_prior.add(int(m.split("|")[0].split("_")[-1].strip("d")))
            return list(val_dialog_ids_prior)
        else:
            return None

    def get_val_dialog_ids(self):
        data_size = len(os.listdir(self.in_dir))
        val_dialog_ids = random.sample(range(data_size), k=self.val_size)
        # print("val_dialog_ids:", val_dialog_ids)
        return val_dialog_ids

    def load_metadata(self):
        with open(os.path.join(self.data_dir, "metadata.json")) as f:
            metadata = json.load(f)
        return metadata

    def build_from_path(self):
        os.makedirs((os.path.join(self.out_dir, "mel")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "pitch")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "energy")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "duration")), exist_ok=True)

        print("Processing Data ...")
        out = list()
        n_frames = 0
        pitch_scaler = StandardScaler()
        energy_scaler = StandardScaler()

        train_set = list()
        val_set = list()

        # Compute pitch, energy, attn_prior, and mel-spectrogram
        # speakers = {}
        for i, speaker in enumerate(tqdm(os.listdir(self.in_dir))):
            # speakers[speaker] = i
            for wav_name in tqdm(os.listdir(os.path.join(self.in_dir, speaker))):
                if ".wav" not in wav_name:
                    continue

                basename = wav_name.split(".")[0]

                ret = self.process_utterance(speaker, basename)
                if ret is None:
                    continue
                else:
                    info, pitch, energy, frame = ret

                if int(speaker) not in self.val_dialog_ids:
                    train_set.append(info)
                else:
                    val_set.append(info)

                # out.append(info)

                if len(pitch) > 0:
                    pitch_scaler.partial_fit(pitch.reshape((-1, 1)))
                if len(energy) > 0:
                    energy_scaler.partial_fit(energy.reshape((-1, 1)))

                n_frames += frame

        print("Computing statistic quantities ...")
        # Perform normalization if necessary
        if self.pitch_normalization:
            pitch_mean = pitch_scaler.mean_[0]
            pitch_std = pitch_scaler.scale_[0]
        else:
            # A numerical trick to avoid normalization...
            pitch_mean = 0
            pitch_std = 1
        if self.energy_normalization:
            energy_mean = energy_scaler.mean_[0]
            energy_std = energy_scaler.scale_[0]
        else:
            energy_mean = 0
            energy_std = 1

        pitch_min, pitch_max = self.normalize(
            os.path.join(self.out_dir, "pitch"), pitch_mean, pitch_std
        )
        energy_min, energy_max = self.normalize(
            os.path.join(self.out_dir, "energy"), energy_mean, energy_std
        )

        if len(self.speakers) != 0:
            speaker_dict = dict()
            for i, speaker in enumerate(list(self.speakers)):
                speaker_dict[speaker] = int(speaker)
            with open(os.path.join(self.out_dir, "speakers.json"), "w") as f:
                f.write(json.dumps(speaker_dict))

        # if len(self.emotions) != 0:
        #     emotion_dict = dict()
        #     for i, emotion in enumerate(list(self.emotions)):
        #         emotion_dict[emotion] = i
        #     with open(os.path.join(self.out_dir, "emotions.json"), "w") as f:
        #         f.write(json.dumps(emotion_dict))

        with open(os.path.join(self.out_dir, "stats.json"), "w") as f:
            stats = {
                "pitch": [
                    float(pitch_min),
                    float(pitch_max),
                    float(pitch_mean),
                    float(pitch_std),
                ],
                "energy": [
                    float(energy_min),
                    float(energy_max),
                    float(energy_mean),
                    float(energy_std),
                ],
            }
            f.write(json.dumps(stats))

        print(
            "Total time: {} hours".format(
                n_frames * self.hop_length / self.sampling_rate / 3600
            )
        )

        random.seed(777)
        random.shuffle(train_set)
        train_set = [r for r in train_set if r is not None]
        val_set = [r for r in val_set if r is not None]

        # Sort validation set by dialog
        train_set = sorted(train_set, key=lambda x: (int(x.split("|")[0].split("_")[-1].lstrip("d")), int(x.split("|")[0].split("_")[0])))
        val_set = sorted(val_set, key=lambda x: (int(x.split("|")[0].split("_")[-1].lstrip("d")), int(x.split("|")[0].split("_")[0])))

        # Write metadata
        with open(os.path.join(self.out_dir, "train.txt"), "w", encoding="utf-8") as f:
            for m in train_set:
                f.write(m + "\n")
        with open(os.path.join(self.out_dir, "val.txt"), "w", encoding="utf-8") as f:
            for m in val_set:
                f.write(m + "\n")

        return out

    def load_audio(self, wav_path):
        wav_raw, _ = librosa.load(wav_path, self.sampling_rate)
        _, index = librosa.effects.trim(wav_raw, top_db=self.trim_top_db, frame_length=self.filter_length, hop_length=self.hop_length)
        wav = wav_raw[index[0]:index[1]]
        duration = (index[1] - index[0]) / self.hop_length
        return wav_raw.astype(np.float32), wav.astype(np.float32), int(duration)

    def process_utterance(self, speaker, basename):
        wav_path = os.path.join(self.in_dir, speaker, "{}.wav".format(basename))
        text_path = os.path.join(self.in_dir, speaker, "{}.lab".format(basename))
        tg_path = os.path.join(
            self.out_dir, "TextGrid", speaker, "{}.TextGrid".format(basename)
        )

        speaker = basename.split("_")[1]
        dialog_id = basename.split("_")[-1].lstrip("d")
        uttr_id = basename.split("_")[0]
        emotion = self.metadata[dialog_id][uttr_id]["emotion"]
        if emotion == "no emotion":
            emotion = "none"
        self.speakers.add(speaker)
        self.emotions.add(emotion)

        # Get alignments
        if os.path.exists(tg_path):
            textgrid = tgt.io.read_textgrid(tg_path)
            phone, duration, start, end = self.get_alignment(
                textgrid.get_tier_by_name("phones")
            )
            text = "{" + " ".join(phone) + "}"
            if start >= end:
                return None

            # Read and trim wav files
            wav, _ = librosa.load(wav_path)
            wav = wav[
                int(self.sampling_rate * start) : int(self.sampling_rate * end)
            ].astype(np.float32)

            # Read raw text
            with open(text_path, "r") as f:
                raw_text = f.readline().strip("\n")

            # Compute fundamental frequency
            pitch, t = pw.dio(
                wav.astype(np.float64),
                self.sampling_rate,
                frame_period=self.hop_length / self.sampling_rate * 1000,
            )
            pitch = pw.stonemask(wav.astype(np.float64), pitch, t, self.sampling_rate)

            if np.sum(pitch != 0) <= 1:
                return None

            # Compute mel-scale spectrogram and energy
            mel_spectrogram, energy = Audio.tools.get_mel_from_wav(wav, self.STFT)

            # size setting
            end_set = min(pitch.shape[0], mel_spectrogram.shape[1])
            pitch = pitch[: end_set]
            mel_spectrogram = mel_spectrogram[:, : end_set]
            energy = energy[: end_set]

            if self.pitch_phoneme_averaging:
                # perform linear interpolation
                nonzero_ids = np.where(pitch != 0)[0]
                interp_fn = interp1d(
                    nonzero_ids,
                    pitch[nonzero_ids],
                    fill_value=(pitch[nonzero_ids[0]], pitch[nonzero_ids[-1]]),
                    bounds_error=False,
                )
                pitch = interp_fn(np.arange(0, len(pitch)))

                # Phoneme-level average
                pos = 0
                for i, d in enumerate(duration):
                    if d > 0:
                        pitch[i] = np.mean(pitch[pos : pos + d])
                    else:
                        pitch[i] = 0
                    pos += d
                pitch = pitch[: len(duration)]

            if self.energy_phoneme_averaging:
                # Phoneme-level average
                pos = 0
                for i, d in enumerate(duration):
                    if d > 0:
                        energy[i] = np.mean(energy[pos : pos + d])
                    else:
                        energy[i] = 0
                    pos += d
                energy = energy[: len(duration)]

            # Save files
            dur_filename = "{}-duration-{}.npy".format(speaker, basename)
            np.save(os.path.join(self.out_dir, "duration", dur_filename), duration)

            pitch_filename = "{}-pitch-{}.npy".format(speaker, basename)
            np.save(os.path.join(self.out_dir, "pitch", pitch_filename), pitch)

            energy_filename = "{}-energy-{}.npy".format(speaker, basename)
            np.save(os.path.join(self.out_dir, "energy", energy_filename), energy)

            mel_filename = "{}-mel-{}.npy".format(speaker, basename)
            np.save(
                os.path.join(self.out_dir, "mel", mel_filename),
                mel_spectrogram.T,
            )

            return (
                "|".join([basename, speaker, text, raw_text]),
                self.remove_outlier(pitch),
                self.remove_outlier(energy),
                mel_spectrogram.shape[1],
            )
        else:
            return None

    def get_alignment(self, tier):
        sil_phones = ["sil", "sp", "spn"]

        phones = []
        durations = []
        start_time = 0
        end_time = 0
        end_idx = 0
        for t in tier._objects:
            s, e, p = t.start_time, t.end_time, t.text

            # Trim leading silences
            if phones == []:
                if p in sil_phones:
                    continue
                else:
                    start_time = s

            if p not in sil_phones:
                # For ordinary phones
                phones.append(p)
                end_time = e
                end_idx = len(phones)
            else:
                # For silent phones
                phones.append(p)

            durations.append(
                int(
                    np.round(e * self.sampling_rate / self.hop_length)
                    - np.round(s * self.sampling_rate / self.hop_length)
                )
            )

        # Trim tailing silences
        phones = phones[:end_idx]
        durations = durations[:end_idx]

        return phones, durations, start_time, end_time

    def remove_outlier(self, values):
        values = np.array(values)
        p25 = np.percentile(values, 25)
        p75 = np.percentile(values, 75)
        lower = p25 - 1.5 * (p75 - p25)
        upper = p75 + 1.5 * (p75 - p25)
        normal_indices = np.logical_and(values > lower, values < upper)

        return values[normal_indices]

    def normalize(self, in_dir, mean, std):
        max_value = np.finfo(np.float64).min
        min_value = np.finfo(np.float64).max
        for filename in os.listdir(in_dir):
            filename = os.path.join(in_dir, filename)
            values = (np.load(filename) - mean) / std
            np.save(filename, values)

            max_value = max(max_value, max(values))
            min_value = min(min_value, min(values))

        return min_value, max_value
