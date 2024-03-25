from dataclasses import dataclass
from typing import Optional

import librosa
import torch
from librosa.filters import mel as librosa_mel_fn

import numpy as np
import soundfile as sf

mel_basis = {}
hann_window = {}


@dataclass
class AudioFeaturesParams:
    win_size: int = 1024
    hop_size: int = 240
    num_mel_bins: int = 80
    sampling_rate: int = 24000
    fmin: float = 50.0
    fmax: Optional[float] = 11025.0
    n_fft: int = 2048
    central_padding: bool = False


def load_and_preprocess_audio(audio_file: str, sr: int, trim=False):
    audio, _ = librosa.load(audio_file, sr=sr)

    if trim:
        audio, _ = librosa.effects.trim(audio, top_db=10)

    audio = torch.FloatTensor(audio).squeeze()
    audio /= torch.abs(audio).max()

    audio = audio.unsqueeze(0)
    return audio


def preprocess_audio(audio: np.array, trim=False):
    if trim:
        audio, _ = librosa.effects.trim(audio, top_db=10)

    audio = torch.FloatTensor(audio).squeeze()
    audio /= torch.abs(audio).max()

    audio = audio.unsqueeze(0)
    return audio


def dynamic_range_compression_torch(
        x: torch.Tensor, C: int = 1, clip_val: float = 1e-5
) -> torch.Tensor:
    return torch.log(torch.clamp(x, min=clip_val) * C)


def spectral_normalize_torch(magnitudes: torch.Tensor) -> torch.Tensor:
    output = dynamic_range_compression_torch(magnitudes)
    return output


def mel_spectrogram(
        audio: torch.Tensor, audio_params: AudioFeaturesParams
) -> torch.Tensor:
    """Computes melspectrogram according to parameters

    Args:
        audio [wav_frames] or [1, wav_frames]: input wav
        audio_params : parameters of input audio and desired mel
        loss : min and max frequencies depend of that argument

    Returns:
        [num_bins, mel_frames] : melspectrogram
    """
    if audio.ndim < 2:
        audio = audio.unsqueeze(0)

    fmax = audio_params.fmax
    fmin = audio_params.fmin

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(
            sr=audio_params.sampling_rate,
            n_fft=audio_params.n_fft,
            n_mels=audio_params.num_mel_bins,
            fmin=fmin,
            fmax=fmax,
        )
        mel_basis[str(fmin) + "_" + str(fmax) + "_" + str(audio.device)] = (
            torch.from_numpy(mel).float().to(audio.device)
        )
        hann_window[str(audio.device)] = torch.hann_window(
            audio_params.win_size
        ).to(audio.device)

    audio = torch.nn.functional.pad(
        audio.unsqueeze(1),
        [
            int((audio_params.n_fft - audio_params.hop_size) / 2),
            int((audio_params.n_fft - audio_params.hop_size) / 2),
        ],
        mode="reflect",
    )
    audio = audio.squeeze(1)

    spec = torch.stft(
        audio,
        audio_params.n_fft,
        hop_length=audio_params.hop_size,
        win_length=audio_params.win_size,
        window=hann_window[str(audio.device)],
        center=audio_params.central_padding,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )

    spec = torch.view_as_real(spec)

    spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))

    spec = torch.matmul(
        mel_basis[str(fmin) + "_" + str(fmax) + "_" + str(audio.device)], spec
    )
    spec = spectral_normalize_torch(spec)

    return spec


def smooth_transition(start_smooth: int,
                      end_smooth: int,
                      smooth_start_value: float,
                      center_smooth: float,
                      smooth_end_value: float) -> tuple[np.ndarray, np.ndarray]:
    smooth_out = np.linspace(smooth_start_value, center_smooth, start_smooth)
    smooth_in = np.linspace(center_smooth, smooth_end_value, end_smooth)

    return smooth_out, smooth_in


def fade(piece_1: np.array,
         piece_2: np.array,
         fade_out_start: int,
         fade_in_end: int,
         fade_start_value: float,
         center_fade: float,
         fade_end_value: float,
         exp=1) -> tuple[np.ndarray, np.ndarray]:
    fade_out = np.linspace(fade_start_value, center_fade, fade_out_start) ** exp
    # print(len(fade_out))
    # print(len(piece_1[len(piece_1) - fade_out_start:]))
    piece_1[len(piece_1) - fade_out_start:] *= fade_out

    fade_in = np.linspace(center_fade, fade_end_value, fade_in_end) ** exp
    # print(len(fade_in))
    # print(len(piece_2[:fade_in_end]))
    piece_2[:fade_in_end] *= fade_in

    return piece_1, piece_2


def smooth_pitch(audio, pitch, *samples: int):
    k = len(audio) / len(pitch[0, 0])
    # print(samples)
    # print(len(pitch[0, 0, :]))
    pitch_coords = list(map(lambda x: int(x / k), samples))
    # print(pitch_coords)

    cuts = []
    p = pitch[0, 0, :]
    for ind, coord in enumerate(pitch_coords[:-1]):
        if 0 < ind < len(pitch_coords) - 1:
            duration = min(len(p[pitch_coords[ind - 1]:coord]), len(p[coord:pitch_coords[ind + 1]]), 2) // 2
        elif ind == 0 and len(pitch_coords) > 1:
            duration = min(len(p[:coord]), len(p[coord:pitch_coords[ind + 1]]), 2) // 2
        elif ind == len(pitch_coords) - 1 and len(pitch_coords) > 1:
            duration = min(len(p[samples[ind - 1]:coord]), len(p[coord:]), 2) // 2
        cuts.append((coord - duration, coord + duration))

    for start, end in cuts:
        avg = float(sum(pitch[0, 0, start:end]) / len(pitch[0, 0, start:end]))
        smooth_start = float(pitch[0, 0, start])
        smooth_end = float(pitch[0, 0, end])
        smooth_1, smooth_2 = smooth_transition((end - start) // 2 + (end - start) % 2, (end - start) // 2, smooth_start,
                                               avg, smooth_end)

        pitch[0, 0, start:(start + end) // 2 + (end - start) % 2] = torch.FloatTensor(smooth_1)
        pitch[0, 0, (start + end) // 2 + (end - start) % 2:end] = torch.FloatTensor(smooth_2)


def merge_audio(*files: str, trim=False, fade=True) -> tuple[np.ndarray, int]:
    audios = [list(librosa.load(file)) for file in files]

    if trim:
        for ind in range(len(audios)):
            audio, _ = audios[ind]
            audios[ind][0] = librosa.effects.trim(audio, top_db=10)[0]

    if fade:
        fade_duration = 1
        duration = fade_duration / 2

        for i in range(len(audios) - 1):
            audio_1, sr_1 = audios[i]
            audio_2, sr_2 = audios[i + 1]

            # fade_out_duration = int(len(audio_1) - sr_1 * duration)
            fade_out_duration = int(sr_1 * duration)
            fade_in_duration = int(sr_2 * duration)

            faded_audio_1, faded_audio_2 = fade(audio_1, audio_2, fade_out_duration, fade_in_duration, 1.0, 0.6,
                                                1.0)

            audios[i][0] = faded_audio_1
            audios[i + 1][0] = faded_audio_2

    res = np.concatenate([audio[0] for audio in audios])
    return res, int(sum([audio[1] for audio in audios]) / len(audios))


def save_sliced(read_path: str, write_path: str) -> None:
    audio, sr = librosa.load(read_path)

    # режем аудио
    seconds = 2
    center = len(audio) // 2
    audio_1 = audio[:center - sr // int(2 * (1 / seconds))]
    audio_2 = audio[center + sr // int(2 * (1 / seconds)):]

    fade_duration = 0.7
    duration = fade_duration / 2

    fade_out_start = int(len(audio_1) - sr * duration)
    fade_in_end = int(sr * duration)

    faded_audio_1, faded_audio_2 = fade(audio_1, audio_2, fade_out_start, fade_in_end, 1.0, 0.6, 1.0)

    res = np.concatenate([faded_audio_1, faded_audio_2])

    sf.write(write_path, res, int(sr))
