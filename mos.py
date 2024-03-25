import torch
import torchaudio

import lightning_module
import torch.nn as nn

import numpy as np
import tqdm
import glob


class ChangeSampleRate(nn.Module):
    def __init__(self, input_rate: int, output_rate: int):
        super().__init__()
        self.output_rate = output_rate
        self.input_rate = input_rate

    def forward(self, wav: torch.tensor) -> torch.tensor:
        # Only accepts 1-channel waveform input
        wav = wav.view(wav.size(0), -1)
        new_length = wav.size(-1) * self.output_rate // self.input_rate
        indices = (torch.arange(new_length) * (self.input_rate / self.output_rate))
        round_down = wav[:, indices.long()]
        round_up = wav[:, (indices.long() + 1).clamp(max=wav.size(-1) - 1)]
        output = round_down * (1. - indices.fmod(1.)).unsqueeze(0) + round_up * indices.fmod(1.).unsqueeze(0)
        return output


def calc_mos(audio_path, model):
    wav, sr = torchaudio.load(audio_path)
    return calc_mos_raw(wav, sr, model)


def calc_mos_raw(wav, sr, model):
    osr = 16_000
    batch = wav.unsqueeze(0).repeat(10, 1, 1)
    csr = ChangeSampleRate(sr, osr)
    out_wavs = csr(wav)
    batch = {
        'wav': out_wavs,
        'domains': torch.tensor([0]),
        'judge_id': torch.tensor([288])
    }
    with torch.no_grad():
        output = model(batch)
    return output.mean(dim=1).squeeze().detach().numpy() * 2 + 3


def cals_mos_collection(audios: list[tuple[torch.Tensor, int]], model) -> float:
    return np.mean([calc_mos_raw(audio[0], audio[1], model) for audio in audios])


def calc_mos_dir(dir_path, model, mean=True):
    pred_mos = []
    for path in sorted(glob.glob(f"{dir_path}/*.wav")):
        pred_mos.append(calc_mos(path, model))
    if mean:
        return np.mean(pred_mos)
    else:
        return pred_mos

# buckets_mos = calc_mos_dir("data/buckets")
# shuffled_mos = calc_mos_dir("data/shuffled")
# converted_mos = calc_mos_dir("data/converted")
# method_1_mos = calc_mos_dir("data/method_1")
# method_2_mos = calc_mos_dir("data/method_2")
# method_3_mos = calc_mos_dir("data/method_3")
#
# print()
# print(f'MOS on buckets (ref): {buckets_mos}')
# print(f'MOS on shuffled: {shuffled_mos}')
# print(f'MOS on converted: {converted_mos}')
# print(f'MOS on method_1: {method_1_mos}')
# print(f'MOS on method_2: {method_2_mos}')
# print(f'MOS on method_3: {method_3_mos}')
