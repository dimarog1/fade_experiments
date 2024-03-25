from audio_utils import AudioFeaturesParams, mel_spectrogram, load_and_preprocess_audio, fade, preprocess_audio, \
    smooth_pitch
from f0_utils import get_lf0_from_wav
import torch
import librosa
import soundfile as sf
import json
import os
import gc
import tqdm

gc.collect()
torch.cuda.empty_cache()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = "cuda"
model_jit_path = "model.pt"

params = AudioFeaturesParams()


def method_1(audio, *samples):
    for ind, sample in enumerate(samples):
        if 0 < ind < len(samples) - 1:
            fade_duration = min(len(audio[samples[ind - 1]:sample]), len(audio[sample:samples[ind + 1]])) // 2
        elif ind == 0 and len(samples) > 1:
            fade_duration = min(len(audio[:sample]), len(audio[sample:samples[ind + 1]])) // 2
        elif ind == len(samples) - 1 and len(samples) > 1:
            fade_duration = min(len(audio[samples[ind - 1]:sample]), len(audio[sample:])) // 2
        duration = fade_duration // 2
        fade(audio[:sample], audio[sample:], duration, duration, 1, 0.1, 1)
    return audio


def method_2(audio, sr, pitch, mel_ref, *samples):
    method_1(audio, sr, *samples)
    return convert(audio, pitch, mel_ref)


def method_3(audio, pitch, mel_ref, *samples):
    smooth_pitch(audio, pitch, *samples)
    return convert(audio, pitch, mel_ref)


def convert(audio, pitch, mel_ref):
    wav_source = preprocess_audio(audio).to(device)

    with torch.no_grad():
        traced = torch.jit.load(model_jit_path).eval()
        converted = traced(wav_source, mel_ref, pitch)
    return converted


# Чтение данных аудио
raw_data = open('data/log.json', 'r').read()
data = json.loads(raw_data)

# Первый метод
for number, audio_data in enumerate(tqdm.tqdm(data)):
    audio, sr = librosa.load(audio_data['myPath'], sr=16000)

    wav_ref = load_and_preprocess_audio(audio_data['origPath'], 24000, trim=True)
    mel_ref = mel_spectrogram(wav_ref, params).to(device)

    pitch = get_lf0_from_wav(audio_data['myPath']).to(device).float()

    samples = audio_data['splices']
    try:
        smoothed = convert(audio, pitch, mel_ref)
    except Exception as e:
        print(audio_data['myPath'])
        continue
    path = f'data/converted/converted{number}.wav'
    sf.write(path, smoothed.cpu().squeeze().detach().numpy(), 24000)

# Второй метод
for number, audio_data in enumerate(tqdm.tqdm(data)):
    audio, sr = librosa.load(audio_data['myPath'], sr=16000)

    wav_ref = load_and_preprocess_audio(audio_data['origPath'], 24000, trim=True)
    mel_ref = mel_spectrogram(wav_ref, params).to(device)

    pitch = get_lf0_from_wav(audio_data['myPath']).to(device).float()

    samples = audio_data['splices']
    try:
        smoothed = method_2(audio, sr, pitch, mel_ref, *samples)
    except Exception as e:
        print(audio_data['myPath'])
        print(e)
        continue
    path = f'data/method_2/smoothed{number}.wav'
    sf.write(path, smoothed.cpu().squeeze().detach().numpy(), 24000)

# Третий метод
for number, audio_data in enumerate(tqdm.tqdm(data)):
    audio, sr = librosa.load(audio_data['myPath'], sr=16000)

    wav_ref = load_and_preprocess_audio(audio_data['origPath'], 24000, trim=True)
    mel_ref = mel_spectrogram(wav_ref, params).to(device)

    pitch = get_lf0_from_wav(audio_data['myPath']).to(device).float()

    samples = audio_data['splices']
    try:
        smoothed = method_3(audio, pitch, mel_ref, *samples)
    except Exception as e:
        print(e)
        continue
    path = f'data/method_3/smoothed{number}.wav'
    sf.write(path, smoothed.cpu().squeeze().detach().numpy(), 24000)
