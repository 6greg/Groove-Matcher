
from pathlib import Path
import sys

import joblib
import librosa
import numpy as np
import pandas as pd
from scipy.stats import binned_statistic

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
from source.cca import prepare, pool

tensors_dir = project_root / "data" / "results" / "tensors"
drums_dir = project_root / "data" / "train" / "drums"
model_dir = project_root / "data" / "model"

arrangements_types = ["bass", "harm", "full", "full_no_bass"]
top_k = 100

# parametri griglia e audio
bars = 8
beats_per_bar = 4
grid_resolution = 16
griglia = bars * grid_resolution  # 128
sr = 44100
n_fft = 2048
hop_length = 512

# ESTRAZIONE FEATURE AUDIO

# feature audio 
def load_audio(path, bpm):
    """carica l'audio e lo taglia alle 8 battute in base al bpm"""
    duration = bars * beats_per_bar * (60.0 / bpm)
    y, _ = librosa.load(path, sr=sr, mono=True, duration=duration)
    return y

def stft_magnitude(y):
    """calcola la STFT e restituisce la matrice di magnitudo e le frequenze corrispondenti"""
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window="hann", center=True)
    stft = np.abs(stft)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    return stft, freqs

def spectral_flux_band(stft, freqs, low_hz, high_hz):
    """calcola il flusso spettrale per bande"""
    band_magnitude = stft[(freqs >= low_hz) & (freqs < high_hz)]
    positive_delta = np.maximum(np.diff(band_magnitude, axis=1), 0)
    return np.sqrt(np.sum(positive_delta * positive_delta, axis=0))

def spectral_flux(stft, freqs):
    spectral_flux_low = spectral_flux_band(stft, freqs, 40, 250)
    spectral_flux_mid = spectral_flux_band(stft, freqs, 250, 4000)
    spectral_flux_high = spectral_flux_band(stft, freqs, 4000, 20000)
    return spectral_flux_low, spectral_flux_mid, spectral_flux_high

def frame_times(n_frames, offset=0.0):
    """istante del centro di ogni frame; offset=0.5 per il flux e chroma_change, 0.0 per rms"""
    return (np.arange(n_frames) + offset) * hop_length / sr

def grid(values, times, bpm, stat="max"):
    """feature sulla griglia da 128 step (ogni frame va allo step più vicino)"""
    step_duration = (60.0 / bpm) * 4 / grid_resolution
    k = np.arange(griglia + 1)
    edges = (k - 0.5) * step_duration   # bordi a metà tra gli step
    out, _, _ = binned_statistic(times, values, statistic=stat, bins=edges)
    return np.nan_to_num(out, nan=0.0).astype(np.float32) # step vuoti -> 0

def extract_audio_features(path, bpm):
    """Estrae le feature da un file audio e le restituisce come tensore"""
    y = load_audio(path, bpm)
    stft, freqs = stft_magnitude(y)
    flux = frame_times(stft.shape[1] - 1, offset=0.5) # il flux è la differenza tra il frame i e il frame i+1

    features = []

    # spectral flux
    spectral_flux_low, spectral_flux_mid, spectral_flux_high = spectral_flux(stft, freqs)
    features.append(grid(spectral_flux_low, flux, bpm))
    features.append(grid(spectral_flux_mid, flux, bpm))
    features.append(grid(spectral_flux_high, flux, bpm))

    # rms
    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length, center=True).ravel()
    features.append(grid(rms, frame_times(rms.size), bpm, stat="mean"))

    # chroma
    chroma = librosa.feature.chroma_stft(S=stft**2, sr=sr, n_fft=n_fft, norm=2, tuning=0.0)
    chroma_change = np.linalg.norm(np.diff(chroma, axis=1), axis=0)
    features.append(grid(chroma_change, frame_times(chroma_change.size, offset=0.5), bpm))

    return np.stack(features)


# LIBRERIA:
def library():
    """restituisce la libreria di batterie e gli id corrispondenti"""
    drum_tensor = np.load(tensors_dir / "tensor_drums.npz")
    drum_ids = drum_tensor["ids"] # id delle batterie nella libreria
    drum_vectors = drum_tensor["Y"] # tensore (70, 5, 128) 

    midi_files = {}
    for midi_path in drums_dir.rglob("*.mid"):
        _, drum_id, _ = midi_path.stem.split("_") 
        midi_files[int(drum_id)] = midi_path
    
    return drum_vectors, drum_ids, midi_files

# MATCHING
def matching(arrangement_type, features, drum_vectors):
    """correlazione tra l'arrangiamento e ogni batteria della libreria"""
    path = model_dir / f"cca_{arrangement_type}.joblib"
    cca = joblib.load(path)["cca"]

    u = prepare(features.reshape(1, 5, griglia)) # arrangiamento: (1, 5, 128)
    v = prepare(drum_vectors)                    # batterie: (70, 5, 128)

    u, v = cca.transform(pool(u), pool(v)) # variabili canoniche
    u = u[:, 0] # arrangiamento: (128)
    v = v[:, 0].reshape(len(drum_vectors), griglia) # batterie: (70, 128)

    return np.corrcoef(u, v)[0, 1:]


# GROOVE MATCHER
def groove_matcher(audio_path, bpm, arrangement_type):
    if audio_path is None:
        raise ValueError("Load an audio file")
    if bpm is None:
        raise ValueError("Insert the BPM of the arrangement")
    if arrangement_type is None:
        raise ValueError("Choose the type of arrangement")
    if librosa.get_duration(path=audio_path) < bars * beats_per_bar * 60 / bpm:
        raise ValueError(f"The audio must be at least {bars} bars long")

    features = extract_audio_features(audio_path, bpm)
    drum_vectors, drum_ids, midi_files = library()
    scores = matching(arrangement_type, features, drum_vectors)

    best = np.argsort(-scores)[:top_k] # posizioni delle top_k batterie, dalla più correlata
    best_files = [midi_files[i] for i in drum_ids[best]]


    table = pd.DataFrame({
        "rank": np.arange(1, len(best) + 1),
        "drum": [f.name for f in best_files],
        "correlation": np.round(scores[best], 3),
    })
    return table, [str(f) for f in best_files]
