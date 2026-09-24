import os, sys

# Suppress TensorFlow C++ backend logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=all, 1=INFO, 2=WARNING, 3=ERROR
# Now import TensorFlow
import tensorflow as tf
# Suppress high-level TensorFlow Python logs
tf.get_logger().setLevel('ERROR')
# Optional: suppress other noisy loggers        
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import time
import re
import pandas as pd
import json, glob
import numpy as np
import nibabel as nib
import random
import math
import pickle
from itertools import product
from scipy.signal import resample_poly
from pathlib import Path
from typing import Dict, List, Tuple, Union
from sklearn.model_selection import GroupKFold
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from osl_dynamics.data import Data
from osl_dynamics.models.hmm import Model
from osl_dynamics.models.hmm import Config
from osl_dynamics.data.processing import standardize
from osl_dynamics.data.processing import time_embed
from osl_dynamics.models import load
from osl_dynamics.utils import plotting
from osl_dynamics.analysis import spectral
from osl_dynamics.analysis import power
from osl_dynamics.inference import modes

#Plotting options:
import matplotlib.colors as mcolors

color_list=["#010213", "#2b3cd8", "#22e9f0", "#74dd2e", "#F0E43F", "#e79c39", "#e7514c", "#ef3df5"] #"#9b0bbe"
cmap_custom = mcolors.ListedColormap(color_list, name="my_custom")
plt.register_cmap(cmap=cmap_custom)

# ===== 1. Load the datsaset ====== #
# === Functions ===
def pack_sequences_and_groups(
    data: Union[List[np.ndarray], Dict[int, List[np.ndarray]]]
) -> Tuple[Data, np.ndarray, bool]:
    """
    Accepts:
      - List[np.ndarray]  -> single-subject (legacy)
      - Dict[subject_id, List[np.ndarray]] -> multi-subject
    Returns:
      - Data(list of sequences)
      - groups: np.ndarray[int], subject id per sequence
      - is_multi: bool
    """
    if isinstance(data, dict):
        seqs, groups = [], []
        for sid, trials in data.items():
            for X in trials:
                seqs.append(X)
                groups.append(int(sid))
        return Data(seqs), np.asarray(groups, int), True
    else:
        # legacy single-subject: group id = 0 for all sequences
        seqs = list(data)
        groups = np.zeros(len(seqs), dtype=int)
        return Data(seqs), groups, False

def subset_Data(data_all: Data, seq_idx: np.ndarray) -> Data:
    # Rebuild a Data object from a subset of sequences
    seqs = [data_all.arrays[i] for i in seq_idx.tolist()]
    return Data(seqs)
# === end Functions === #

#load data dict pca object for future use
with open('/mnt/uphummel/scratch/nbenlamr/osl_results/data_dict_all_stand.pkl', 'rb') as file:
    data_dict_raw_loaded = pickle.load(file)
#load data dict pca object for future use
with open('/mnt/uphummel/scratch/nbenlamr/osl_results/data_dict_all_pca.pkl', 'rb') as file:
    data_dict_pca_loaded = pickle.load(file)

data_all_pca, group_all_pca, is_multi_pca = pack_sequences_and_groups(data_dict_pca_loaded)

# ===== 2. Load the trained HMM model ====== #

model_loaded = load("/mnt/uphummel/scratch/nbenlamr/osl_results/Results_HMM_bestconfig_all")
n_states = model_loaded.config.n_states
sequence_length = model_loaded.config.sequence_length
#Check hyperparameters of the best model
print(f"Number of hidden states of the best hyperparameter configuration: {model_loaded.config.n_states}.\n")
print(f"Learning rate of the best hyperparameter configuration: {model_loaded.config.learning_rate}.\n")

# ===== 3.1 Feature extraction: time-domain features ====== 

#Extract saved free energy
#fe = model_loaded.free_energy(data_all_pca)
#Extract state time courses (alphas)
alphas = model_loaded.get_alpha(data_all_pca) #(n-trials, n-timepoints, n-states)
alpha_avgtrials_all = np.concatenate(alphas, axis=0)
# Hard classify the state probabilities
stc = modes.argmax_time_courses(alphas)
# Plot the state time course for the first subject 
stc_avgtrials_all = np.concatenate(stc, axis=0)

#Compute fractional occupancy (FO) 
fo = modes.fractional_occupancies(stc)
gfo = np.mean(fo, axis=0)
print(fo.shape) # (n-trials, n-states)
print(gfo.shape)

# ===== 3.2 Feature extraction: spectral features ====== 

#Trimming of time-series
all_trials = [trial for trials in data_dict_raw_loaded.values() for trial in trials]
data_raw = Data(all_trials)
x = data_raw.trim_time_series(n_embeddings=15, sequence_length=sequence_length)

#PSD extraction
f, psd, coh, w = spectral.multitaper_spectra(
    data=x,
    alpha=alphas,
    sampling_frequency=250,
    time_half_bandwidth=4,
    n_tapers=7,
    frequency_range=[0, 80],
    #standardize=True,
    return_weights=True,
    n_jobs=1
)
print(f.shape) #(n-freqs,)
print(psd.shape) #(n-trials, n-states, n-parcels, n-freqs)
print(coh.shape) #(n-trials, n-states, n-parcels, n-parcels, n-freqs)
print(w.shape) # (n-trials,)

# Sanity check on coherence: ensure no NaN or Inf
coh_clean = np.nan_to_num(coh, nan=0.0, posinf=0.0, neginf=0.0)
coh_clean = np.clip(coh_clean, 0, 1)
print(coh_clean.shape)

#Save features
np.save("/mnt/uphummel/scratch/nbenlamr/osl_results/spectra/AcuteStroke/f_v003_downsampl_0Hz.npy", f)
np.save("/mnt/uphummel/scratch/nbenlamr/osl_results/spectra/AcuteStroke/psd_v003_downsampl_0Hz.npy", psd)
np.save("/mnt/uphummel/scratch/nbenlamr/osl_results/spectra/AcuteStroke/coh_v003_downsampl.npy", coh)
np.save("/mnt/uphummel/scratch/nbenlamr/osl_results/spectra/AcuteStroke/w_v003_downsampl.npy", w)
np.save("/mnt/uphummel/scratch/nbenlamr/osl_results/spectra/AcuteStroke/coh_clean_v003_downsampl.npy", coh_clean)
# Clean exit
import gc, sys, tensorflow as tf
tf.keras.backend.clear_session()
gc.collect()
print("\n✅ Feature extraction complete. Exiting cleanly.")
sys.exit(0)