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
import random
import math
from itertools import product
from scipy.signal import resample_poly
from pathlib import Path
from typing import Dict, List, Tuple, Union
from sklearn.model_selection import GroupKFold
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
from osl_dynamics.data import Data
from osl_dynamics.models.hmm import Model
from osl_dynamics.models.hmm import Config
from osl_dynamics.data.processing import standardize
from osl_dynamics.data.processing import time_embed
from osl_dynamics.models import load
from osl_dynamics.utils import plotting
import pickle

import threading, time

#Avoid Run:AI’s idle detection
def keepalive():
    while True:
        print("[Keepalive] Job active", flush=True)
        time.sleep(300)  # Every 5 minutes

#threading.Thread(target=keepalive, daemon=True).start()

# =============== DATA PREPROCESSING ============== #

# ===== 1. SANITY CHECK: NaN and zero variance ====== #
# === Functions ===
def check_data_object(trials):
    """
    Check the trials (npy arrays) contained in an osl_dynamics.data.Data object.
    Detects NaN or Inf values and prints per-scout variance (min/median/max).
    """
    print(f"Number of trials: {len(trials)}")

    # Ensure it's a list of 2D arrays (time x scouts)
    if isinstance(trials, np.ndarray) and trials.dtype == object:
        trials = list(trials)

    for i, trial in enumerate(trials):
        assert trial.ndim == 2, f"trial{i:02d} not 2D"
        assert np.all(np.isfinite(trial)), f"trial{i:02d} has NaN/Inf values"

        v = trial.var(axis=0)
        print(f"trial{i+1:02d}: shape={trial.shape}, "
              f"min/median/max var = {v.min():.2e}/{np.median(v):.2e}/{v.max():.2e}")      
# === end Functions ===

#with open('/mnt/uphummel/scratch/nbenlamr/osl_results/data_dict_left_lesion_pca_56subj_v2.pkl', 'rb') as f:
#   data_dict_pca = pickle.load(f)
with open("/mnt/uphummel/scratch/nbenlamr/osl_results/data_dict_all_pca.pkl", "rb") as f:
    data_dict_pca = pickle.load(f)


# Sanity check after PCA
#Optional - Check again the dictionary data to see if pca was successful
for sid, trials in data_dict_pca.items():
    print(f"\nChecking first trial for subject {sid} - AFTER PCA")
    check_data_object(trials[:1])

# infer the PCA dimensionality from the first trial
example_sid = next(iter(data_dict_pca))
example_trial = data_dict_pca[example_sid][0]
k = example_trial.shape[1]

print(f"PCA dimensionality (n_channels for HMM): {k}")

# #Save for future use
# with open('/mnt/uphummel/scratch/nbenlamr/osl_results/data_dict_pca_nodownsampl_56subj.pkl', 'wb') as f:
#     pickle.dump(data_dict_pca, f) 

# =============== TDE-HMM MODEL ============== #

# ===== 1. Chunck-based early stopping model FITTING ====== #
# === Functions ===
def fit_with_val_es(cfg: Config, data_train: Data, data_val: Data,
                    total_epochs=200, chunk=10, patience=3):
    """Train in chunks; stop when val free energy stops improving."""
    model = Model(cfg)
    best = (np.inf, None)  # (val_energy, weights)
    wait = 0
    epochs_done = 0

    while epochs_done < total_epochs:
        # advance training by 'chunk' epochs (continues from current weights)
        cfg.n_epochs = chunk
        model.fit(data_train)

        # --- suppress TensorFlow C++ spam just for free_energy call ---
        stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')  # Suppress TF C++ runtime noise
        
        val_energy = model.free_energy(data_val)

        stderr = sys.stderr
        # --------------------------------------------------------------

        print(f"Free energy for this chunk: {val_energy}")

        if val_energy < best[0] - 1e-3:  # small delta to avoid noise flaps
            best = (val_energy, model.get_weights())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

        epochs_done += chunk

    # restore best weights
    if best[1] is not None:
        model.set_weights(best[1])

    return model, best[0]
# === end Functions ===

# ===== 2. Build .Config osl object ====== #
# === Functions ===
def build_config(n_states, lr, n_init_epochs=None):
    return Config(
        model_name="HMM",
        n_states=n_states,
        n_channels=int(k),
        sequence_length=600,

        # Emissions
        learn_means=False,
        learn_covariances=True,
        initial_means=None,
        initial_covariances=None,
        diagonal_covariances=False,
        covariances_epsilon=1e-6,

        # Transition model
        initial_trans_prob=None,
        learn_trans_prob=True,
        trans_prob_update_delay=5,
        trans_prob_update_forget=0.7,
        initial_state_probs=None,
        learn_initial_state_probs=True,
        baum_welch_implementation="log",

        # Training
        batch_size=24,
        learning_rate=lr,
        lr_decay=0.05,
        n_epochs=10,       # per chunk
        loss_calc="mean",
        optimizer="adam",  # default
        gradient_clip=None,

        # Misc
        multiple_dynamics=False,
        multi_gpu=False
    )
# === end Functions ===

# ===== 3. Hyperparametrization with grid search ====== #
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
        return Data(
        seqs,
        store_dir="/mnt/uphummel/scratch/nbenlamr/osl_dynamics/tmp"
    ), np.asarray(groups, int), True
    else:
        # legacy single-subject: group id = 0 for all sequences
        seqs = list(data)
        groups = np.zeros(len(seqs), dtype=int)
        return Data(
        seqs,
        store_dir="/mnt/uphummel/scratch/nbenlamr/osl_dynamics/tmp"
    ), np.asarray(groups, int), False

def subset_Data(data_all: Data, seq_idx: np.ndarray) -> Data:
    # Rebuild a Data object from a subset of sequences
    seqs = [data_all.arrays[i] for i in seq_idx.tolist()]
    return Data(
        seqs,
        store_dir="/mnt/uphummel/scratch/nbenlamr/osl_dynamics/tmp"
    )

def grid_search_tdehmm(
    data_input,               
    K: int = 5,               
    total_epochs: int = 200,
    chunk: int = 10,
    patience: int = 3,
):
    """
    Single CV (no nesting). For each hyperparameter combination:
      - Build GroupKFold folds by subject
      - Train with fit_with_val_es()
      - Score by mean validation loss/free energy across folds
    Returns:
      best_model, results_sorted  where results_sorted = [(config_dict, mean_val), ...] (ascending)
    """

    # 0) Pack sequences and group ids (subjects)
    data_all, groups, is_multi = pack_sequences_and_groups(data_input)
    print(f"Total sequences: {len(data_all.arrays)}")
    assert len(data_all.arrays) > 0, "Data object is empty!"
    n_seq = len(data_all.arrays)

    # 1) Define search space (grid)
    space = {
        "n_states": [8], #[6, 8, 10]
        "lr": [3e-4,], #[3e-4, 5e-4,]
        "n_init_epochs": [1],
    }

    # Build full grid of all combinations
    keys = list(space.keys())
    grid = [dict(zip(keys, vals)) for vals in product(*space.values())]
    print(f"🔍 Grid search: {len(grid)} total configurations")

    # 2) Build folds
    folds = []
    if is_multi and len(np.unique(groups)) >= max(2, K):
        gkf = GroupKFold(n_splits=K)
        seq_idx = np.arange(n_seq)
        for tr_idx, va_idx in gkf.split(seq_idx, groups=groups):
            folds.append((tr_idx, va_idx))
    else:
        folds = [("temporal_fallback", "temporal_fallback")]

    # 3) Grid search loop
    results = []
    best_score = np.inf
    best_model = None

    for i, params in enumerate(grid, 1):
        print(f"=== Grid Search [{i}/{len(grid)}] {params} === \n")

        cfg = build_config(
            n_states=params["n_states"],
            lr=params["lr"],
            n_init_epochs=params["n_init_epochs"],
        )

        val_energys = []

        if isinstance(folds[0][0], np.ndarray):
            print("----- Verifying GroupKFold splits -----")
            for fold_id, (tr_idx, va_idx) in enumerate(folds):
                train_sids = set(groups[tr_idx])
                val_sids   = set(groups[va_idx])
                overlap    = train_sids.intersection(val_sids)

                print(f"Fold {fold_id}: Train subjects = {len(train_sids)}, Val subjects = {len(val_sids)}, Overlap = {overlap}")
                assert len(overlap) == 0, f"Subject leakage detected in fold {fold_id}!"
            # --- Grouped K-fold ---
            for k, (tr_idx, va_idx) in enumerate(folds, 1):

                print(f"---- Training fold {k}/{len(folds)} ----")

                data_train = subset_Data(data_all, tr_idx)
                data_val   = subset_Data(data_all, va_idx)

                model, val_energy = fit_with_val_es(
                    cfg=cfg,
                    data_train=data_train,
                    data_val=data_val,
                    total_epochs=total_epochs,
                    chunk=chunk,
                    patience=patience
                )
                val_energys.append(float(val_energy))
        else:
            # --- Single-subject fallback ---
            tr_seqs, va_seqs = [], []
            for X in data_all.arrays:
                T = X.shape[0]
                cut = int(np.floor(0.8 * T))
                tr_seqs.append(X[:cut])
                va_seqs.append(X[cut:])
            data_train = Data(tr_seqs)
            data_val   = Data(va_seqs)

            model, val_energy = fit_with_val_es(
                cfg=cfg,
                data_train=data_train,
                data_val=data_val,
                total_epochs=total_epochs,
                chunk=chunk,
                patience=patience
            )
            val_energys.append(float(val_energy))

        mean_val = float(np.mean(val_energys))
        results.append((params, mean_val))

        if mean_val < best_score:
            best_score = mean_val
            #best_model = model

        print(f"\n\n [{i}/{len(grid)}] {params} -> mean_val={mean_val:.6f} === \n")
        print("=============================================\n")

    results_sorted = sorted(results, key=lambda x: x[1])

    best_params = results_sorted[0][0]  # dict like {"n_states": ..., "lr": ..., ...}
    final_cfg = build_config(
        n_states=best_params["n_states"],
        lr=best_params["lr"],
        n_init_epochs=best_params.get("n_init_epochs", None),
    )

    print("\n=== Training final model on ALL data ===")

    # Do a simple full training without validation or early stopping
    final_cfg.n_epochs = 100   # Train for all epochs at once
    final_model = Model(final_cfg)
    final_model.fit(data_all)

    return final_model, results_sorted
# === end Functions ===runai exec -it osl-hmm-v001-10 -- bash


# ===== 0. Filter short trials ====== #
# === Functions ===
def filter_short_trials(data_dict: Dict[int, List[np.ndarray]], min_len: int = 200) -> Dict[int, List[np.ndarray]]:
    """
    Remove trials shorter than `min_len` samples.
    Returns a cleaned copy of the data_dict (subject_id -> list of np.ndarrays).

    Parameters
    ----------
    data_dict : dict
        {subject_id: [trial1, trial2, ...]}
    min_len : int
        Minimum number of timepoints required to keep a trial.

    Returns
    -------
    cleaned_dict : dict
        Dictionary with only trials having T >= min_len.
    """
    cleaned = {}
    total_dropped = 0

    print(f"\n🧹 Checking and filtering short trials (min_len = {min_len})")

    for sid, trials in data_dict.items():
        long_trials = [X for X in trials if X.shape[0] >= min_len]
        dropped = len(trials) - len(long_trials)
        total_dropped += dropped

        if dropped > 0:
            print(f"  Subject {sid}: dropped {dropped}/{len(trials)} short trials")

        if len(long_trials) > 0:
            cleaned[sid] = long_trials
        else:
            print(f"  ⚠️ Subject {sid} removed completely (no valid trials)")

    # Print summary statistics
    all_lens = [X.shape[0] for trials in cleaned.values() for X in trials]
    if all_lens:
        print(f"\n✅ After filtering: {len(cleaned)} subjects, {sum(len(t) for t in cleaned.values())} trials total")
        print(f"   Sequence length range: min={min(all_lens)}, median={np.median(all_lens):.0f}, max={max(all_lens)}")
    else:
        print("\n❌ All trials were removed! Check min_len or preprocessing steps.")
    
    return cleaned
# === end Functions ===

data_dict_pca = filter_short_trials(data_dict_pca, min_len=200)

# ===== 4. Run the model ====== #
final_model, results_sorted = grid_search_tdehmm(
    data_dict_pca,
    K=3, #3
    total_epochs=100, #100
    chunk=5, #5
    patience=3,
)

# Save the model
#final_model.save("/mnt/uphummel/scratch/nbenlamr/osl_results/Results_HMM_bestconfig_left_lesion")
#analysis with all the groups
final_model.save("/mnt/uphummel/scratch/nbenlamr/osl_results/Results_HMM_bestconfig_all")
#to save the grid search results
with open("/mnt/uphummel/scratch/nbenlamr/osl_results/grid_search_results_all.pkl", "wb") as f:
    pickle.dump(results_sorted, f)

# Clean exit
import gc, sys, tensorflow as tf
tf.keras.backend.clear_session()
gc.collect()
print("\n✅ Training complete. Exiting cleanly.")
sys.exit(0) 