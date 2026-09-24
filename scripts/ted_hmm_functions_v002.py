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
from osl_dynamics.data import Data
from osl_dynamics.models.hmm import Model
from osl_dynamics.models.hmm import Config
from osl_dynamics.data.processing import standardize
from osl_dynamics.data.processing import time_embed
from osl_dynamics.models import load
from osl_dynamics.utils import plotting
import pickle

import threading, time

# =============== DATA LOADING ============== #

# === Functions ===
def load_trial_csv(path: str) -> Tuple[np.ndarray, List[str]]:
    """
    Load one trial CSV into (data, scout_names).

    Parameters
    ----------
    path : str
        Path to the CSV file for one trial.

    Returns
    -------
    X : np.ndarray , Shape (T, C)
        T = timepoints \n
        C = scouts.
    scout_names : List[str]
        List of scout names; scouts are defined as the ROI in the DKT parcellation atlas used in the source reconstruction step.
    
    """
    try:
        # Faster engine when available; falls back gracefully.
        if "pyarrow" in pd.__dict__:
            df = pd.read_csv(path, index_col=0, engine="pyarrow")
        else:
            df = pd.read_csv(path, index_col=0)
    except Exception as e:
        raise RuntimeError(f"Failed to read {path}: {e}")

    if df.empty:
        raise ValueError(f"CSV {path} is empty!")

    X = df.to_numpy(dtype=np.float32).T  # (T, C)
    scouts = list(map(str, df.index.tolist()))
    return X, scouts


def _is_newer(a: Path, b: Path) -> bool:
    """
    Return True if file a exists and is newer than file b (or b missing). Used to check if cached .npy is up-to-date with source .csv.

    Parameters
    ----------
    a : str or Path
        Path to the .npy file (cached).
    b : str or Path
        Path to the .csv file (source).

    Returns
    -------
    bool
        True if file a exists and is newer than file b (or b missing).

    """
    return a.exists() and (not b.exists() or a.stat().st_mtime >= b.stat().st_mtime)


def build_multisubject_dict_from_tree(
    base_dir: str,
    sub_glob: str = "sub-*", *,
    force: bool = False
) -> Dict[int, List[np.ndarray]]:
    """
    Load CSVs for each subject -> save as .npy -> return dict[subject_id] = [trials].

    Parameters
    ----------
    base_dir : str
        Base directory containing subject subdirectories.
    sub_glob : str
        Glob pattern to match subject directories (default "sub-*").
    force : bool
        If True, ignore cache and reprocess all CSVs (default False).
    
    Returns
    -------
    data_dict : Dict[int, List[np.ndarray]]
        Dictionary mapping subject_id to list of trial arrays (each shape (T, C)).
            T: timepoints \n
            C: scouts

    """
    data_dict: Dict[int, List[np.ndarray]] = {}

    for sub_dir in sorted(glob.glob(os.path.join(base_dir, sub_glob))):
        
        folder_name = os.path.basename(sub_dir)
        match = re.search(r"(\d+)$", folder_name)  # capture trailing digits
        if match:
            sid = int(match.group(1))
        else:
            sid = abs(hash(folder_name)) % (10**6)

        csv_files = sorted(glob.glob(os.path.join(sub_dir, "*.csv")))
        if not csv_files:
            print(f"[WARN] No CSV in {sub_dir}, skipping.")
            continue

        out_root = Path(sub_dir) / "scout_data"
        out_root.mkdir(parents=True, exist_ok=True)

        scout_names_ref = None
        trials: List[np.ndarray] = []

        for i, csv_path in enumerate(csv_files, start=1):
            csv_p = Path(csv_path)
            npy_p = out_root / f"sub{sid:03d}_trial{i:02d}.npy"

            # Decide whether to reuse cached file
            use_cache = npy_p.exists() and (not force and _is_newer(npy_p, csv_p))
            if use_cache:
                try:
                    X = np.load(npy_p, mmap_mode=None)
                except Exception as e:
                    print(f"[WARN] Failed to load cached {npy_p}: {e}. Rebuilding...")
                    use_cache = False

            if not use_cache:
                try:
                    X, scout_names = load_trial_csv(csv_p)
                    if scout_names_ref is None:
                        scout_names_ref = scout_names
                    elif scout_names != scout_names_ref:
                        raise ValueError(f"[{sub_dir}] Scout order mismatch in {csv_p.name}")

                    os.makedirs(os.path.dirname(npy_p), exist_ok=True)
                    tmp_p = npy_p.with_suffix(".npy.tmp")

                    try:
                        os.makedirs(os.path.dirname(tmp_p), exist_ok=True)
                        with open(tmp_p, "wb") as f:
                            np.save(f, X)
                            f.flush()
                            os.fsync(f.fileno())
                        if os.path.exists(tmp_p):
                            os.replace(tmp_p, npy_p)
                            print(f"[OK] Saved {npy_p.name} (shape={X.shape})")
                        else:
                            print(f"[WARN] tmp file not found right after saving {npy_p.name}")
                    except Exception as e:
                        print(f"[ERROR] np.save failed for {npy_p.name}: {e}")

                except Exception as e:
                    print(f"[ERROR] Failed to process {csv_p.name}: {e}")
                    continue

            trials.append(X)

        # Write meta.json
        meta_p = out_root / f"meta_sub{sid:03d}.json"
        meta = {
            "subject_id": sid,
            "n_trials": len(trials),
            "n_scouts": len(scout_names_ref) if scout_names_ref else None,
            "scout_names": scout_names_ref,
            "csv_files": [Path(p).name for p in csv_files],
            "cache_dir": str(out_root),
            "last_updated": time.time(),
        }

        rewrite_meta = force or (not meta_p.exists())
        if not rewrite_meta:
            try:
                old = json.loads(meta_p.read_text())
                rewrite_meta = (
                    old.get("n_trials") != meta["n_trials"]
                    or old.get("n_scouts") != meta["n_scouts"]
                    or old.get("scout_names") != meta["scout_names"]
                )
            except Exception:
                rewrite_meta = True

        if rewrite_meta:
            tmp_mp = meta_p.with_suffix(".json.tmp")
            tmp_mp.write_text(json.dumps(meta, indent=2))
            os.replace(tmp_mp, meta_p)
            print(f"[OK] Updated meta for sub-{sid:03d}")

        data_dict[sid] = trials

    if not data_dict:
        raise FileNotFoundError(f"No subjects found under {base_dir}/{sub_glob}")

    print(f"\n✅ Finished building data for {len(data_dict)} subjects.")
    return data_dict
# === end Functions ===

# =============== DATA PREPROCESSING ============== #

# ===== 1. SANITY CHECK: NaN and zero variance ====== #
# === Functions ===
def check_data_object(trials):
    """
    Check the trials (npy arrays) contained in an osl_dynamics.data.Data object.
    Detects NaN or Inf values and prints per-scout variance (min/median/max).

    Parameters
    ----------
    trials : List[np.ndarray]
        List of trial arrays (each shape (T, C))
            T: timepoints \n
            C: scouts

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

# # ===== 2. DOWNSAMPLING at 250 Hz (from 5 kHz)  ====== #
# === Functions ===
def downsample_data_object(trials, orig_sfreq, target_sfreq=250):
    """
    Downsample each trial to target_sfreq.

    Parameters
    ----------
    trials : List[np.ndarray]
        List of trial arrays (each shape (T, C))
            T: timepoints \n
            C: scouts
    orig_sfreq : float
        Original sampling frequency (e.g., 5000 Hz).
    target_sfreq : float
        Desired sampling frequency (default 250 Hz).

    Returns
    -------
    downsampled_trials : List[np.ndarray]
        List of downsampled trial arrays (each shape (T_new, C))
            T_new = floor(T * target_sfreq / orig_sfreq) 

       
    """
    up = int(target_sfreq)
    down = int(orig_sfreq)

    # Ensure it's a list of 2D arrays (time x channels)
    if isinstance(trials, np.ndarray) and trials.dtype == object:
        trials = list(trials)

    # Standardize each trial
    downsampled_trials = []
    for trial in trials:
        X_down = resample_poly(trial, up, down, axis=0)
        downsampled_trials.append(X_down)

    return downsampled_trials
# === end Functions ===

# ===== 3. STANDARDIZATION - z-score method  ====== #
# === Functions ===
def standardize_data_object(trials):
    """
    Standardize the trials using z-score normalization.

    Parameters
    ----------
    trials : List[np.ndarray]
        List of trial arrays (each shape (T, C))
            T: timepoints \n
            C: scouts   
    
    Returns
    -------
    standardized_trials : List[np.ndarray]
        List of standardized trial arrays (each shape (T, C));   
        each column (scout) has mean ~0 and std ~1 across timepoints

    """
    
    # Ensure it's a list of 2D arrays (time x channels)
    if isinstance(trials, np.ndarray) and trials.dtype == object:
        trials = list(trials)

    # Standardize each trial
    standardized_trials = []
    for trial in trials:
        standardized_trial = standardize(trial, axis=0, create_copy=True)
        standardized_trials.append(standardized_trial)

    return standardized_trials
# === end Functions ===

# ===== 4. TIME-EMBEDDING ====== #
# === Functions ===
def tdelay_embed_data_object(trials, n_embeddings):
    """
    Apply time delay embedding to the trials.

    Parameters
    ----------
    trials : List[np.ndarray]
        List of trial arrays (each shape (T, C))
        T: timepoints \n
        C: scouts
    n_embeddings : int
        Number of time embeddings (e.g., 15 for 60 ms at 250 Hz)

    Returns
    -------
    embedded_trials : List[np.ndarray]
        List of time-embedded trial arrays (each shape (T_new, C * n_embeddings))
            T_new = T - n_embeddings + 1 due to embedding.

    """
    
    # Ensure it's a list of 2D arrays (time x scouts)
    if isinstance(trials, np.ndarray) and trials.dtype == object:
        trials = list(trials)

    # Apply time delay embedding to each trial
    embedded_trials = []
    for trial in trials:
        embedded_trial = time_embed(trial, n_embeddings=n_embeddings)
        embedded_trials.append(embedded_trial)

    return embedded_trials
# === end Functions ===

# ===== 5. PCA ====== #
# === Functions ===
def fit_global_pca_fixed(data_dict, n_channels=62, whiten=True):
    """
    Fit one PCA model across all trials and subjects using an incremental approach. 
    Trials are divided into batches to compute the PCA components without loading everything into memory at once. Each batch slightly refines the estimate of PCs and explained variance. 
    The number of components is set to 2 × n_channels (124 for 62 scouts) to capture both amplitude and phase information.

    Parameters
    ----------
    data_dict : Dict[int, List[np.ndarray]]
        Dictionary mapping subject_id to list of trial arrays (each shape (T, C)).
            T: timepoints \n
            C: scouts
    n_channels : int
        Number of scouts (default 62).
    whiten : bool
        Whether to whiten the data in PCA (default True).

    Returns
    -------
    pca : PCA object
        Fitted PCA model that can be applied to all trials.
    n_components : int
        Number of PCA components used (2 × n_channels).

    """
    # Number of components = 2 × n_channels
    n_components = 2 * n_channels

    # Collect all trials
    all_trials = [trial for trials in data_dict.values() for trial in trials]

    # Stack all data
    X_all = np.vstack(all_trials)

    # First: fit full PCA to obtain variance profile
    pca_full = PCA(whiten=whiten).fit(X_all)

    # Compute cumulative explained variance
    cum_var = np.cumsum(pca_full.explained_variance_ratio_)
    
    # Print stats
    print("=== PCA GLOBAL VARIANCE PROFILE ===")
    print(f"Total components available: {len(pca_full.explained_variance_ratio_)}")
    print(f"Using n_components = {n_components}")
    print(f"Explained variance (first {n_components} PCs) = "
          f"{cum_var[n_components-1]*100:.2f}%")

    # Fit PCA with fixed n_components
    pca = PCA(n_components=n_components, whiten=whiten).fit(X_all)

    return pca, n_components


def apply_pca_to_all(data_dict, pca):
    """
    Apply the fitted PCA model retunred by 'fit_global_pca_fixed' to all subjects/trials.

    Parameters
    ----------  
    data_dict : Dict[int, List[np.ndarray]]
        Dictionary mapping subject_id to list of trial
    pca : PCA object
        Fitted PCA model that can be applied to all trials.

    Returns
    -------
    pca_data_dict : Dict[int, List[np.ndarray]]
        Dictionary mapping subject_id to list of PCA-transformed trial arrays (each shape (T, n_components)).
            T: timepoints \n
            n_components: number of PCA components (e.g., 124 for 62 scouts).
    
    """
    pca_data_dict = {}
    for sid, trials in data_dict.items():
        trials = list(trials)
        pca_data_dict[sid] = [pca.transform(trial) for trial in trials]
    return pca_data_dict

# === end Functions ===


# =============== TDE-HMM MODEL ============== #

# ===== 1. Chunck-based early stopping model FITTING ====== #
# === Functions ===
def fit_with_val_es(cfg: Config, data_train: Data, data_val: Data,
                    total_epochs=200, chunk=10, patience=3):
    """
    Fit the HMM-TDE model on training dataset while computing free energy on validation dataset. 
    A chunk-based approach and early-stopping are used. 

    Parameters
    ----------
    cfg : osl_dynamics.models.hmm.Config
        Configuration object for the HMM model.
    data_train : osl_dynamics.data.Data
        Training dataset (list of trial arrays).
    data_val : osl_dynamics.data.Data
        Validation dataset (list of trial arrays).
    total_epochs : int
        Maximum total number of epochs to train (default 200).
    chunk : int
        Number of epochs to train before each validation check (default 10).
    patience : int
        Number of consecutive validation checks with no improvement before stopping (default 3).

    Returns
    -------
    model : osl_dynamics.models.hmm.Model
        The HMM model fitted with the best weights found during training.
    best_val_energy : float
        The best free energy achieved on the validation set during training.

    """
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
    """
    For each random search combination of hyperparameters, build a Config object with fixed settings for all other parameters.

    Parameters
    ----------
    model_name : str
        Name of the model (e.g., "HMM")
    n_states : int
        Number of HMM states (K)
    n_channels : int
        Number of channels (C) after PCA (e.g., 124 for 62 scouts with 2 embeddings)
    sequence_length : int
        Length of each sequence (T) after time embedding (e.g., 200 )    
    learn_means : bool
        Whether to learn state means (default False for TDE-HMM)
    learn_covariances : bool
        Whether to learn state covariances (default True for TDE-HMM)
    initial_means : np.ndarray or None
        Initial state means (shape (K, C)) or None for random initialization (default None)
    initial_covariances : np.ndarray or None
        Initial state covariances (shape (K, C, C)) or None for random initialization (default None)
    diagonal_covariances : bool
        Whether to constrain covariances to be diagonal (default False for TDE-HMM)
    covariances_epsilon : float
        Small value added to covariances for numerical stability (default 1e-6)
    initial_trans_prob : np.ndarray or None
        Initial transition probability matrix (shape (K, K)) or None for random initialization (default None)
    learn_trans_prob : bool
        Whether to learn transition probabilities (default True for TDE-HMM)    
    trans_prob_update_delay : int
        Number of epochs to wait before starting to update transition probabilities (default 5)
    trans_prob_update_forget : float
        Forgetting factor for updating transition probabilities (default 0.7)
    initial_state_probs : np.ndarray or None
        Initial state occupancy probabilities (shape (K,)) or None for random initialization (default None
    learn_initial_state_probs : bool
        Whether to learn initial state occupancy probabilities (default True for TDE-HMM)
    baum_welch_implementation : str
        Implementation to use for Baum-Welch algorithm (default "log" for numerical stability)  
    batch_size : int
        Number of sequences per training batch (default 16)
    learning_rate : float
        Learning rate for the optimizer (default 3e-4)
    lr_decay : float
        Learning rate decay factor applied after each epoch (default 0.05)
    n_epochs : int
        Number of epochs to train (default 10 per chunk in fit_with_val_es)
    loss_calc : str
        Method to calculate loss ("mean" or "sum", default "mean")
    optimizer : str
        Optimizer to use for training (default "adam")
    gradient_clip : float or None
        Maximum norm for gradient clipping (default None for no clipping)
    multiple_dynamics : bool    
        Whether to fit multiple dynamics models (default False for single HMM)
    multi_gpu : bool
        Whether to use multiple GPUs for training (default False)

    Returns
    -------
    config : osl_dynamics.models.hmm.Config
         Config object with the specified hyperparameters and fixed settings for other parameters.

    
    """
    return Config(
        model_name="HMM",
        n_states=n_states,
        n_channels=int(k),
        sequence_length=200,

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
        batch_size=16,
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
    Convert input data (either dict of subject_id -> list of trials or legacy list of trials) into a Data object and group labels for GroupKFold validation.

    Parameters
    ----------
    data : Union[List[np.ndarray], Dict[int, List[np.ndarray]]]
        Input data (either dict of subject_id -> list of trials or legacy list of trials)

    Returns
    -------
    data : osl_dynamics.data.Data
        Data object containing the sequences
    groups : np.ndarray[int]
        Subject id per sequence
    is_multi : bool
        Whether the data is multi-subject

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
    """
    Create a new Data object containing only the sequences indexed by seq_idx from the original Data object. 
    This is used to create training and validation subsets for each fold in GroupKFold.

    Parameters
    ----------
    data_all : osl_dynamics.data.Data
        Original Data object containing all sequences.

    seq_idx : np.ndarray
        Indices of sequences to include in the subset.

    Returns
    -------
    data : osl_dynamics.data.Data
        Data object containing only the specified sequences.

    """
    # Rebuild a Data object from a subset of sequences
    seqs = [data_all.arrays[i] for i in seq_idx.tolist()]
    return Data(seqs)

def grid_search_tdehmm(
    data_input,               
    K: int = 5,               
    total_epochs: int = 200,
    chunk: int = 10,
    patience: int = 3,
):
    """
    Perform a grid search over specified hyperparameters for the TDE-HMM model using GroupKFold validation. 
    This function calls all the following functions in sequence:
        - pack_sequences_and_groups
        - subset_Data
        - build_config
        - fit_with_val_es
    
    Parameters
    ----------
    data_input : Union[List[np.ndarray]], Dict[int, List[np.ndarray]]]
        Input data (either dict of subject_id -> list of trials or legacy list of trials)
    K : int
        Number of folds for GroupKFold validation (default 5)
    total_epochs : int
        Maximum total number of epochs to train for each hyperparameter combination (default 200)
    chunk : int
        Number of epochs to train before each validation check in fit_with_val_es (default 10)
    patience : int
        Number of consecutive validation checks with no improvement before stopping in fit_with_val_es (default 3)

    Returns
    -------
    best_model : osl_dynamics.models.hmm.Model
        The HMM model fitted with the best hyperparameters found during grid search.
    results_sorted : List[Tuple[Dict[str, Any], float]]
        List of tuples containing hyperparameter combinations and their corresponding mean validation free energy, sorted by performance (best first).
        
    """

    # 0) Pack sequences and group ids (subjects)
    data_all, groups, is_multi = pack_sequences_and_groups(data_input)
    print(f"Total sequences: {len(data_all.arrays)}")
    assert len(data_all.arrays) > 0, "Data object is empty!"
    n_seq = len(data_all.arrays)

    # 1) Define search space (grid)
    space = {
        "n_states": [8, 9], #[6, 8, 10]
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