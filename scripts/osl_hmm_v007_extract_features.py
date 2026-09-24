import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import gc
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from osl_dynamics.data import Data
from osl_dynamics.models import load

try:
    import mne
    HAVE_MNE = True #non-negative matrix factorization (NMF) is used for spectral decomposition if MNE is available
except Exception:
    HAVE_MNE = False


RESULTS_DIR = Path("/mnt/uphummel/scratch/nbenlamr/osl_results")
GROUP_LABELS = RESULTS_DIR / "group_labels_all.pkl"
MODEL_DIR = RESULTS_DIR / "Results_HMM_bestconfig_all"

PCA_DATA = RESULTS_DIR / "data_dict_all_pca.pkl"
STAND_DATA = RESULTS_DIR / "data_dict_all_stand.pkl"
#LEFT_DATA_DIR = Path("/mnt/uphummel/scratch/nbenlamr/osl_data//left_lesion")
#to include the 3 types 
DATA_DIRS = {
    "healthyold": Path("/mnt/uphummel/scratch/nbenlamr/osl_data/healthyold"),
    "left_lesion": Path("/mnt/uphummel/scratch/nbenlamr/osl_data/acutestroke_left_lesion"),
    "right_lesion_flipped": Path("/mnt/uphummel/scratch/nbenlamr/osl_data/acutestroke_right_lesion_flipped"),
}

OUT_DIR = RESULTS_DIR / "features_all"
PLOT_DIR = OUT_DIR / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

SFREQ = 250
N_EMBEDDINGS = 15

BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma_low": (30, 45),
}


def load_scout_names():
    #meta_files = sorted(LEFT_DATA_DIR.glob("TiMeS_WP11_*/scout_data/meta_sub*.json"))
    #if not meta_files:
    #    raise FileNotFoundError(f"No meta_sub*.json files found under {LEFT_DATA_DIR}")

    meta_files = []

    for group_name, data_dir in DATA_DIRS.items():
        meta_files.extend(sorted(data_dir.rglob("scout_data/meta_sub*.json")))

    if not meta_files:
        searched = "\n".join(str(p) for p in DATA_DIRS.values())
        raise FileNotFoundError(f"No meta_sub*.json files found under:\n{searched}")

    with open(meta_files[0], "r") as f:
        ref = json.load(f)

    scout_names = ref["scout_names"]

    for mf in meta_files[1:]:
        with open(mf, "r") as f:
            meta = json.load(f)
        if meta["scout_names"] != scout_names:
            raise ValueError(f"Scout order mismatch in {mf}")

    print(f"[INFO] Loaded {len(scout_names)} scout names from {meta_files[0]}")
    return scout_names


def pack_sequences(data_dict, group_labels):
    seqs, meta = [], []
    for sid, trials in sorted(data_dict.items()):
        group = group_labels.get(sid, "unknown")
        for trial_idx, x in enumerate(trials):
            seqs.append(x)
            meta.append({
                "subject_id": int(sid),
                "trial_idx": int(trial_idx),
                "group": group,
            })
    return seqs, meta


def argmax_stc(alpha): #This function finds the most likely state at each time point
    labels = np.argmax(alpha, axis=1)
    stc = np.zeros_like(alpha, dtype=int)
    stc[np.arange(len(labels)), labels] = 1
    return stc, labels


def run_lengths(labels, state): #Computes how long each visit to a state lasts.
    x = labels == state
    if not np.any(x):
        return np.array([], dtype=int)

    padded = np.r_[False, x, False]
    starts = np.where(np.diff(padded.astype(int)) == 1)[0]
    ends = np.where(np.diff(padded.astype(int)) == -1)[0]
    return ends - starts


def temporal_features_one(alpha, sfreq): #Extracts temporal features for one trial.
#fractional_occupancy: proportion of time spent in a state
#mean_lifetime_s: average duration of state visits
#max_lifetime_s: longest visit
#n_visits: number of times the state appears
#mean_interval_s: average time between visits
#switching_rate_hz: how often the model switches states
#mean_alpha: average probability of the state
    _, labels = argmax_stc(alpha)
    n_samples, n_states = alpha.shape
    duration = n_samples / sfreq
    switches = np.sum(labels[1:] != labels[:-1])

    rows = []
    for state in range(n_states):
        visits = run_lengths(labels, state)
        is_state = labels == state

        starts = np.where(np.diff(np.r_[False, is_state].astype(int)) == 1)[0]
        if len(starts) > 1:
            intervals = np.diff(starts) / sfreq
            mean_interval = float(np.mean(intervals))
        else:
            mean_interval = np.nan

        rows.append({
            "state": state,
            "fractional_occupancy": float(np.mean(is_state)),
            "mean_lifetime_s": float(np.mean(visits) / sfreq) if len(visits) else 0.0,
            "max_lifetime_s": float(np.max(visits) / sfreq) if len(visits) else 0.0,
            "n_visits": int(len(visits)),
            "mean_interval_s": mean_interval,
            "switching_rate_hz": float(switches / duration) if duration > 0 else np.nan,
            "mean_alpha": float(np.mean(alpha[:, state])),
        })

    return rows


def covariance_to_correlation(cov): #Converts a covariance matrix into a correlation matrix.

#Covariance depends on scale; correlation is normalized between -1 and 1, making it easier to interpret.

    d = np.sqrt(np.diag(cov))
    denom = np.outer(d, d)
    corr = np.divide(cov, denom, out=np.zeros_like(cov), where=denom != 0)
    np.fill_diagonal(corr, 1.0)
    return corr


def save_spatial_network_features(model): #When State 3 is active, what frequency content is present?
                                          #Uses MNE if available, otherwise SciPy.
    covs = np.asarray(model.get_covariances())
    np.save(OUT_DIR / "state_covariances_pca_space.npy", covs)

    corrs = np.stack([covariance_to_correlation(c) for c in covs])
    np.save(OUT_DIR / "state_correlations_pca_space.npy", corrs)

    rows = []
    for state, corr in enumerate(corrs):
        abs_corr = np.abs(corr.copy())
        np.fill_diagonal(abs_corr, 0.0)
        node_strength = abs_corr.sum(axis=1)

        rows.append({
            "state": state,
            "mean_abs_connectivity": float(abs_corr.mean()),
            "max_abs_connectivity": float(abs_corr.max()),
            "mean_node_strength": float(node_strength.mean()),
            "max_node_strength": float(node_strength.max()),
        })

    pd.DataFrame(rows).to_csv(
        OUT_DIR / "spatial_network_features_pca_space.csv",
        index=False,
    )


def align_standardized_trial(x_stand, alpha, n_embeddings):
    target_len = alpha.shape[0]
    start = n_embeddings // 2
    end = start + target_len

    if end <= x_stand.shape[0]:
        return x_stand[start:end]

    return x_stand[:target_len]


def weighted_welch_by_state(x, alpha, sfreq, nperseg=512):
    n_states = alpha.shape[1]
    freqs = None
    spectra = []

    for state in range(n_states):
        weights = alpha[:, state]
        if np.sum(weights) <= 0:
            spectra.append(None)
            continue

        x_weighted = x * weights[:, None]

        if HAVE_MNE:
            psd, freqs = mne.time_frequency.psd_array_welch(
                x_weighted.T,
                sfreq=sfreq,
                fmin=0,
                fmax=sfreq / 2,
                n_fft=min(nperseg, x_weighted.shape[0]),
                n_overlap=0,
                verbose=False,
            )
            spectra.append(psd)
        else:
            freqs, pxx = signal.welch(
                x_weighted,
                fs=sfreq,
                axis=0,
                nperseg=min(nperseg, x_weighted.shape[0]),
            )
            spectra.append(pxx.T)

    return freqs, spectra


def bandpower(freqs, psd, fmin, fmax):
    mask = (freqs >= fmin) & (freqs < fmax)
    if not np.any(mask):
        return np.full(psd.shape[0], np.nan)
    return np.trapz(psd[:, mask], freqs[mask], axis=1)


def save_spectral_features(alpha_list, meta, stand_dict):
    rows = []
    spectra_accum = {}
    freqs_ref = None

    for alpha, m in zip(alpha_list, meta):
        sid = m["subject_id"]
        trial_idx = m["trial_idx"]

        x_stand = stand_dict[sid][trial_idx]
        x = align_standardized_trial(x_stand, alpha, N_EMBEDDINGS)

        n = min(x.shape[0], alpha.shape[0])
        x = x[:n]
        alpha = alpha[:n]

        freqs, state_spectra = weighted_welch_by_state(x, alpha, SFREQ)
        freqs_ref = freqs

        for state, psd in enumerate(state_spectra):
            if psd is None:
                continue

            spectra_accum.setdefault(state, []).append(psd)

            row = {
                "subject_id": sid,
                "trial_idx": trial_idx,
                "group": m["group"],
                "state": state,
            }

            for band, (fmin, fmax) in BANDS.items():
                bp = bandpower(freqs, psd, fmin, fmax)
                row[f"{band}_power_mean"] = float(np.nanmean(bp))
                row[f"{band}_power_max"] = float(np.nanmax(bp))

            rows.append(row)

    pd.DataFrame(rows).to_csv(
        OUT_DIR / "spectral_features_by_trial.csv",
        index=False,
    )

    mean_spectra = {}
    for state, psds in spectra_accum.items():
        mean_spectra[int(state)] = np.mean(np.stack(psds), axis=0)

    with open(OUT_DIR / "state_mean_spectra_scout_space.pkl", "wb") as f:
        pickle.dump({"freqs": freqs_ref, "mean_spectra": mean_spectra}, f)


def save_scout_bandpower_features(alpha_list, meta, stand_dict, scout_names):
    rows = []

    for alpha, m in zip(alpha_list, meta):
        sid = m["subject_id"]
        trial_idx = m["trial_idx"]

        x_stand = stand_dict[sid][trial_idx]
        x = align_standardized_trial(x_stand, alpha, N_EMBEDDINGS)

        n = min(x.shape[0], alpha.shape[0])
        x = x[:n]
        alpha = alpha[:n]

        freqs, state_spectra = weighted_welch_by_state(x, alpha, SFREQ)

        for state, psd in enumerate(state_spectra):
            if psd is None:
                continue

            for band, (fmin, fmax) in BANDS.items():
                bp = bandpower(freqs, psd, fmin, fmax)

                for scout_idx, power in enumerate(bp):
                    rows.append({
                        "subject_id": sid,
                        "trial_idx": trial_idx,
                        "group": m["group"],
                        "state": state,
                        "scout_idx": scout_idx,
                        "scout_name": scout_names[scout_idx],
                        "band": band,
                        "power": float(power),
                    })

    out = OUT_DIR / "scout_bandpower_long.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[INFO] Saved scout-level bandpower: {out}")


def plot_alpha_and_states(alpha, meta, max_points=2000):
    a = alpha[0]
    m = meta[0]
    n = min(max_points, a.shape[0])
    labels = np.argmax(a[:n], axis=1)

    fig, ax = plt.subplots(figsize=(14, 3))
    im = ax.imshow(labels[None, :], aspect="auto", interpolation="nearest", cmap="tab20")
    ax.set_yticks([])
    ax.set_xlabel("Time samples")
    ax.set_title(f"Example most likely state: subject {m['subject_id']}, trial {m['trial_idx']}")
    fig.colorbar(im, ax=ax, label="State")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "example_state_sequence.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 5))
    for state in range(a.shape[1]):
        ax.plot(a[:n, state], label=f"State {state}")
    ax.set_xlabel("Time samples")
    ax.set_ylabel("Probability")
    ax.set_title(f"Example state probabilities: subject {m['subject_id']}, trial {m['trial_idx']}")
    ax.legend(ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "example_state_probabilities.png", dpi=200)
    plt.close(fig)


def plot_temporal_features():
    temporal = pd.read_csv(OUT_DIR / "temporal_features_by_subject.csv")

    for feature in ["fractional_occupancy", "mean_lifetime_s", "switching_rate_hz", "n_visits"]:
        fig, ax = plt.subplots(figsize=(8, 5))
        temporal.boxplot(column=feature, by="state", ax=ax)
        ax.set_title(feature.replace("_", " ").title())
        ax.set_xlabel("State")
        ax.set_ylabel(feature)
        fig.suptitle("")
        fig.tight_layout()
        fig.savefig(PLOT_DIR / f"{feature}_by_state.png", dpi=200)
        plt.close(fig)


def plot_spatial_heatmaps():
    corr_path = OUT_DIR / "state_correlations_pca_space.npy"
    if not corr_path.exists():
        return

    corrs = np.load(corr_path)

    for state, corr in enumerate(corrs):
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_title(f"State {state} correlation matrix (PCA space)")
        ax.set_xlabel("PCA component")
        ax.set_ylabel("PCA component")
        fig.colorbar(im, ax=ax, label="Correlation")
        fig.tight_layout()
        fig.savefig(PLOT_DIR / f"state_{state:02d}_correlation_heatmap_pca_space.png", dpi=200)
        plt.close(fig)


def plot_spectra():
    spectra_path = OUT_DIR / "state_mean_spectra_scout_space.pkl"
    if not spectra_path.exists():
        print("[WARN] No spectra file found, skipping spectra plots.")
        return

    with open(spectra_path, "rb") as f:
        spec = pickle.load(f)

    freqs = spec["freqs"]
    mean_spectra = spec["mean_spectra"]

    fig, ax = plt.subplots(figsize=(9, 5))
    for state, psd in sorted(mean_spectra.items()):
        mean_psd = np.mean(psd, axis=0)
        ax.plot(freqs, mean_psd, label=f"State {state}")

    ax.set_xlim(1, 45)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power")
    ax.set_title("Mean state spectra")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "mean_state_spectra.png", dpi=200)
    plt.close(fig)


def plot_bandpower():
    band_path = OUT_DIR / "spectral_features_by_trial.csv"
    if not band_path.exists():
        return

    df = pd.read_csv(band_path)
    band_cols = [c for c in df.columns if c.endswith("_power_mean")]
    means = df.groupby("state")[band_cols].mean()

    fig, ax = plt.subplots(figsize=(11, 5))
    means.plot(kind="bar", ax=ax)
    ax.set_xlabel("State")
    ax.set_ylabel("Mean band power")
    ax.set_title("State-specific band power")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "state_bandpower_mean.png", dpi=200)
    plt.close(fig)


def plot_state_scout_heatmaps():
    path = OUT_DIR / "scout_bandpower_long.csv"
    if not path.exists():
        return

    df = pd.read_csv(path)

    summary = (
        df.groupby(["state", "scout_name", "band"], as_index=False)["power"]
        .mean()
    )

    for band in sorted(summary["band"].unique()):
        pivot = summary[summary["band"] == band].pivot(
            index="scout_name",
            columns="state",
            values="power",
        )

        fig, ax = plt.subplots(figsize=(10, 14))
        im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=6)
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels([f"State {s}" for s in pivot.columns], rotation=45)
        ax.set_title(f"DKT scout band power by state - {band}")
        fig.colorbar(im, ax=ax, label="Mean power")
        fig.tight_layout()
        fig.savefig(PLOT_DIR / f"dkt_scout_state_heatmap_{band}.png", dpi=200)
        plt.close(fig)


def plot_top_scout_bandpower():
    path = OUT_DIR / "scout_bandpower_long.csv"
    if not path.exists():
        return

    df = pd.read_csv(path)

    summary = (
        df.groupby(["state", "scout_name", "band"], as_index=False)["power"]
        .mean()
    )

    for band in sorted(summary["band"].unique()):
        for state in sorted(summary["state"].unique()):
            sub = summary[(summary["band"] == band) & (summary["state"] == state)]
            top = sub.sort_values("power", ascending=False).head(15)

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(top["scout_name"][::-1], top["power"][::-1])
            ax.set_xlabel("Mean power")
            ax.set_title(f"Top DKT scouts - State {state} - {band}")
            fig.tight_layout()
            fig.savefig(PLOT_DIR / f"top_dkt_scouts_state_{state:02d}_{band}.png", dpi=200)
            plt.close(fig)


def main():
    print("[INFO] Checking inputs")
    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Missing model directory: {MODEL_DIR}")
    if not PCA_DATA.exists():
        raise FileNotFoundError(f"Missing PCA data: {PCA_DATA}")
    if not STAND_DATA.exists():
        raise FileNotFoundError(f"Missing standardized data: {STAND_DATA}")
    if not GROUP_LABELS.exists():
        raise FileNotFoundError(f"Missing group labels: {GROUP_LABELS}")

    print(f"[INFO] MNE available: {HAVE_MNE}")

    scout_names = load_scout_names()
    print("[INFO] Loading PCA data")
    with open(PCA_DATA, "rb") as f:
        data_dict_pca = pickle.load(f)

    print("[INFO] Loading group labels")
    with open(GROUP_LABELS, "rb") as f:
        group_labels = pickle.load(f)

    seqs, meta = pack_sequences(data_dict_pca, group_labels)

    DATA_STORE_DIR = OUT_DIR / "tmp_data"
    DATA_STORE_DIR.mkdir(parents=True, exist_ok=True)
    data = Data(seqs, store_dir=str(DATA_STORE_DIR))

    print(f"[INFO] Loaded {len(seqs)} sequences from {len(data_dict_pca)} subjects")

    print("[INFO] Loading trained HMM")
    model = load(str(MODEL_DIR))

    print("[INFO] Inferring state probabilities")
    alpha = model.get_alpha(data)

    with open(OUT_DIR / "alpha.pkl", "wb") as f:
        pickle.dump({"alpha": alpha, "meta": meta}, f)

    print("[INFO] Extracting temporal features")
    temporal_rows = []
    for a, m in zip(alpha, meta):
        for row in temporal_features_one(a, SFREQ):
            temporal_rows.append({**m, **row})

    temporal = pd.DataFrame(temporal_rows)
    temporal.to_csv(OUT_DIR / "temporal_features_by_trial.csv", index=False)

    subject_temporal = (
        temporal
        .groupby(["group", "subject_id", "state"], as_index=False)
        .mean(numeric_only=True)
    )
    subject_temporal.to_csv(OUT_DIR / "temporal_features_by_subject.csv", index=False)

    print("[INFO] Extracting spatial/network features")
    save_spatial_network_features(model)

    print("[INFO] Loading standardized scout-space data")
    with open(STAND_DATA, "rb") as f:
        stand_dict = pickle.load(f)

    print("[INFO] Extracting spectral features")
    save_spectral_features(alpha, meta, stand_dict)

    print("[INFO] Extracting DKT scout-level bandpower features")
    save_scout_bandpower_features(alpha, meta, stand_dict, scout_names)

    print("[INFO] Creating plots")
    plot_alpha_and_states(alpha, meta)
    plot_temporal_features()
    plot_spatial_heatmaps()
    plot_spectra()
    plot_bandpower()
    plot_state_scout_heatmaps()
    plot_top_scout_bandpower()

    with open(OUT_DIR / "feature_extraction_summary.json", "w") as f:
        json.dump({
            "model_dir": str(MODEL_DIR),
            "pca_data": str(PCA_DATA),
            "stand_data": str(STAND_DATA),
            "group_labels": str(GROUP_LABELS),
            "groups": sorted(set(group_labels.values())),
            "data_dirs": {name: str(path) for name, path in DATA_DIRS.items()},
            "sfreq": SFREQ,
            "n_embeddings": N_EMBEDDINGS,
            "n_sequences": len(seqs),
            "n_subjects": len(data_dict_pca),
            "n_scouts": len(scout_names),
            "scout_names": scout_names,
            "mne_available": HAVE_MNE,
            "outputs": str(OUT_DIR),
            "plots": str(PLOT_DIR),
        }, f, indent=2)

    print(f"[DONE] Features saved to: {OUT_DIR}")
    print(f"[DONE] Plots saved to: {PLOT_DIR}")

    gc.collect()
    sys.exit(0)


if __name__ == "__main__":
    
    main()