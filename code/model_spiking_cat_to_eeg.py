# Minimal model->EEG(64ch) projection using an fsaverage template forward model.
#
# Inputs (you provide):
#   v1.npy, pmd.npy, m1.npy  (each shape: [n_times], same length)
#
# Outputs:
#   eeg_64.npy  (shape: [64, n_times]) predicted EEG in Volts (arbitrary scaling)

import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.datasets import fetch_fsaverage

# -----------------------------
# 1) Load model time series
# -----------------------------

# izp = np.array([
#     [50, -80, -25, 40, 0.01, -20, -55, 150, 1],  # dms A 0 (MSN)
#     [50, -80, -25, 40, 0.01, -20, -55, 150, 1],  # dms B 1 (MSN)
#     [100, -60, -40, 35, 0.03, -2, -50, 100, 0.7],  # premotor A 2
#     [100, -60, -40, 35, 0.03, -2, -50, 100, 0.7],  # premotor B 3
#     [50, -80, -25, 40, 0.01, -20, -55, 150, 1],  # dls A 4 (MSN)
#     [50, -80, -25, 40, 0.01, -20, -55, 150, 1],  # dls B 5 (MSN)
#     [100, -60, -40, 35, 0.03, -2, -50, 100, 0.7],  # motor A 6
#     [100, -60, -40, 35, 0.03, -2, -50, 100, 0.7],  # motor B 7
# ])

g =  np.load("../output/model_spiking_lesion_trials_0-9_cells_DLS_g.npy")
n_times = g.shape[-1]

# TODO: implement properly later
# trl = -1
# v1 = g[0, 0, trl, :]
# pmd_A = g[2, 0, trl, :]
# pmd_B = g[3, 0, trl, :]
# m1_A = g[6, 0, trl, :]
# m1_B = g[7, 0, trl, :]
# 
# # TODO: fix this hack
# pmd = pmd_A
# m1 = m1_A

# TODO: implement properly later by loading from model output
rng = np.random.default_rng(1)
v1 = rng.poisson(0.1, size=(n_times, 1)).astype(float)
pm = rng.poisson(0.1, size=(n_times, 1)).astype(float)
m1 = rng.poisson(0.1, size=(n_times, 1)).astype(float)
v1 = np.convolve(v1[:, 0], np.exp(-np.arange(0, 200) / 20.0), mode="full")[:n_times]
pm = np.convolve(pm[:, 0], np.exp(-np.arange(0, 200) / 20.0), mode="full")[:n_times]
m1 = np.convolve(m1[:, 0], np.exp(-np.arange(0, 200) / 20.0), mode="full")[:n_times]

# scale model signals to something dipole-moment-like.
# EEG forward models assume source amplitude in Am (Ampere-meters).
# If you don't know scaling yet, keep as-is and later fit a global scale factor.
scale = 1e-9
v1 *= scale
pmd *= scale
m1 *= scale

# -----------------------------
# 2) Get fsaverage + BioSemi-64
# -----------------------------
fs_dir = fetch_fsaverage(verbose=True)
subjects_dir = fs_dir.parent
subject = "fsaverage"
trans = "fsaverage"  # built-in transform for fsaverage

# Source space + BEM that ship with MNE fsaverage dataset
src_path = fs_dir / "bem" / "fsaverage-ico-5-src.fif"
bem_path = fs_dir / "bem" / "fsaverage-5120-5120-5120-bem-sol.fif"

src = mne.read_source_spaces(src_path)
bem = mne.read_bem_solution(bem_path)

# BioSemi-64 montage
montage = mne.channels.make_standard_montage("biosemi64")
info = mne.create_info(ch_names=montage.ch_names, sfreq=1000.0, ch_types="eeg")  # sfreq just metadata here
info.set_montage(montage)

# -----------------------------
# 3) Compute forward (leadfield)
# -----------------------------
fwd = mne.make_forward_solution(
    info=info,
    trans=trans,
    src=src,
    bem=bem,
    eeg=True,
    meg=False,
    mindist=5.0,
    n_jobs=1,
    verbose=True,
)

# Convert to fixed orientation (normal to cortex)
fwd = mne.convert_forward_solution(fwd, force_fixed=True, use_cps=True)

# fwd['sol']['data'] is the gain matrix: (n_sensors, n_sources)
G = fwd["sol"]["data"]

# -----------------------------
# 4) Pick nearest fsaverage vertex to target MNI coords
#    (We do this in MNI space for convenience.)
# -----------------------------
# Example target MNI coordinates (mm):
#   V1:  (-8, -80,  4)  (left)  -- example reported for V1 activation
#   PMd: (-39,   3, 54)  (left)  -- example PMd coordinate
#   M1:  (-38, -26, 60)  (left)  -- example M1 hand-ish coordinate
#
# Notes:
# - These are starting points. You can swap in coordinates you prefer.
# - We'll place ONE vertex per region (and two for V1: left+right).

targets = [
    ("V1-lh", "lh", np.array([-8.0, -80.0,  4.0])),
    ("V1-rh", "rh", np.array([+8.0, -80.0,  4.0])),  # mirrored heuristic
    ("PMd-lh", "lh", np.array([-39.0,  3.0, 54.0])),
    ("M1-lh",  "lh", np.array([-38.0, -26.0, 60.0])),
]

# Forward/source ordering in MNE is typically: all LH vertices, then all RH vertices (for surface source spaces).
lh_verts = src[0]["vertno"]
rh_verts = src[1]["vertno"]
n_lh = len(lh_verts)

# Precompute vertex MNI coords for each hemi
lh_mni = mne.vertex_to_mni(lh_verts, hemis=0, subject=subject, subjects_dir=subjects_dir).astype(float)
rh_mni = mne.vertex_to_mni(rh_verts, hemis=1, subject=subject, subjects_dir=subjects_dir).astype(float)

picked_cols = []
picked_names = []

for name, hemi, mni_xyz in targets:
    if hemi == "lh":
        d = np.linalg.norm(lh_mni - mni_xyz[None, :], axis=1)
        ii = int(np.argmin(d))
        vert = int(lh_verts[ii])
        col = ii  # LH columns start at 0
    elif hemi == "rh":
        d = np.linalg.norm(rh_mni - mni_xyz[None, :], axis=1)
        ii = int(np.argmin(d))
        vert = int(rh_verts[ii])
        col = n_lh + ii  # RH columns come after LH
    else:
        raise ValueError("hemi must be 'lh' or 'rh'")

    picked_names.append(f"{name}@vert{vert}")
    picked_cols.append(col)

picked_cols = np.array(picked_cols, dtype=int)

print("Picked sources:")
for nm, col in zip(picked_names, picked_cols):
    print(" ", nm, "-> G column", int(col))

# -----------------------------
# 5) Build source time series and project to EEG
# -----------------------------
# We map: V1-lh and V1-rh get the same v1(t) by default.
# PMd-lh gets pmd(t), M1-lh gets m1(t).
S = np.zeros((len(picked_cols), n_times), dtype=float)
S[0, :] = v1
S[1, :] = v1
S[2, :] = pmd
S[3, :] = m1

G_sel = G[:, picked_cols]              # (64, 4)
eeg_64 = G_sel @ S                     # (64, n_times)

np.save("eeg_64.npy", eeg_64)
print("Saved eeg_64.npy with shape", eeg_64.shape)

# -----------------------------
# 6) write an MNE Raw file for quick plotting/inspection
# -----------------------------
raw_pred = mne.io.RawArray(eeg_64, info)
raw_pred.set_eeg_reference("average", projection=True)

sfreq = raw_pred.info["sfreq"]

data = raw_pred.get_data()

windows = [
    (0.10, 0.14),
    (0.20, 0.24),
    (0.30, 0.34),
    (0.40, 0.44),
]

# PLOT TOPOMAPS
fig, axes = plt.subplots(1, len(windows), figsize=(3 * len(windows), 3))

for ax, (t0, t1) in zip(axes, windows):
    i0 = int(round(t0 * sfreq))
    i1 = int(round(t1 * sfreq))

    topo = data[:, i0:i1].mean(axis=1)

    mne.viz.plot_topomap(
        topo,
        raw_pred.info,
        axes=ax,
        show=False,
        contours=0,
    )
    ax.set_title(f"{t0:.2f}–{t1:.2f} s")

plt.suptitle("Predicted EEG (window-averaged topographies)", y=0.95)
plt.tight_layout()
plt.show()


# PLOT TIME SERIES
raw_pred.apply_proj()
data = raw_pred.get_data()
sfreq = raw_pred.info["sfreq"]
t = np.arange(data.shape[1]) / sfreq

# Choose a short window to zoom into (seconds)
t0, t1 = 0.0, 1.0
i0 = int(t0 * sfreq)
i1 = int(t1 * sfreq)

# Convert to µV for visibility
data_uv = data[:, i0:i1] * 1e6
t_win = t[i0:i1]

plt.figure(figsize=(10, 4))
plt.plot(t_win, data_uv.T, linewidth=0.8)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (µV)")
plt.title("Predicted EEG butterfly (zoomed, µV)")
plt.show()

