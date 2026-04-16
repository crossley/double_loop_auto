from __future__ import annotations

import argparse
import os
from pathlib import Path

for env_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(env_var, "1")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = REPO_ROOT / "output"
FIGURES_DIR = REPO_ROOT / "figures"
OUTPUT_PREFIX = "model_spiking_cat_90vs180_gadi"


def build_prefix(fig_label: str, output_dir: Path) -> Path:
    return output_dir / f"{OUTPUT_PREFIX}_{fig_label}"


def simulate(lesioned_trials,
             lesion_cell_inds,
             lesion_mean,
             lesion_sd,
             fig_label,
             rotation,
             probe_trial_onsets,
             n_probe_trials,
             n_trials,
             n_simulations,
             output_dir,
             save_stimulus_plot=False,
             figures_dir=None,
             seed=1):

    np.random.seed(seed)

    ds, ds_90, ds_180 = make_stim_cats(
        n_trials // 2,
        save_plot=save_stimulus_plot,
        figures_dir=figures_dir,
        figure_stem=f"{OUTPUT_PREFIX}_stimuli_{fig_label}",
    )
    ds_0 = ds.sample(n=n_probe_trials, random_state=0).reset_index(drop=True)
    ds_90 = ds_90.sample(n=n_probe_trials, random_state=0).reset_index(drop=True)
    ds_180 = ds_180.sample(n=n_probe_trials, random_state=0).reset_index(drop=True)

    if rotation == 0:
        ds_probe = ds_0
    elif rotation == 90:
        ds_probe = ds_90
    elif rotation == 180:
        ds_probe = ds_180
    else:
        raise ValueError(f"Unsupported rotation: {rotation}")

    ds["phase"] = "train"
    ds_probe["phase"] = "probe"

    for onset in sorted(probe_trial_onsets, reverse=True):
        ds_top = ds.iloc[:onset, :].reset_index(drop=True)
        ds_bottom = ds.iloc[onset:, :].reset_index(drop=True)
        ds = pd.concat([ds_top, ds_probe, ds_bottom], ignore_index=True)

    n_trials = ds.shape[0]

    tau = 1
    T = 3000
    t = np.arange(0, T, tau)
    n_steps = t.shape[0]

    alpha_critic = 0.05

    nmda_thresh = 0.0

    alpha_w_vis_dms = 1e-9
    beta_w_vis_dms = 1e-11
    gamma_w_vis_dms = 0.0

    alpha_w_premotor_dls = 2e-15
    beta_w_premotor_dls = 1e-15
    gamma_w_premotor_dls = 0.0

    alpha_w_vis_premotor = 5e-11 * 0
    beta_w_vis_premotor = 5e-11 * 0

    alpha_w_premotor_motor = 1e-18 * 0
    beta_w_premotor_motor = 1e-18 * 0

    vis_dim = 100
    vis_amp = 7
    vis_sig = 7
    vis = np.zeros((vis_dim, vis_dim))
    w_vis_dms_A = np.zeros((vis_dim, vis_dim))
    w_vis_dms_B = np.zeros((vis_dim, vis_dim))
    w_vis_pm_A = np.zeros((vis_dim, vis_dim))
    w_vis_pm_B = np.zeros((vis_dim, vis_dim))

    cat = np.zeros((n_simulations, n_trials))
    resp = np.zeros((n_simulations, n_trials))
    rt = np.zeros((n_simulations, n_trials))
    r = np.zeros((n_simulations, n_trials))
    p = np.ones((n_simulations, n_trials)) * 0.5
    rpe = np.zeros((n_simulations, n_trials))

    izp = np.array([
        [50, -80, -25, 40, 0.01, -20, -55, 150, 1],
        [50, -80, -25, 40, 0.01, -20, -55, 150, 1],
        [100, -60, -40, 35, 0.03, -2, -50, 100, 0.7],
        [100, -60, -40, 35, 0.03, -2, -50, 100, 0.7],
        [50, -80, -25, 40, 0.01, -20, -55, 150, 1],
        [50, -80, -25, 40, 0.01, -20, -55, 150, 1],
        [100, -60, -40, 35, 0.03, -2, -50, 100, 0.7],
        [100, -60, -40, 35, 0.03, -2, -50, 100, 0.7],
    ])

    C, vr, vt, vpeak, a, b, c, d, k = izp.T

    mu = np.ones((n_trials, izp.shape[0]))
    sig = np.zeros((n_trials, izp.shape[0]))

    sig[:, 0] = 1
    sig[:, 1] = 1
    sig[:, 4] = 1
    sig[:, 5] = 1

    mu[np.ix_(lesioned_trials, lesion_cell_inds)] = lesion_mean
    sig[np.ix_(lesioned_trials, lesion_cell_inds)] = lesion_sd

    n_cells = izp.shape[0]

    psp_amp = 1e5
    psp_decay = 200
    resp_thresh = 5e6

    I_ext = np.zeros((n_cells, n_steps))
    I_net = np.zeros((n_cells, n_steps))

    v = np.zeros((n_cells, n_steps))
    u = np.zeros((n_cells, n_steps))
    g = np.zeros((n_cells, n_steps))
    spike = np.zeros((n_cells, n_steps))
    v[:, 0] = izp[:, 1]

    w = np.zeros((n_cells, n_cells))

    v_rec = np.zeros((n_cells, n_simulations, n_trials, n_steps))
    u_rec = np.zeros((n_cells, n_simulations, n_trials, n_steps))
    g_rec = np.zeros((n_cells, n_simulations, n_trials, n_steps))
    spike_rec = np.zeros((n_cells, n_simulations, n_trials, n_steps))
    w_rec = np.zeros((n_cells, n_cells, n_simulations, n_trials))
    w_vis_dms_A_rec = np.zeros((vis_dim, vis_dim, n_simulations, n_trials))
    w_vis_dms_B_rec = np.zeros((vis_dim, vis_dim, n_simulations, n_trials))
    w_vis_pm_A_rec = np.zeros((vis_dim, vis_dim, n_simulations, n_trials))
    w_vis_pm_B_rec = np.zeros((vis_dim, vis_dim, n_simulations, n_trials))

    for sim in range(n_simulations):
        print(f"Simulation {sim + 1}/{n_simulations}", flush=True)

        w_vis_dms_A = np.random.uniform(0.4, 0.6, (vis_dim, vis_dim))
        w_vis_dms_B = np.random.uniform(0.4, 0.6, (vis_dim, vis_dim))

        w[0, 2] = 0.04
        w[0, 3] = 0
        w[1, 2] = 0
        w[1, 3] = 0.04

        w[2, 4] = np.random.uniform(0.49, 0.51)
        w[2, 5] = np.random.uniform(0.49, 0.51)
        w[3, 4] = np.random.uniform(0.49, 0.51)
        w[3, 5] = np.random.uniform(0.49, 0.51)

        w[4, 6] = 0.04
        w[4, 7] = 0
        w[5, 6] = 0
        w[5, 7] = 0.04

        w_vis_pm_A = np.random.uniform(0.001, 0.01, (vis_dim, vis_dim)) * 0
        w_vis_pm_B = np.random.uniform(0.001, 0.01, (vis_dim, vis_dim)) * 0

        w[2, 6] = np.random.uniform(0.001, 0.01) * 0
        w[2, 7] = np.random.uniform(0.001, 0.01) * 0
        w[3, 6] = np.random.uniform(0.001, 0.01) * 0
        w[3, 7] = np.random.uniform(0.001, 0.01) * 0

        w[0, 1] = -0.2
        w[1, 0] = -0.2

        w[2, 3] = -0.01 * 0
        w[3, 2] = -0.01 * 0

        w[4, 5] = -0.5
        w[5, 4] = -0.5

        for trl in range(n_trials - 1):
            print(f"Trial {trl}/{n_trials}", flush=True)

            I_ext.fill(0)
            I_net.fill(0)
            v.fill(0)
            u.fill(0)
            g.fill(0)
            spike.fill(0)

            v[:, 0] = izp[:, 1]

            x = ds["x"][trl]
            y = ds["y"][trl]
            cat[sim, trl] = ds["cat"][trl]

            xg, yg = np.meshgrid(
                np.arange(0, vis_dim, 1),
                np.arange(0, vis_dim, 1),
            )

            vis = vis_amp * np.exp(-(((xg - x)**2 + (yg - y)**2) /
                                     (2 * vis_sig**2)))

            vis_dms_act_A = np.dot(vis.flatten(), w_vis_dms_A.flatten())
            vis_dms_act_B = np.dot(vis.flatten(), w_vis_dms_B.flatten())

            I_ext[0, n_steps // 3:2 * n_steps // 3] = vis_dms_act_A
            I_ext[1, n_steps // 3:2 * n_steps // 3] = vis_dms_act_B

            vis_pm_act_A = np.dot(vis.flatten(), w_vis_pm_A.flatten())
            vis_pm_act_B = np.dot(vis.flatten(), w_vis_pm_B.flatten())

            I_ext[2, n_steps // 3:2 * n_steps // 3] = vis_pm_act_A
            I_ext[3, n_steps // 3:2 * n_steps // 3] = vis_pm_act_B

            for i in range(1, n_steps):
                dt = t[i] - t[i - 1]

                I_net[:, i - 1] = w.T @ g[:, i - 1] - np.diag(w) * g[:, i - 1]
                I_net[:, i - 1] += I_ext[:, i - 1]

                noise = np.random.normal(mu[trl], sig[trl])
                dvdt = (k * (v[:, i - 1] - vr) * (v[:, i - 1] - vt) -
                        u[:, i - 1] + I_net[:, i - 1] * noise) / C
                dudt = a * (b * (v[:, i - 1] - vr) - u[:, i - 1])
                dgdt = (-g[:, i - 1] + psp_amp * spike[:, i - 1]) / psp_decay

                v[:, i] = v[:, i - 1] + dvdt * dt
                u[:, i] = u[:, i - 1] + dudt * dt
                g[:, i] = g[:, i - 1] + dgdt * dt

                mask = v[:, i] < -100
                v[mask, i] = -100

                mask = v[:, i] >= vpeak
                v[mask, i - 1] = vpeak[mask]
                v[mask, i] = c[mask]
                u[mask, i] += d[mask]
                spike[mask, i] = 1

                if (g[6, i] - g[7, i]) > resp_thresh:
                    resp[sim, trl] = 1
                    rt[sim, trl] = i
                    break
                if (g[7, i] - g[6, i]) > resp_thresh:
                    resp[sim, trl] = 2
                    rt[sim, trl] = i
                    break

            if rt[sim, trl] == 0:
                rt[sim, trl] = i
                if g[6, :].sum() > g[7, :].sum():
                    resp[sim, trl] = 1
                elif g[7, :].sum() > g[6, :].sum():
                    resp[sim, trl] = 2
                else:
                    resp[sim, trl] = np.random.choice([1, 2])

            if cat[sim, trl] == resp[sim, trl]:
                r[sim, trl] = 1
            else:
                r[sim, trl] = 0

            rpe[sim, trl] = r[sim, trl] - p[sim, trl]
            p[sim, trl + 1] = p[sim, trl] + alpha_critic * rpe[sim, trl]

            dms_A = g[0, :].sum()
            dms_B = g[1, :].sum()

            for ii in range(vis_dim):
                for jj in range(vis_dim):
                    pre_activity = vis[ii, jj]

                    post_activity = dms_A
                    dw_1 = alpha_w_vis_dms * pre_activity * np.clip(
                        post_activity - nmda_thresh, 0, None) * np.clip(
                            rpe[sim, trl], 0, None) * (1 - w_vis_dms_A[ii, jj])
                    dw_2 = beta_w_vis_dms * pre_activity * np.clip(
                        post_activity - nmda_thresh, 0, None) * np.clip(
                            rpe[sim, trl], None, 0) * w_vis_dms_A[ii, jj]
                    dw_3 = -gamma_w_vis_dms * pre_activity * np.clip(
                        nmda_thresh - post_activity, 0, None) * w_vis_dms_A[ii,
                                                                            jj]
                    w_vis_dms_A[ii, jj] += dw_1 + dw_2 + dw_3
                    w_vis_dms_A[ii, jj] = np.clip(w_vis_dms_A[ii, jj], 0, 1)

                    post_activity = dms_B
                    dw_1 = alpha_w_vis_dms * pre_activity * np.clip(
                        post_activity - nmda_thresh, 0, None) * np.clip(
                            rpe[sim, trl], 0, None) * (1 - w_vis_dms_B[ii, jj])
                    dw_2 = beta_w_vis_dms * pre_activity * np.clip(
                        post_activity - nmda_thresh, 0, None) * np.clip(
                            rpe[sim, trl], None, 0) * w_vis_dms_B[ii, jj]
                    dw_3 = -gamma_w_vis_dms * pre_activity * np.clip(
                        nmda_thresh - post_activity, 0, None) * w_vis_dms_B[ii,
                                                                            jj]
                    w_vis_dms_B[ii, jj] += dw_1 + dw_2 + dw_3
                    w_vis_dms_B[ii, jj] = np.clip(w_vis_dms_B[ii, jj], 0, 1)

            synapses = np.array([(2, 4), (2, 5), (3, 4), (3, 5)])
            pre_indices = synapses[:, 0]
            post_indices = synapses[:, 1]

            pre_activity = g[pre_indices, :].sum(axis=1)
            post_activity = g[post_indices, :].sum(axis=1)

            dw_1 = alpha_w_premotor_dls * pre_activity * np.clip(
                post_activity - nmda_thresh, 0, None) * np.clip(
                    rpe[sim, trl], 0, None) * (1 - w[pre_indices, post_indices])
            dw_2 = beta_w_premotor_dls * pre_activity * np.clip(
                post_activity - nmda_thresh, 0, None) * np.clip(
                    rpe[sim, trl], None, 0) * w[pre_indices, post_indices]
            dw_3 = -gamma_w_premotor_dls * pre_activity * np.clip(
                nmda_thresh - post_activity, 0, None) * w[pre_indices,
                                                          post_indices]

            dw = dw_1 + dw_2 + dw_3
            w[pre_indices, post_indices] += dw
            w[pre_indices,
              post_indices] = np.clip(w[pre_indices, post_indices], 0, 1)

            pm_A = g[2, :].sum()
            pm_B = g[3, :].sum()

            for ii in range(vis_dim):
                for jj in range(vis_dim):
                    pre_activity = vis[ii, jj]

                    post_activity = pm_A
                    dw_1 = alpha_w_vis_premotor * pre_activity * np.clip(
                        post_activity - nmda_thresh, 0,
                        None) * (1 - w_vis_pm_A[ii, jj])
                    dw_2 = -beta_w_vis_premotor * pre_activity * np.clip(
                        nmda_thresh - post_activity, 0, None) * w_vis_pm_A[ii,
                                                                           jj]
                    w_vis_pm_A[ii, jj] += dw_1 + dw_2
                    w_vis_pm_A[ii, jj] = np.clip(w_vis_pm_A[ii, jj], 0, 1)

                    post_activity = pm_B
                    dw_1 = alpha_w_vis_premotor * pre_activity * np.clip(
                        post_activity - nmda_thresh, 0,
                        None) * (1 - w_vis_pm_B[ii, jj])
                    dw_2 = -beta_w_vis_premotor * pre_activity * np.clip(
                        nmda_thresh - post_activity, 0, None) * w_vis_pm_B[ii,
                                                                           jj]
                    w_vis_pm_B[ii, jj] += dw_1 + dw_2
                    w_vis_pm_B[ii, jj] = np.clip(w_vis_pm_B[ii, jj], 0, 1)

            synapses = np.array([(2, 6), (2, 7), (3, 6), (3, 7)])
            pre_indices = synapses[:, 0]
            post_indices = synapses[:, 1]

            pre_activity = g[pre_indices, :].sum(axis=1)
            post_activity = g[post_indices, :].sum(axis=1)

            dw_1 = alpha_w_premotor_motor * pre_activity * np.clip(
                post_activity - nmda_thresh, 0,
                None) * (1 - w[pre_indices, post_indices])
            dw_2 = -beta_w_premotor_motor * pre_activity * np.clip(
                nmda_thresh - post_activity, 0, None) * w[pre_indices,
                                                          post_indices]

            dw = dw_1 + dw_2
            w[pre_indices, post_indices] += dw
            w[pre_indices,
              post_indices] = np.clip(w[pre_indices, post_indices], 0, 1)

            v_rec[:, sim, trl, :] = v
            u_rec[:, sim, trl, :] = u
            g_rec[:, sim, trl, :] = g
            spike_rec[:, sim, trl, :] = spike
            w_rec[:, :, sim, trl] = w
            w_vis_dms_A_rec[:, :, sim, trl] = w_vis_dms_A
            w_vis_dms_B_rec[:, :, sim, trl] = w_vis_dms_B
            w_vis_pm_A_rec[:, :, sim, trl] = w_vis_pm_A
            w_vis_pm_B_rec[:, :, sim, trl] = w_vis_pm_B

    prefix = build_prefix(fig_label, output_dir)
    np.save(prefix.with_name(prefix.name + "_v.npy"), v_rec)
    np.save(prefix.with_name(prefix.name + "_g.npy"), g_rec)
    np.save(prefix.with_name(prefix.name + "_w.npy"), w_rec)
    np.save(prefix.with_name(prefix.name + "_rpe.npy"), rpe)
    np.save(prefix.with_name(prefix.name + "_p.npy"), p)
    np.save(prefix.with_name(prefix.name + "_r.npy"), r)
    np.save(prefix.with_name(prefix.name + "_resp.npy"), resp)
    np.save(prefix.with_name(prefix.name + "_cat.npy"), cat)
    np.save(prefix.with_name(prefix.name + "_rt.npy"), rt)
    np.save(prefix.with_name(prefix.name + "_w_vis_dms_A_rec.npy"),
            w_vis_dms_A_rec)
    np.save(prefix.with_name(prefix.name + "_w_vis_dms_B_rec.npy"),
            w_vis_dms_B_rec)
    np.save(prefix.with_name(prefix.name + "_w_vis_pm_A_rec.npy"),
            w_vis_pm_A_rec)
    np.save(prefix.with_name(prefix.name + "_w_vis_pm_B_rec.npy"),
            w_vis_pm_B_rec)

    ds.to_csv(prefix.with_name(prefix.name + "_ds.csv"), index=False)

    return v_rec, g_rec, w_rec, rpe, p, resp, cat, rt


def make_stim_cats(n_stimuli_per_category=2000,
                   save_plot=False,
                   figures_dir=None,
                   figure_stem="model_spiking_cat_90vs180_gadi_stimuli"):

    var = 100
    corr = 0.9
    sigma = np.sqrt(var)

    theta = 45 * np.pi / 180
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta), np.cos(theta)]])

    category_A_mean = [40, 60]
    category_B_mean = [60, 40]

    std_major = sigma * np.sqrt(1 + corr)
    std_minor = sigma * np.sqrt(1 - corr)

    def sample_within_ellipse(mean, n_samples):
        r = np.sqrt(np.random.uniform(0, 9, n_samples))
        angle = np.random.uniform(0, 2 * np.pi, n_samples)

        x = r * np.cos(angle)
        y = r * np.sin(angle)

        x_scaled = x * std_major
        y_scaled = y * std_minor

        points = np.dot(rotation_matrix, np.vstack([x_scaled, y_scaled]))

        points[0, :] += mean[0]
        points[1, :] += mean[1]

        return points.T

    stimuli_A = sample_within_ellipse(category_A_mean, n_stimuli_per_category)
    stimuli_B = sample_within_ellipse(category_B_mean, n_stimuli_per_category)

    labels_A = np.array([1] * n_stimuli_per_category)
    labels_B = np.array([2] * n_stimuli_per_category)

    stimuli = np.concatenate([stimuli_A, stimuli_B])
    labels = np.concatenate([labels_A, labels_B])

    ds = pd.DataFrame({"x": stimuli[:, 0], "y": stimuli[:, 1], "cat": labels})

    ds["xt"] = ds["x"] * 5 / 100
    ds["yt"] = (ds["y"] * 90 / 100) * np.pi / 180

    ds = ds.sample(frac=1).reset_index(drop=True)

    ds_90 = ds.copy()
    ds_90["x"] = ds_90["x"] - 50
    ds_90["y"] = ds_90["y"] - 50
    theta = 90 * np.pi / 180
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta), np.cos(theta)]])
    rotated_points = np.dot(rotation_matrix, ds_90[["x", "y"]].T).T
    ds_90["x"] = rotated_points[:, 0]
    ds_90["y"] = rotated_points[:, 1]
    ds_90["x"] = ds_90["x"] + 50
    ds_90["y"] = ds_90["y"] + 50

    ds_180 = ds.copy()
    ds_180["x"] = ds_180["x"] - 50
    ds_180["y"] = ds_180["y"] - 50
    theta = 180 * np.pi / 180
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta), np.cos(theta)]])
    rotated_points = np.dot(rotation_matrix, ds_180[["x", "y"]].T).T
    ds_180["x"] = rotated_points[:, 0]
    ds_180["y"] = rotated_points[:, 1]
    ds_180["x"] = ds_180["x"] + 50
    ds_180["y"] = ds_180["y"] + 50

    if save_plot:
        if figures_dir is None:
            raise ValueError("figures_dir must be provided when save_plot=True")
        fig, ax = plt.subplots(1, 3, squeeze=False, figsize=(12, 6))
        sns.scatterplot(data=ds, x="x", y="y", hue="cat", alpha=0.5, ax=ax[0, 0])
        sns.scatterplot(data=ds_90,
                        x="x",
                        y="y",
                        hue="cat",
                        alpha=0.5,
                        ax=ax[0, 1])
        sns.scatterplot(data=ds_180,
                        x="x",
                        y="y",
                        hue="cat",
                        alpha=0.5,
                        ax=ax[0, 2])
        plt.tight_layout()
        plt.savefig(figures_dir / f"{figure_stem}.png", dpi=150)
        plt.close(fig)

    return ds, ds_90, ds_180


def load_simulation(fig_label, output_dir):
    prefix = build_prefix(fig_label, output_dir)

    v_rec = np.load(prefix.with_name(prefix.name + "_v.npy"))
    g_rec = np.load(prefix.with_name(prefix.name + "_g.npy"))
    w_rec = np.load(prefix.with_name(prefix.name + "_w.npy"))
    rpe = np.load(prefix.with_name(prefix.name + "_rpe.npy"))
    p = np.load(prefix.with_name(prefix.name + "_p.npy"))
    r = np.load(prefix.with_name(prefix.name + "_r.npy"))
    resp = np.load(prefix.with_name(prefix.name + "_resp.npy"))
    cat = np.load(prefix.with_name(prefix.name + "_cat.npy"))
    rt = np.load(prefix.with_name(prefix.name + "_rt.npy"))
    w_vis_dms_A_rec = np.load(prefix.with_name(prefix.name + "_w_vis_dms_A_rec.npy"))
    w_vis_dms_B_rec = np.load(prefix.with_name(prefix.name + "_w_vis_dms_B_rec.npy"))
    w_vis_pm_A_rec = np.load(prefix.with_name(prefix.name + "_w_vis_pm_A_rec.npy"))
    w_vis_pm_B_rec = np.load(prefix.with_name(prefix.name + "_w_vis_pm_B_rec.npy"))
    ds = pd.read_csv(prefix.with_name(prefix.name + "_ds.csv"))

    return (v_rec, g_rec, w_rec, rpe, p, r, resp, cat, rt, w_vis_dms_A_rec,
            w_vis_dms_B_rec, w_vis_pm_A_rec, w_vis_pm_B_rec, ds)


def plot_simulation(fig_label, output_dir, figures_dir):
    res = load_simulation(fig_label, output_dir)

    v_rec = res[0]
    g_rec = res[1]
    w_rec = res[2]
    rpe = res[3]
    p = res[4]
    resp = res[6]
    cat = res[7]
    rt = res[8]
    w_vis_dms_A_rec = res[9]
    w_vis_dms_B_rec = res[10]
    w_vis_pm_A_rec = res[11]
    w_vis_pm_B_rec = res[12]

    n_trials = v_rec.shape[2]

    mean_g = g_rec.mean(axis=1)
    mean_rpe = rpe.mean(axis=0)
    mean_p = p.mean(axis=0)
    mean_accuracy = (resp == cat).mean(axis=0)

    pathway_A = [0, 2, 4, 6]
    pathway_B = [1, 3, 5, 7]

    pathway_A_names = ["DMS A", "Premotor A", "DLS A", "Motor A"]
    pathway_B_names = ["DMS B", "Premotor B", "DLS B", "Motor B"]

    fig, axes = plt.subplots(5, 4, figsize=(20, 15))
    trials = np.arange(n_trials)

    for idx, cell in enumerate(pathway_A):
        ax = axes[idx, 0]
        sns.heatmap(mean_g[cell], ax=ax, cbar=True, cmap="viridis")
        ax.set_title(f"A Pathway: {pathway_A_names[idx]}")
        ax.set_ylabel("Trial")
        ax.set_xlabel("Time (ms)")

    for idx, cell in enumerate(pathway_B):
        ax = axes[idx, 1]
        sns.heatmap(mean_g[cell], ax=ax, cbar=True, cmap="viridis")
        ax.set_title(f"B Pathway: {pathway_B_names[idx]}")
        ax.set_ylabel("Trial")
        ax.set_xlabel("Time (ms)")

    tt = np.arange(0, n_trials)

    axx = axes[0, 2]
    w_vis_dms_A_avg = np.mean(w_vis_dms_A_rec[:, :, -1, :], axis=(0, 1))
    w_vis_dms_B_avg = np.mean(w_vis_dms_B_rec[:, :, -1, :], axis=(0, 1))
    axx.plot(tt, w_vis_dms_A_avg, marker="o", label="w_vis_dms_A_avg")
    axx.plot(tt, w_vis_dms_B_avg, marker="o", label="w_vis_dms_B_avg")
    axx.set_ylim(-0.1, 1.5)
    axx.legend(loc="upper center", ncol=2)
    axx.set_xticks([])
    axx.set_title("Stage 1 subcortical")

    axx = axes[1, 2]
    axx.plot(tt, w_rec[2, 4, -1, :], marker="o", label="(PM A to DLS A)")
    axx.plot(tt, w_rec[2, 5, -1, :], marker="o", label="(PM A to DLS B)")
    axx.plot(tt, w_rec[3, 4, -1, :], marker="o", label="(PM B to DLS A)")
    axx.plot(tt, w_rec[3, 5, -1, :], marker="o", label="(PM B to DLS B)")
    axx.set_xticks([])
    axx.set_ylim(-0.1, 1.5)
    axx.legend(loc="upper center", ncol=2)
    axx.set_title("Stage 2 subcortical")

    axx = axes[2, 2]
    w_vis_pm_A_avg = np.mean(w_vis_pm_A_rec[:, :, -1, :], axis=(0, 1))
    w_vis_pm_B_avg = np.mean(w_vis_pm_B_rec[:, :, -1, :], axis=(0, 1))
    axx.plot(tt, w_vis_pm_A_avg, marker="o", label="w_vis_pm_A_avg")
    axx.plot(tt, w_vis_pm_B_avg, marker="o", label="w_vis_pm_B_avg")
    axx.set_ylim(-0.1, 1.5)
    axx.legend(loc="upper center", ncol=2)
    axx.set_xticks([])
    axx.set_title("Stage 1 cortical")

    axx = axes[3, 2]
    axx.plot(tt, w_rec[2, 6, -1, :], marker="o", label="(PM A to M1 A)")
    axx.plot(tt, w_rec[2, 7, -1, :], marker="o", label="(PM A to M1 B)")
    axx.plot(tt, w_rec[3, 6, -1, :], marker="o", label="(PM B to M1 A)")
    axx.plot(tt, w_rec[3, 7, -1, :], marker="o", label="(PM B to M1 B)")
    axx.set_xticks([])
    axx.set_ylim(-0.1, 1.5)
    axx.legend(loc="upper center", ncol=2)
    axx.set_title("Stage 2 cortical")

    ax = axes[0, 3]
    ax.plot(trials, mean_rpe, color="C0", marker="o", label="RPE")
    ax.plot(trials,
            mean_p,
            color="C1",
            linestyle="--",
            marker="o",
            label="Prediction (p)")
    ax.set_title("RPE and Prediction")
    ax.set_ylabel("RPE")
    ax.grid()

    ax_acc = axes[1, 3]
    ax_acc.plot(trials, mean_accuracy, color="C3", marker="o", label="Accuracy")
    ax_acc.set_title("Response Accuracy")
    ax_acc.set_ylabel("Accuracy (1 = Correct)")
    ax_acc.set_xlabel("Trial")
    ax_acc.grid()

    ax_rt = axes[2, 3]
    ax_rt.plot(trials,
               rt.mean(axis=0),
               color="C4",
               marker="o",
               label="Response Time")
    ax_rt.set_title("Response Time")
    ax_rt.set_ylabel("Time (ms)")
    ax_rt.set_xlabel("Trial")
    ax_rt.grid()

    axx = axes[-1][0]
    im0 = axx.imshow(w_vis_dms_A_rec[:, :, -1, -2], cmap="viridis")
    axx.invert_yaxis()
    plt.colorbar(im0, ax=axx, fraction=0.046, pad=0.04)
    axx.set_xticks([])
    axx.set_yticks([])
    axx.set_title("w_vis_dms_A (current)")

    axx = axes[-1][1]
    im1 = axx.imshow(w_vis_dms_B_rec[:, :, -1, -2], cmap="viridis")
    axx.invert_yaxis()
    plt.colorbar(im1, ax=axx, fraction=0.046, pad=0.04)
    axx.set_xticks([])
    axx.set_yticks([])
    axx.set_title("w_vis_dms_B (current)")

    axx = axes[-1][2]
    im2 = axx.imshow(w_vis_pm_A_rec[:, :, -1, -2], cmap="viridis")
    axx.invert_yaxis()
    plt.colorbar(im2, ax=axx, fraction=0.046, pad=0.04)
    axx.set_xticks([])
    axx.set_yticks([])
    axx.set_title("w_vis_pm_A (current)")

    axx = axes[-1][3]
    im3 = axx.imshow(w_vis_pm_B_rec[:, :, -1, -2], cmap="viridis")
    axx.invert_yaxis()
    plt.colorbar(im3, ax=axx, fraction=0.046, pad=0.04)
    axx.set_xticks([])
    axx.set_yticks([])
    axx.set_title("w_vis_pm_B (current)")

    plt.tight_layout()
    plt.savefig(figures_dir / f"{OUTPUT_PREFIX}_{fig_label}.png", dpi=150)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Gadi-safe single-process runner for model_spiking_cat_90vs180.py"
    )
    parser.add_argument("--rotations",
                        nargs="+",
                        type=int,
                        default=[90, 180],
                        choices=[0, 90, 180])
    parser.add_argument("--n-simulations", type=int, default=1)
    parser.add_argument("--n-trials", type=int, default=600)
    parser.add_argument("--probe-trial-onsets", nargs="+", type=int, default=[600])
    parser.add_argument("--n-probe-trials", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot-stimuli", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    lesion_mean = 0.0
    lesion_sd = 0.0
    lesioned_trials = []
    lesion_cell_inds = []

    print(f"Writing outputs to: {OUTPUT_DIR}", flush=True)
    print(f"Writing figures to: {FIGURES_DIR}", flush=True)
    print(f"Rotations: {args.rotations}", flush=True)
    print(f"n_simulations={args.n_simulations}", flush=True)
    print(f"n_trials={args.n_trials}", flush=True)
    print(f"probe_trial_onsets={args.probe_trial_onsets}", flush=True)
    print(f"n_probe_trials={args.n_probe_trials}", flush=True)

    for rotation in args.rotations:
        fig_label = str(rotation)
        simulate(
            lesioned_trials,
            lesion_cell_inds,
            lesion_mean,
            lesion_sd,
            fig_label,
            rotation,
            args.probe_trial_onsets,
            args.n_probe_trials,
            args.n_trials,
            args.n_simulations,
            OUTPUT_DIR,
            save_stimulus_plot=args.plot_stimuli,
            figures_dir=FIGURES_DIR,
            seed=args.seed,
        )
        if args.plot:
            plot_simulation(fig_label, OUTPUT_DIR, FIGURES_DIR)


if __name__ == "__main__":
    main()
