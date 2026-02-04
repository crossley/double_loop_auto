import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import differential_evolution
from scipy.optimize import Bounds


def make_stim_cats(n_stimuli_per_category=2000):

    # Define covariance matrix parameters
    var = 100
    corr = 0.9
    sigma = np.sqrt(var)

    # Rotation matrix
    theta = 45 * np.pi / 180
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta), np.cos(theta)]])

    # Means for the two categories
    category_A_mean = [40, 60]
    category_B_mean = [60, 40]

    # Standard deviations along major and minor axes
    std_major = sigma * np.sqrt(1 + corr)
    std_minor = sigma * np.sqrt(1 - corr)

    def sample_within_ellipse(mean, n_samples):

        # Sample radius
        r = np.sqrt(np.random.uniform(
            0, 9, n_samples))  # 3 standard deviations, squared is 9

        # Sample angle
        angle = np.random.uniform(0, 2 * np.pi, n_samples)

        # Convert polar to Cartesian coordinates
        x = r * np.cos(angle)
        y = r * np.sin(angle)

        # Scale by standard deviations
        x_scaled = x * std_major
        y_scaled = y * std_minor

        # Apply rotation
        points = np.dot(rotation_matrix, np.vstack([x_scaled, y_scaled]))

        # Translate to mean
        points[0, :] += mean[0]
        points[1, :] += mean[1]

        return points.T

    # Generate stimuli
    stimuli_A = sample_within_ellipse(category_A_mean, n_stimuli_per_category)
    stimuli_B = sample_within_ellipse(category_B_mean, n_stimuli_per_category)

    # Define the labels
    labels_A = np.array([1] * n_stimuli_per_category)
    labels_B = np.array([2] * n_stimuli_per_category)

    # Concatenate the stimuli and labels
    stimuli = np.concatenate([stimuli_A, stimuli_B])
    labels = np.concatenate([labels_A, labels_B])

    # Put the stimuli and labels together into a dataframe
    ds = pd.DataFrame({"x": stimuli[:, 0], "y": stimuli[:, 1], "cat": labels})

    # Add a transformed version of the stimuli
    # let xt map x from [0, 100] to [0, 5]
    # let yt map y from [0, 100] to [0, 90]
    ds["xt"] = ds["x"] * 5 / 100
    ds["yt"] = (ds["y"] * 90 / 100) * np.pi / 180

    # shuffle rows of ds
    ds = ds.sample(frac=1).reset_index(drop=True)

    # create 90 degree rotation stim
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

    # create 180 degree rotation stim
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

    # fig, ax = plt.subplots(1, 3, squeeze=False, figsize=(12, 6))
    # sns.scatterplot(data=ds, x="x", y="y", hue="cat", alpha=0.5, ax=ax[0, 0])
    # sns.scatterplot(data=ds_90,
    #                 x="x",
    #                 y="y",
    #                 hue="cat",
    #                 alpha=0.5,
    #                 ax=ax[0, 1])
    # sns.scatterplot(data=ds_180,
    #                 x="x",
    #                 y="y",
    #                 hue="cat",
    #                 alpha=0.5,
    #                 ax=ax[0, 2])
    # plt.tight_layout()
    # plt.show()

    return ds, ds_90, ds_180


def objective_function(params, *args):
    """Objective function based on 90 degree vs 180 degree probes"""

    n_simulations, n_trials, probe_trial_onsets, n_probe_trials = args

    args = (n_simulations, n_trials, probe_trial_onsets, n_probe_trials, 90)
    acc_90 = simulate_model(params, *args)

    args = (n_simulations, n_trials, probe_trial_onsets, n_probe_trials, 180)
    acc_180 = simulate_model(params, *args)

    cost90 = probe_costs(acc_90, probe_trial_onsets, n_probe_trials)
    cost180 = probe_costs(acc_180, probe_trial_onsets, n_probe_trials)

    loss = constraint_loss(cost90, cost180)

    return loss


def probe_costs(acc_by_trial, probe_trial_onsets, n_probe_trials):
    """Converts accuracy by trial into probe costs at each probe onset."""
    costs = np.empty(len(probe_trial_onsets), dtype=float)

    for i, t in enumerate(probe_trial_onsets):
        pre_start = t - n_probe_trials
        pre_end = t
        probe_end = t + n_probe_trials

        pre_acc = acc_by_trial[pre_start:pre_end].mean()
        probe_acc = acc_by_trial[t:probe_end].mean()
        costs[i] = probe_acc - pre_acc

    return costs


def constraint_loss(cost_90, cost_180):
    """
    Optimization goals:

    Goal 1:
        Cost at 90 should be greater than cost at 180 for the first probe.

    Goal 2:
        Cost at 90 should increase from the first probe to the second probe.

    Goal 3:
        Cost at 180 should remain stable from the first probe to the second probe.

    Goal 4:
        Cost at 90 should remain stable from the second probe to the third probe.

    Goal 5:
        Cost at 180 should increase from the second probe to the third probe.
    """
    # goal 1 (hinge): want cost_90[0] > cost_180[0]
    goal_1_violation = np.maximum(0.0, cost_180[0] - cost_90[0])

    # goal 2 (hinge): want cost_90[1] > cost_90[0]
    goal_2_violation = np.maximum(0.0, cost_90[0] - cost_90[1])

    # goal 3 (equality): want cost_180[1] == cost_180[0]
    goal_3_violation = (cost_180[1] - cost_180[0])**2

    # goal 4 (equality): want cost_90[2] == cost_90[1]
    goal_4_violation = (cost_90[2] - cost_90[1])**2

    # goal 5 (hinge): want cost_180[2] > cost_180[1]
    goal_5_violation = np.maximum(0.0, cost_180[1] - cost_180[2])

    # Vectorized aggregation (no math change): sum all goal violations
    goal_violations = np.array([
        goal_1_violation,
        goal_2_violation,
        goal_3_violation,
        goal_4_violation,
        goal_5_violation,
    ],
                               dtype=float)

    return goal_violations.sum()


def simulate_model(params, *args):
    """Simulate the model with given parameters and return a performance metric."""

    n_simulations, n_trials, probe_trial_onsets, n_probe_trials, rotation = args

    alpha_critic = params[0]
    alpha_w_vis_dms = params[1]
    beta_w_vis_dms = params[2]
    gamma_w_vis_dms = params[3]
    alpha_w_premotor_dls = params[4]
    beta_w_premotor_dls = params[5]
    gamma_w_premotor_dls = params[6]
    alpha_w_vis_premotor = params[7]
    beta_w_vis_premotor = params[8]
    alpha_w_premotor_motor = params[9]
    beta_w_premotor_motor = params[10]
    resp_thresh = params[11]
    lat_inhib_dms = params[12]
    lat_inhib_dls = params[13]
    noise_dms = params[14]
    noise_dls = params[15]

    ds, ds_90, ds_180 = make_stim_cats(n_trials // 2)
    ds_0 = ds.sample(n=n_probe_trials, random_state=0).reset_index(drop=True)
    ds_90 = ds_90.sample(n=n_probe_trials,
                         random_state=0).reset_index(drop=True)
    ds_180 = ds_180.sample(n=n_probe_trials,
                           random_state=0).reset_index(drop=True)

    if rotation == 0:
        ds_probe = ds_0
    elif rotation == 90:
        ds_probe = ds_90
    elif rotation == 180:
        ds_probe = ds_180

    ds['phase'] = 'train'
    ds_probe['phase'] = 'probe'

    for onset in sorted(probe_trial_onsets, reverse=True):
        ds_top = ds.iloc[:onset, :].reset_index(drop=True)
        ds_bottom = ds.iloc[onset:, :].reset_index(drop=True)
        ds = pd.concat([ds_top, ds_probe, ds_bottom], ignore_index=True)

    n_trials = ds.shape[0]

    tau = 1
    T = 3000
    t = np.arange(0, T, tau)
    n_steps = t.shape[0]

    psp_amp = 1e5
    psp_decay = 200
    nmda_thresh = 0.0

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
        [50, -80, -25, 40, 0.01, -20, -55, 150, 1],  # dms A 0 (MSN)
        [50, -80, -25, 40, 0.01, -20, -55, 150, 1],  # dms B 1 (MSN)
        [100, -60, -40, 35, 0.03, -2, -50, 100, 0.7],  # premotor A 2
        [100, -60, -40, 35, 0.03, -2, -50, 100, 0.7],  # premotor B 3
        [50, -80, -25, 40, 0.01, -20, -55, 150, 1],  # dls A 4 (MSN)
        [50, -80, -25, 40, 0.01, -20, -55, 150, 1],  # dls B 5 (MSN)
        [100, -60, -40, 35, 0.03, -2, -50, 100, 0.7],  # motor A 6
        [100, -60, -40, 35, 0.03, -2, -50, 100, 0.7],  # motor B 7
    ])

    C, vr, vt, vpeak, a, b, c, d, k = izp.T

    mu = np.ones((n_trials, izp.shape[0]))
    sig = np.zeros((n_trials, izp.shape[0]))

    # noise in units
    sig[:, 0] = noise_dms
    sig[:, 1] = noise_dms
    sig[:, 4] = noise_dls
    sig[:, 5] = noise_dls

    n_cells = izp.shape[0]

    # input into cells from the periphery
    I_ext = np.zeros((n_cells, n_steps))

    # input into cells from other cells
    I_net = np.zeros((n_cells, n_steps))

    v = np.zeros((n_cells, n_steps))
    u = np.zeros((n_cells, n_steps))
    g = np.zeros((n_cells, n_steps))
    spike = np.zeros((n_cells, n_steps))
    v[:, 0] = izp[:, 1]

    w = np.zeros((n_cells, n_cells))

    for sim in range(n_simulations):

        print(f"Simulation {sim + 1}/{n_simulations}")

        # vis->dms: fully connected
        w_vis_dms_A = np.random.uniform(0.4, 0.6, (vis_dim, vis_dim))
        w_vis_dms_B = np.random.uniform(0.4, 0.6, (vis_dim, vis_dim))

        w[0, 2] = 0.04
        w[0, 3] = 0
        w[1, 2] = 0
        w[1, 3] = 0.04

        # premotor->dls: fully connected
        w[2, 4] = np.random.uniform(0.49, 0.51)
        w[2, 5] = np.random.uniform(0.49, 0.51)
        w[3, 4] = np.random.uniform(0.49, 0.51)
        w[3, 5] = np.random.uniform(0.49, 0.51)

        # dls->motor: one to one
        w[4, 6] = 0.04
        w[4, 7] = 0
        w[5, 6] = 0
        w[5, 7] = 0.04

        # vis->premotor: fully connected
        w_vis_pm_A = np.random.uniform(0.001, 0.01, (vis_dim, vis_dim)) * 0
        w_vis_pm_B = np.random.uniform(0.001, 0.01, (vis_dim, vis_dim)) * 0

        # premotor->motor: fully connected
        w[2, 6] = np.random.uniform(0.001, 0.01) * 0
        w[2, 7] = np.random.uniform(0.001, 0.01) * 0
        w[3, 6] = np.random.uniform(0.001, 0.01) * 0
        w[3, 7] = np.random.uniform(0.001, 0.01) * 0

        # lateral inhibition between DMS units
        w[0, 1] = lat_inhib_dms
        w[1, 0] = lat_inhib_dms

        # lateral inhibition between DLS units
        w[4, 5] = lat_inhib_dls
        w[5, 4] = lat_inhib_dls

        for trl in range(n_trials - 1):

            print(f"Trial {trl}/{n_trials}")

            # reset trial variables
            I_ext.fill(0)
            I_net.fill(0)
            v.fill(0)
            u.fill(0)
            g.fill(0)
            spike.fill(0)

            v[:, 0] = izp[:, 1]

            # trial info
            x = ds["x"][trl]
            y = ds["y"][trl]
            cat[sim, trl] = ds["cat"][trl]

            # visual input
            xg, yg = np.meshgrid(np.arange(0, vis_dim, 1),
                                 np.arange(0, vis_dim, 1))

            vis = vis_amp * np.exp(-(((xg - x)**2 + (yg - y)**2) /
                                     (2 * vis_sig**2)))

            # define external inputs (visual input to DMS layer)
            vis_dms_act_A = np.dot(vis.flatten(), w_vis_dms_A.flatten())
            vis_dms_act_B = np.dot(vis.flatten(), w_vis_dms_B.flatten())

            I_ext[0, n_steps // 3:2 * n_steps // 3] = vis_dms_act_A
            I_ext[1, n_steps // 3:2 * n_steps // 3] = vis_dms_act_B

            # define external inputs (visual input to PM layer)
            vis_pm_act_A = np.dot(vis.flatten(), w_vis_pm_A.flatten())
            vis_pm_act_B = np.dot(vis.flatten(), w_vis_pm_B.flatten())

            I_ext[2, n_steps // 3:2 * n_steps // 3] = vis_pm_act_A
            I_ext[3, n_steps // 3:2 * n_steps // 3] = vis_pm_act_B

            for i in range(1, n_steps):

                dt = t[i] - t[i - 1]

                # Compute net input using matrix multiplication and remove self-connections
                I_net[:, i - 1] = w.T @ g[:, i - 1] - np.diag(w) * g[:, i - 1]

                # Add external inputs
                I_net[:, i - 1] += I_ext[:, i - 1]

                # Euler's method
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

                # response
                if (g[6, i] - g[7, i]) > resp_thresh:
                    resp[sim, trl] = 1
                    rt[sim, trl] = i
                    break
                elif (g[7, i] - g[6, i]) > resp_thresh:
                    resp[sim, trl] = 2
                    rt[sim, trl] = i
                    break

            # pick a response if it hasn't happened already
            if rt[sim, trl] == 0:
                rt[sim, trl] = i
                if g[6, :].sum() > g[7, :].sum():
                    resp[sim, trl] = 1
                elif g[7, :].sum() > g[6, :].sum():
                    resp[sim, trl] = 2
                else:
                    resp[sim, trl] = np.random.choice([1, 2])

            # feedback
            if cat[sim, trl] == resp[sim, trl]:
                r[sim, trl] = 1
            else:
                r[sim, trl] = 0

            # reward prediction error
            rpe[sim, trl] = r[sim, trl] - p[sim, trl]
            p[sim, trl + 1] = p[sim, trl] + alpha_critic * rpe[sim, trl]

            # NOTE: 3-factor vis-dms
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

            # NOTE: 3-factor premotor-dls
            synapses = np.array([(2, 4), (2, 5), (3, 4), (3, 5)])

            # Extract presynaptic and postsynaptic indices
            pre_indices = synapses[:, 0]
            post_indices = synapses[:, 1]

            # Compute presynaptic and postsynaptic activity sums
            pre_activity = g[pre_indices, :].sum(axis=1)
            post_activity = g[post_indices, :].sum(axis=1)

            # Vectorized weight update components
            dw_1 = alpha_w_premotor_dls * pre_activity * np.clip(
                post_activity - nmda_thresh, 0, None) * np.clip(
                    rpe[sim, trl], 0,
                    None) * (1 - w[pre_indices, post_indices])
            dw_2 = beta_w_premotor_dls * pre_activity * np.clip(
                post_activity - nmda_thresh, 0, None) * np.clip(
                    rpe[sim, trl], None, 0) * w[pre_indices, post_indices]
            dw_3 = -gamma_w_premotor_dls * pre_activity * np.clip(
                nmda_thresh - post_activity, 0, None) * w[pre_indices,
                                                          post_indices]

            # Apply the total weight change
            dw = dw_1 + dw_2 + dw_3
            w[pre_indices, post_indices] += dw
            w[pre_indices,
              post_indices] = np.clip(w[pre_indices, post_indices], 0, 1)

            # NOTE: 2-factor vis-premotor
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

            # NOTE: 2-factor premotor-motor
            synapses = np.array([
                (2, 6),
                (2, 7),
                (3, 6),
                (3, 7)  # premotor->motor
            ])

            # Extract presynaptic and postsynaptic indices
            pre_indices = synapses[:, 0]
            post_indices = synapses[:, 1]

            # Compute presynaptic and postsynaptic activity sums
            pre_activity = g[pre_indices, :].sum(axis=1)
            post_activity = g[post_indices, :].sum(axis=1)

            # Vectorized weight update components
            dw_1 = alpha_w_premotor_motor * pre_activity * np.clip(
                post_activity - nmda_thresh, 0,
                None) * (1 - w[pre_indices, post_indices])
            dw_2 = -beta_w_premotor_motor * pre_activity * np.clip(
                nmda_thresh - post_activity, 0, None) * w[pre_indices,
                                                          post_indices]

            # Apply the total weight change
            dw = dw_1 + dw_2
            w[pre_indices, post_indices] += dw
            w[pre_indices,
              post_indices] = np.clip(w[pre_indices, post_indices], 0, 1)

    acc = np.array([cat == resp], dtype=np.float32)
    acc = (cat == resp).astype(np.float32).mean(axis=0)

    return acc


def optimize_model(args):
    """Optimize model parameters using differential evolution."""

    bounds_pairs = [
        (0, 1),  # alpha_critic
        (0, 1),  # alpha_w_vis_dms
        (0, 1),  # beta_w_vis_dms
        (0, 0),  # gamma_w_vis_dms
        (0, 1),  # alpha_w_premotor_dls
        (0, 1),  # beta_w_premotor_dls
        (0, 0),  # gamma_w_premotor_dls
        (0, 1),  # alpha_w_vis_premotor
        (0, 1),  # beta_w_vis_premotor
        (0, 1),  # alpha_w_premotor_motor
        (0, 1),  # beta_w_premotor_motor
        (0, 1e4),  # resp_thresh
        (0, 1),  # lat_inhib_dms
        (0, 1),  # lat_inhib_dls
        (0, 1),  # noise_dms
        (0, 1),  # noise_dls
    ]

    lb = [b[0] for b in bounds_pairs]
    ub = [b[1] for b in bounds_pairs]
    bounds = Bounds(lb, ub)

    # result = differential_evolution(
    #     objective_function,
    #     bounds,
    #     args=args,
    #     strategy="best1bin",
    #     maxiter=50,
    #     popsize=10,
    #     tol=1e-3,
    #     mutation=(0.5, 1.0),
    #     recombination=0.7,
    #     polish=True,
    #     disp=True,
    #     updating="deferred",
    #     workers=1,
    # )

    # coarse global search
    result = differential_evolution(
        objective_function,
        bounds,
        args=args,
        maxiter=20,
        popsize=6,
        polish=False,
        workers=-1,
    )

    # # local refinement
    # from scipy.optimize import minimize
    # refined = minimize(
    #     objective_function,
    #     result.x,
    #     args=args,
    #     method="Powell",
    #     bounds=bounds,
    # )


    return result.x, result.fun


if __name__ == "__main__":

    np.random.seed(1)

    # NOTE: args useful everywhere
    n_simulations = 1
    n_trials = 10
    probe_trial_onsets = [3, 6, 9]
    n_probe_trials = 2
    args = (n_simulations, n_trials, probe_trial_onsets, n_probe_trials)

    # NOTE: optimize model
    optimized_params, optimized_value = optimize_model(args)
    np.save("../output/optimized_params.npy", optimized_params)

    # NOTE: inspect results
    params = np.load("../output/optimized_params.npy")
    args = (n_simulations, n_trials, probe_trial_onsets, n_probe_trials, 90)
    acc_90 = simulate_model(params, *args)
    args = (n_simulations, n_trials, probe_trial_onsets, n_probe_trials, 180)
    acc_180 = simulate_model(params, *args)
    acc_90 = np.asarray(acc_90).ravel()
    acc_180 = np.asarray(acc_180).ravel()
    x_90 = np.arange(acc_90.shape[0])
    x_180 = np.arange(acc_180.shape[0])

    plt.figure()
    plt.plot(x_90, acc_90, label="90")
    plt.plot(x_180, acc_180, label="180")
    plt.xlabel("Trial")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.show()
