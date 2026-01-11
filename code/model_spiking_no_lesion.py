import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def simulate(lesioned_trials, lesion_cell_inds, lesion_mean, lesion_sd, fig_label):

    np.random.seed(0)

    tau = 1
    T = 300
    t = np.arange(0, T, tau)
    n_steps = t.shape[0]

    alpha_critic = 0.001

    nmda_thresh = 0.0

    alpha_w_vis_dms = 5e-15
    beta_w_vis_dms = 5e-15
    gamma_w_vis_dms = 0.0

    alpha_w_premotor_dls = 1e-15
    beta_w_premotor_dls = 1e-15
    gamma_w_premotor_dls = 0.0

    alpha_w_vis_premotor = 1e-16
    beta_w_vis_premotor = 1e-16

    alpha_w_premotor_motor = 3e-17
    beta_w_premotor_motor = 3e-17

    cat = np.zeros((n_simulations, n_trials))
    resp = np.zeros((n_simulations, n_trials))
    rt = np.zeros((n_simulations, n_trials))
    r = np.zeros((n_simulations, n_trials))
    p = np.ones((n_simulations, n_trials)) * 0.5
    rpe = np.zeros((n_simulations, n_trials))

    izp = np.array([
        [100, -60, -40, 35, 0.03, -2, -50, 100, 0.7],      # visual A 0
        [100, -60, -40, 35, 0.03, -2, -50, 100, 0.7],      # visual B 1
        [50, -80, -25, 40, 0.01, -20, -55, 150, 1],        # dms A 2 (MSN)
        [50, -80, -25, 40, 0.01, -20, -55, 150, 1],        # dms B 3 (MSN)
        [100, -60, -40, 35, 0.03, -2, -50, 100, 0.7],      # premotor A 4
        [100, -60, -40, 35, 0.03, -2, -50, 100, 0.7],      # premotor B 5
        [50, -80, -25, 40, 0.01, -20, -55, 150, 1],        # dls A 6 (MSN)
        [50, -80, -25, 40, 0.01, -20, -55, 150, 1],        # dls B 7 (MSN)
        [100, -60, -40, 35, 0.03, -2, -50, 100, 0.7],      # motor A 8
        [100, -60, -40, 35, 0.03, -2, -50, 100, 0.7],      # motor B 9
    ])

    C, vr, vt, vpeak, a, b, c, d, k = izp.T

    mu = np.ones((n_trials, izp.shape[0]))
    sig = np.zeros((n_trials, izp.shape[0]))

    # noise in motor units
    sig[:, 8] = 0
    sig[:, 9] = 0

    # lesion
    mu[np.ix_(lesioned_trials, lesion_cell_inds)] = lesion_mean
    sig[np.ix_(lesioned_trials, lesion_cell_inds)] = lesion_sd

    n_cells = izp.shape[0]

    vis_amp = 150
    psp_amp = 1e5
    psp_decay = 100
    resp_thresh = 1e4

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

    # record keeping arrays
    v_rec = np.zeros((n_cells, n_simulations, n_trials, n_steps))
    u_rec = np.zeros((n_cells, n_simulations, n_trials, n_steps))
    g_rec = np.zeros((n_cells, n_simulations, n_trials, n_steps))
    spike_rec = np.zeros((n_cells, n_simulations, n_trials, n_steps))
    w_rec = np.zeros((n_cells, n_cells, n_simulations, n_trials))

    for sim in range(n_simulations):

        print(f"Simulation {sim + 1}/{n_simulations}")

        # vis->dms: fully connected
        w[0, 2] = np.random.uniform(0.4, 0.6)
        w[0, 3] = np.random.uniform(0.4, 0.6)
        w[1, 2] = np.random.uniform(0.4, 0.6)
        w[1, 3] = np.random.uniform(0.4, 0.6)

        # dms->premotor: one to one
        w[2, 4] = 0.04
        w[2, 5] = 0
        w[3, 4] = 0
        w[3, 5] = 0.04

        # premotor->dls: fully connected
        w[4, 6] = np.random.uniform(0.4, 0.6)
        w[4, 7] = np.random.uniform(0.4, 0.6)
        w[5, 6] = np.random.uniform(0.4, 0.6)
        w[5, 7] = np.random.uniform(0.4, 0.6)

        # dls->motor: one to one
        w[6, 8] = 0.04
        w[6, 9] = 0
        w[7, 8] = 0
        w[7, 9] = 0.04

        # vis->premotor: fully connected
        w[0, 4] = np.random.uniform(0.001, 0.01)
        w[0, 5] = np.random.uniform(0.001, 0.01)
        w[1, 4] = np.random.uniform(0.001, 0.01)
        w[1, 5] = np.random.uniform(0.001, 0.01)

        # premotor->motor: fully connected
        w[4, 8] = np.random.uniform(0.001, 0.01)
        w[4, 9] = np.random.uniform(0.001, 0.01)
        w[5, 8] = np.random.uniform(0.001, 0.01)
        w[5, 9] = np.random.uniform(0.001, 0.01)

        # lateral inhibition between DMS units
        w[2, 3] = -0.2
        w[3, 2] = -0.2

        # lateral inhibition between DLS units
        w[6, 7] = -0.2
        w[7, 6] = -0.2

        for trl in range(n_trials - 1):

            print(f"Trial {trl + 1}/{n_trials}")

            # reset trial variables
            I_ext.fill(0)
            I_net.fill(0)
            v.fill(0)
            u.fill(0)
            g.fill(0)
            spike.fill(0)

            v[:, 0] = izp[:, 1]

            # trial info
            x = np.random.choice([0, 1])
            cat[sim, trl] = x

            # define external inputs
            I_ext[x, n_steps // 3:2 * n_steps // 3] = vis_amp

            for i in range(1, n_steps):

                dt = t[i] - t[i - 1]

                # Compute net input using matrix multiplication and remove self-connections
                I_net[:, i - 1] = w.T @ g[:, i - 1] - np.diag(w) * g[:, i - 1]

                # Add external inputs
                I_net[:, i - 1] += I_ext[:, i - 1]

                # Euler's method
                noise = np.random.normal(mu[trl], sig[trl])
                dvdt = (k * (v[:, i - 1] - vr) * (v[:, i - 1] - vt) - u[:, i - 1] + I_net[:, i - 1] * noise) / C
                dudt = a * (b * (v[:, i - 1] - vr) - u[:, i - 1])
                dgdt = (-g[:, i - 1] + psp_amp * spike[:, i - 1]) / psp_decay

                v[:, i] = v[:, i - 1] + dvdt * dt
                u[:, i] = u[:, i - 1] + dudt * dt
                g[:, i] = g[:, i - 1] + dgdt * dt

                mask = v[:, i] >= vpeak
                v[mask, i - 1] = vpeak[mask]
                v[mask, i] = c[mask]
                u[mask, i] += d[mask]
                spike[mask, i] = 1

                # response
                if (g[8, i] - g[9, i]) > resp_thresh:
                    resp[sim, trl] = 0
                    rt[sim, trl] = i
                    break
                elif (g[9, i] - g[8, i]) > resp_thresh:
                    resp[sim, trl] = 1
                    rt[sim, trl] = i
                    break

            # pick a response if it hasn't happened already
            if rt[sim, trl] == 0:
                rt[sim, trl] = i
                if g[8, i] != g[9, i]:
                    resp[sim, trl] = np.argmax(g[8:10, i])
                else:
                    resp[sim, trl] = np.random.choice([0, 1])

            # feedback
            if cat[sim, trl] == resp[sim, trl]:
                r[sim, trl] = 1
            else:
                r[sim, trl] = 0

            # reward prediction error
            rpe[sim, trl] = r[sim, trl] - p[sim, trl]
            p[sim, trl + 1] = p[sim, trl] + alpha_critic * rpe[sim, trl]

            #  NOTE: working with chat: will create an empty lesioned_trials
            #  array and pass it through. using len() to make the condition
            #  false and allow for learning
            # TODO: TEMP HACK TO DEBUG SOME CRAP
            if len(lesioned_trials) > 0 and trl >= lesioned_trials[0] and trl <= lesioned_trials[-1]:
                pass
            else:

                # NOTE: 3-factor vis-dms
                synapses = np.array([
                    (0, 2),
                    (0, 3),
                    (1, 2),
                    (1, 3),  # vis->dms
                ])

                # Extract presynaptic and postsynaptic indices
                pre_indices = synapses[:, 0]
                post_indices = synapses[:, 1]

                # Compute presynaptic and postsynaptic activity sums
                pre_activity = g[pre_indices, :].sum(axis=1)
                post_activity = g[post_indices, :].sum(axis=1)

                # Vectorized weight update components
                dw_1 = alpha_w_vis_dms * pre_activity * np.clip(
                    post_activity - nmda_thresh, 0, None) * np.clip(
                        rpe[sim, trl], 0, None) * (1 - w[pre_indices, post_indices])
                dw_2 = beta_w_vis_dms * pre_activity * np.clip(
                    post_activity - nmda_thresh, 0, None) * np.clip(
                        rpe[sim, trl], None, 0) * w[pre_indices, post_indices]
                dw_3 = -gamma_w_vis_dms * pre_activity * np.clip(
                    nmda_thresh - post_activity, 0, None) * w[pre_indices,
                                                              post_indices]

                # Apply the total weight change
                dw = dw_1 + dw_2 + dw_3
                w[pre_indices, post_indices] += dw
                w[pre_indices, post_indices] = np.clip(w[pre_indices, post_indices], 0,
                                                       1)

                # NOTE: 3-factor premotor-dls
                synapses = np.array([
                    (4, 6),
                    (4, 7),
                    (5, 6),
                    (5, 7)  # premotor->dls
                ])

                # Extract presynaptic and postsynaptic indices
                pre_indices = synapses[:, 0]
                post_indices = synapses[:, 1]

                # Compute presynaptic and postsynaptic activity sums
                pre_activity = g[pre_indices, :].sum(axis=1)
                post_activity = g[post_indices, :].sum(axis=1)

                # Vectorized weight update components
                dw_1 = alpha_w_premotor_dls * pre_activity * np.clip(
                    post_activity - nmda_thresh, 0, None) * np.clip(
                        rpe[sim, trl], 0, None) * (1 - w[pre_indices, post_indices])
                dw_2 = beta_w_premotor_dls * pre_activity * np.clip(
                    post_activity - nmda_thresh, 0, None) * np.clip(
                        rpe[sim, trl], None, 0) * w[pre_indices, post_indices]
                dw_3 = -gamma_w_premotor_dls * pre_activity * np.clip(
                    nmda_thresh - post_activity, 0, None) * w[pre_indices,
                                                              post_indices]

                # Apply the total weight change
                dw = dw_1 + dw_2 + dw_3
                w[pre_indices, post_indices] += dw
                w[pre_indices, post_indices] = np.clip(w[pre_indices, post_indices], 0,
                                                       1)

                # NOTE: 2-factor vis-premotor
                synapses = np.array([
                    (0, 4),
                    (0, 5),
                    (1, 4),
                    (1, 5),  # vis->premotor
                ])

                # Extract presynaptic and postsynaptic indices
                pre_indices = synapses[:, 0]
                post_indices = synapses[:, 1]

                # Compute presynaptic and postsynaptic activity sums
                pre_activity = g[pre_indices, :].sum(axis=1)
                post_activity = g[post_indices, :].sum(axis=1)

                # Vectorized weight update components
                dw_1 = alpha_w_vis_premotor * pre_activity * np.clip(post_activity - nmda_thresh, 0, None) * (1 - w[pre_indices, post_indices])
                dw_2 = -beta_w_vis_premotor * pre_activity * np.clip(nmda_thresh - post_activity, 0, None) * w[pre_indices, post_indices]

                # Apply the total weight change
                dw = dw_1 + dw_2
                w[pre_indices, post_indices] += dw
                w[pre_indices, post_indices] = np.clip(w[pre_indices, post_indices], 0,
                                                       1)

                # NOTE: 2-factor premotor-motor 
                synapses = np.array([
                    (4, 8),
                    (4, 9),
                    (5, 8),
                    (5, 9)  # premotor->motor
                ])

                # Extract presynaptic and postsynaptic indices
                pre_indices = synapses[:, 0]
                post_indices = synapses[:, 1]

                # Compute presynaptic and postsynaptic activity sums
                pre_activity = g[pre_indices, :].sum(axis=1)
                post_activity = g[post_indices, :].sum(axis=1)

                # Vectorized weight update components
                dw_1 = alpha_w_premotor_motor * pre_activity * np.clip( post_activity - nmda_thresh, 0, None) * (1 - w[pre_indices, post_indices])
                dw_2 = -beta_w_premotor_motor * pre_activity * np.clip(nmda_thresh - post_activity, 0, None) * w[pre_indices, post_indices]


                # Apply the total weight change
                dw = dw_1 + dw_2
                w[pre_indices, post_indices] += dw
                w[pre_indices, post_indices] = np.clip(w[pre_indices, post_indices], 0,
                                                   1)

            v_rec[:, sim, trl, :] = v
            u_rec[:, sim, trl, :] = u
            g_rec[:, sim, trl, :] = g
            spike_rec[:, sim, trl, :] = spike
            w_rec[:, :, sim, trl] = w

      # NOTE: commenting out the variable saves
#     np.save('../output/model_spiking_' + fig_label + '_v.npy', v_rec)
#     np.save('../output/model_spiking_' + fig_label + '_g.npy', g_rec)
#     np.save('../output/model_spiking_' + fig_label + '_w.npy', w_rec)
#     np.save('../output/model_spiking_' + fig_label + '_rpe.npy', rpe)
#     np.save('../output/model_spiking_' + fig_label + '_p.npy', p)
#     np.save('../output/model_spiking_' + fig_label + '_resp.npy', resp)
#     np.save('../output/model_spiking_' + fig_label + '_cat.npy', cat)
#     np.save('../output/model_spiking_' + fig_label + '_rt.npy', rt)

    return v_rec, g_rec, w_rec, rpe, p, resp, cat, rt

# NOTE: moving this up here
# Set parameters 
n_simulations = 1
n_trials = 1000

# NOTE: changing lesion mean to 1 so that mu = 1 and therefore noise is * by 1
lesion_mean = 1.0
lesion_sd = 0.0

lesioned_trials = np.array([], dtype=int)   # or ignore if you remove lesions
lesion_cell_inds = np.array([], dtype=int)
fig_label = "play"

if __name__ == "__main__":
    lesioned_trials = np.array([], dtype=int)   # or ignore if you remove lesions
    lesion_cells = np.array([], dtype=int)
    v_rec, g_rec, w_rec, rpe, p, resp, cat, rt = simulate(
        lesioned_trials, lesion_cells, lesion_mean=1.0, lesion_sd=0.0, fig_label="debug"
    )

    # plots
    # average across simulations
    # mean_accuracy = (resp == cat).mean(axis=0)
    # mean_rpe = rpe_rec.mean(axis=0)
    # mean_g = g_rec.mean(axis=1)
    # mean_w = w_rec.mean(axis=2)

    # I want acc plots, w_rec plots for DLS and DRLs, conductances for DMS and DLS (heatmap)
    # VIS -> DMS (0 > 2, 1 > 3) and PM -> DMS (4 > 6, 5 > 7)

    # Compute averages over simulations
    mean_v = v_rec.mean(axis=1)
    mean_g = g_rec.mean(axis=1)
    mean_w = w_rec.mean(axis=2)
    mean_rpe = rpe.mean(axis=0)
    mean_p = p.mean(axis=0)
    mean_accuracy = (resp == cat).mean(axis=0)

    # Define pathways for A and B
    pathway_A = [0, 2, 4, 6, 8]  # visual A, DMS A, premotor A, DLS A, motor A
    pathway_B = [1, 3, 5, 7, 9]  # visual B, DMS B, premotor B, DLS B, motor B

    pathway_A_names = ['Visual A', 'DMS A', 'Premotor A', 'DLS A', 'Motor A']
    pathway_B_names = ['Visual B', 'DMS B', 'Premotor B', 'DLS B', 'Motor B']

    # Define synapse groups
    synapses_vis_dms = [(0, 2), (0, 3), (1, 2), (1, 3)]
    synapses_premotor_dls = [(4, 6), (4, 7), (5, 6), (5, 7)]
    synapses_vis_premotor = [(0, 4), (0, 5), (1, 4), (1, 5)]
    synapses_premotor_motor = [(4, 8), (4, 9), (5, 8), (5, 9)]

    # Create figure and grid of subplots
    fig, axes = plt.subplots(5, 4, figsize=(12, 8))
    trials = np.arange(n_trials)

    # Column 1: A pathway
    for idx, cell in enumerate(pathway_A):
        ax = axes[idx, 0]
        cell_name = pathway_A_names[idx]
        sns.heatmap(mean_g[cell], ax=ax, cbar=True, cmap='viridis')
        ax.set_title(f"A Pathway: {cell_name}")
        ax.set_ylabel("Trial")
        ax.set_xlabel("Time (ms)")

    # Column 2: B pathway
    for idx, cell in enumerate(pathway_B):
        ax = axes[idx, 1]
        cell_name = pathway_B_names[idx]
        sns.heatmap(mean_g[cell], ax=ax, cbar=True, cmap='viridis')
        ax.set_title(f"B Pathway: {cell_name}")
        ax.set_ylabel("Trial")
        ax.set_xlabel("Time (ms)")

    # Column 3: Synaptic weights evolution across trials
    synapse_groups = [(synapses_vis_dms, "vis->dms"),
                      (synapses_premotor_dls, "premotor->dls"),
                      (synapses_vis_premotor, "vis->premotor"),
                      (synapses_premotor_motor, "premotor->motor")]

    for idx, (synapses, title) in enumerate(synapse_groups):
        ax = axes[idx, 2]
        for pre, post in synapses:
            weight_evolution = mean_w[pre, post, :]
            ax.plot(trials, weight_evolution, label=f"w[{pre}->{post}]")
        ax.set_ylim(-0.1, 1.1)
        ax.set_title(f"Evolution of Weights: {title}")
        ax.set_xlabel("Trial")
        ax.set_ylabel("Weight Value")
        ax.legend()
        ax.grid()

    # Column 4: RPE, prediction, and accuracy
    # Plot RPE and Prediction
    ax = axes[0, 3]
    ax.plot(trials, mean_rpe, color='C0', label='RPE')
    ax.plot(trials, mean_p, color='C1', linestyle='--', label='Prediction (p)')
    ax.set_title("RPE and Prediction")
    ax.set_ylabel("RPE")
    ax.grid()

    # Plot Accuracy
    ax_acc = axes[1, 3]
    ax_acc.plot(trials, mean_accuracy, color='C3', marker='o', label='Accuracy')
    ax_acc.set_title("Response Accuracy")
    ax_acc.set_ylabel("Accuracy (1 = Correct)")
    ax_acc.set_xlabel("Trial")
    ax_acc.grid()

    # Plot response times
    ax_rt = axes[2, 3]
    ax_rt.plot(trials, rt.mean(axis=0), color='C4', marker='o', label='Response Time')
    ax_rt.set_title("Response Time")
    ax_rt.set_ylabel("Time (ms)")
    ax_rt.set_xlabel("Trial")
    ax_rt.grid()

    # Show plot
    plt.show()

# NOTE: don't need this for the moment, this comes up in the loop below
# trial_segments = [
#     np.arange(100, n_trials),
#     np.arange(500, n_trials),
#     np.arange(1000, n_trials),
# ]
# 
# lesion_cell_sets = [
#     np.array([2, 3]),  # DMS
#     np.array([6, 7])   # DLS
# ]

# NOTE: commenting out lesioned_trials loop
# for i, lesioned_trials in enumerate(trial_segments):
#     for j, lesion_cell_inds in enumerate(lesion_cell_sets):
#         fig_label = 'lesion_trials_' + str(lesioned_trials[0]) + '-' + str(lesioned_trials[-1]) + '_cells_' + ['DMS', 'DLS'][j]
#         simulate(lesioned_trials, lesion_cell_inds, lesion_mean, lesion_sd, fig_label)
#         plot_simulation(fig_label)

# NOTE: commenting out all plotting code for the moment
# def load_simulation(fig_label):
#     v_rec = np.load('../output/model_spiking_' + fig_label + '_v.npy')
#     g_rec = np.load('../output/model_spiking_' + fig_label + '_g.npy')
#     w_rec = np.load('../output/model_spiking_' + fig_label + '_w.npy')
#     rpe = np.load('../output/model_spiking_' + fig_label + '_rpe.npy')
#     p = np.load('../output/model_spiking_' + fig_label + '_p.npy')
#     resp = np.load('../output/model_spiking_' + fig_label + '_resp.npy')
#     cat = np.load('../output/model_spiking_' + fig_label + '_cat.npy')
#     rt = np.load('../output/model_spiking_' + fig_label + '_rt.npy')
# 
#     return v_rec, g_rec, w_rec, rpe, p, resp, cat, rt
 
# def plot_simulation(fig_label):
# 
#     v_rec, g_rec, w_rec, rpe, p, resp, cat, rt = load_simulation(fig_label)
# 
#     # Compute averages over simulations
#     mean_v = v_rec.mean(axis=1)
#     mean_g = g_rec.mean(axis=1)
#     mean_w = w_rec.mean(axis=2)
#     mean_rpe = rpe.mean(axis=0)
#     mean_p = p.mean(axis=0)
#     mean_accuracy = (resp == cat).mean(axis=0)
# 
#     # Define pathways for A and B
#     pathway_A = [0, 2, 4, 6, 8]  # visual A, DMS A, premotor A, DLS A, motor A
#     pathway_B = [1, 3, 5, 7, 9]  # visual B, DMS B, premotor B, DLS B, motor B
# 
#     pathway_A_names = ['Visual A', 'DMS A', 'Premotor A', 'DLS A', 'Motor A']
#     pathway_B_names = ['Visual B', 'DMS B', 'Premotor B', 'DLS B', 'Motor B']
# 
#     # Define synapse groups
#     synapses_vis_dms = [(0, 2), (0, 3), (1, 2), (1, 3)]
#     synapses_premotor_dls = [(4, 6), (4, 7), (5, 6), (5, 7)]
#     synapses_vis_premotor = [(0, 4), (0, 5), (1, 4), (1, 5)]
#     synapses_premotor_motor = [(4, 8), (4, 9), (5, 8), (5, 9)]
# 
#     # Create figure and grid of subplots
#     fig, axes = plt.subplots(5, 4, figsize=(20, 15))
#     trials = np.arange(n_trials)
# 
#     # Column 1: A pathway
#     for idx, cell in enumerate(pathway_A):
#         ax = axes[idx, 0]
#         cell_name = pathway_A_names[idx]
#         sns.heatmap(mean_g[cell], ax=ax, cbar=True, cmap='viridis')
#         ax.set_title(f"A Pathway: {cell_name}")
#         ax.set_ylabel("Trial")
#         ax.set_xlabel("Time (ms)")
# 
#     # Column 2: B pathway
#     for idx, cell in enumerate(pathway_B):
#         ax = axes[idx, 1]
#         cell_name = pathway_B_names[idx]
#         sns.heatmap(mean_g[cell], ax=ax, cbar=True, cmap='viridis')
#         ax.set_title(f"B Pathway: {cell_name}")
#         ax.set_ylabel("Trial")
#         ax.set_xlabel("Time (ms)")
# 
#     # Column 3: Synaptic weights evolution across trials
#     synapse_groups = [(synapses_vis_dms, "vis->dms"),
#                       (synapses_premotor_dls, "premotor->dls"),
#                       (synapses_vis_premotor, "vis->premotor"),
#                       (synapses_premotor_motor, "premotor->motor")]
# 
#     for idx, (synapses, title) in enumerate(synapse_groups):
#         ax = axes[idx, 2]
#         for pre, post in synapses:
#             weight_evolution = mean_w[pre, post, :]
#             ax.plot(trials, weight_evolution, label=f"w[{pre}->{post}]")
#         ax.set_ylim(-0.1, 1.1)
#         ax.set_title(f"Evolution of Weights: {title}")
#         ax.set_xlabel("Trial")
#         ax.set_ylabel("Weight Value")
#         ax.legend()
#         ax.grid()
# 
#     # Column 4: RPE, prediction, and accuracy
#     # Plot RPE and Prediction
#     ax = axes[0, 3]
#     ax.plot(trials, mean_rpe, color='C0', label='RPE')
#     ax.plot(trials, mean_p, color='C1', linestyle='--', label='Prediction (p)')
#     ax.set_title("RPE and Prediction")
#     ax.set_ylabel("RPE")
#     ax.grid()
# 
#     # Plot Accuracy
#     ax_acc = axes[1, 3]
#     ax_acc.plot(trials, mean_accuracy, color='C3', marker='o', label='Accuracy')
#     ax_acc.set_title("Response Accuracy")
#     ax_acc.set_ylabel("Accuracy (1 = Correct)")
#     ax_acc.set_xlabel("Trial")
#     ax_acc.grid()
# 
#     # Plot response times
#     ax_rt = axes[2, 3]
#     ax_rt.plot(trials, rt.mean(axis=0), color='C4', marker='o', label='Response Time')
#     ax_rt.set_title("Response Time")
#     ax_rt.set_ylabel("Time (ms)")
#     ax_rt.set_xlabel("Trial")
#     ax_rt.grid()
# 
#     # Adjust layout and show plot
#     plt.tight_layout()
#     plt.savefig('../figures/model_spiking_' + fig_label  + '.png')
#     plt.close()
# 
# 
# def plot_simulation_2(fig_label):
# 
#     trial_segments = [
#         np.arange(0, 8000),
#         np.arange(2000, 8000),
#         np.arange(4000, 8000),
#         np.arange(6000, 8000),
#     ]
# 
#     lesion_cell_sets = [
#         np.array([2, 3]),  # DMS
#         np.array([6, 7])   # DLS
#     ]
# 
#     d_rec = []
#     for i, lesioned_trials in enumerate(trial_segments):
#         for j, lesion_cell_inds in enumerate(lesion_cell_sets):
#             fig_label = 'lesion_trials_' + str(i) + '_cells_' + ['DMS', 'DLS'][j]
#             v_rec, g_rec, w_rec, rpe, p, resp, cat, rt = load_simulation(fig_label)
# 
#             d = pd.DataFrame({
#                 'lesioned_trials': i,
#                 'lesioned_cells': ['DMS', 'DLS'][j],
#                 'trial': np.arange(cat.mean(axis=0).shape[0]),
#                 'acc': cat.mean(axis=0) == resp.mean(axis=0),
#                 'rt': rt.mean(axis=0)})
#             d_rec.append(d)
# 
#     d = pd.concat(d_rec)
# 
#     fig, ax = plt.subplots(4, 2, squeeze=False, figsize=(10, 5))
#     for i, lt in enumerate(d.lesioned_trials.unique()):
#         for j, lc in enumerate(d.lesioned_cells.unique()):
#             dd = d[(d.lesioned_trials == lt) & (d.lesioned_cells == lc)]
#             sns.lineplot(data=dd, x='trial', y='acc', hue='lesioned_cells', ax=ax[i, j])
#             ax[i, j].set_xlabel('Trial')
#             ax[i, j].set_ylabel('Accuracy')
#             ax[i, j].set_title(f'Lesion Trials {lt} Cells {lc}')
#     plt.tight_layout()
#     plt.show()
