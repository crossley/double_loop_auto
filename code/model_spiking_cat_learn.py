import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def simulate(lesioned_trials, lesion_cell_inds, lesion_mean, lesion_sd, fig_label):

    np.random.seed(0)

    ds = make_stim_cats()

    tau = 1
    T = 3000
    t = np.arange(0, T, tau)
    n_steps = t.shape[0]

    alpha_critic = 0.001

    nmda_thresh = 0.0

    # NOTE have to change these
    alpha_w_vis_dms = 5e-15
    beta_w_vis_dms = 5e-15
    gamma_w_vis_dms = 0.0

    alpha_w_premotor_dls = 1e-15
    beta_w_premotor_dls = 1e-15
    gamma_w_premotor_dls = 0.0

    # NOTE have to change these also
    alpha_w_vis_premotor = 1e-16
    beta_w_vis_premotor = 1e-16

    alpha_w_premotor_motor = 3e-17
    beta_w_premotor_motor = 3e-17

    cat = np.zeros((n_simulations, n_trials))        # cat of stim shown
    resp = np.zeros((n_simulations, n_trials))       # network response
    rt = np.zeros((n_simulations, n_trials))         # reaction time
    r = np.zeros((n_simulations, n_trials))          # reward
    p = np.ones((n_simulations, n_trials)) * 0.5     # predicted reward (init @ 0.5) 
    rpe = np.zeros((n_simulations, n_trials))        # reward prediction error

    # create vis units
    vis_dim = 100
    vis_sigma = 10
    vis = np.zeros((vis_dim, vis_dim))

    izp = np.array([
        [50, -80, -25, 40, 0.01, -20, -55, 150, 1],        # 0 dms A 2 (MSN)
        [50, -80, -25, 40, 0.01, -20, -55, 150, 1],        # 1 dms B 3 (MSN)
        [100, -60, -40, 35, 0.03, -2, -50, 100, 0.7],      # 2 premotor A 4
        [100, -60, -40, 35, 0.03, -2, -50, 100, 0.7],      # 3 premotor B 5
        [50, -80, -25, 40, 0.01, -20, -55, 150, 1],        # 4 dls A 6 (MSN)
        [50, -80, -25, 40, 0.01, -20, -55, 150, 1],        # 5 dls B 7 (MSN)
        [100, -60, -40, 35, 0.03, -2, -50, 100, 0.7],      # 6 motor A 8
        [100, -60, -40, 35, 0.03, -2, -50, 100, 0.7],      # 7 motor B 9
    ])

    C, vr, vt, vpeak, a, b, c, d, k = izp.T
    
    mu = np.ones((n_trials, izp.shape[0]))
    sig = np.zeros((n_trials, izp.shape[0]))

    # noise in motor units
    sig[:, 6] = 0
    sig[:, 7] = 0

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

    v = np.zeros((n_cells, n_steps))       # membrane potential
    u = np.zeros((n_cells, n_steps))       # recovery 
    g = np.zeros((n_cells, n_steps))       # conductance
    spike = np.zeros((n_cells, n_steps))   # if spike crossed threshold
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
        w_vis_dms_A = np.random.uniform(0.4, 0.6, (vis_dim, vis_dim))
        w_vis_dms_B = np.random.uniform(0.4, 0.6, (vis_dim, vis_dim))

        # dms->premotor: one to one
        w[0, 2] = 0.04
        w[0, 2] = 0
        w[1, 3] = 0
        w[1, 3] = 0.04

        # premotor->dls: fully connected
        w[2, 4] = np.random.uniform(0.4, 0.6)
        w[2, 5] = np.random.uniform(0.4, 0.6)
        w[3, 4] = np.random.uniform(0.4, 0.6)
        w[3, 5] = np.random.uniform(0.4, 0.6)

        # dls->motor: one to one
        w[4, 6] = 0.04
        w[4, 7] = 0
        w[5, 6] = 0
        w[5, 7] = 0.04

        # vis->premotor: fully connected
        # NOTE DOUBLE CHECK THESE, IS THIS OKAY? 
        w_vis_pm_A = np.random.uniform(0.001, 0.01, (vis_dim, vis_dim))
        w_vis_pm_B = np.random.uniform(0.001, 0.01, (vis_dim, vis_dim))

        # w[0, 2] = np.random.uniform(0.001, 0.01)
        # w[0, 3] = np.random.uniform(0.001, 0.01)
        # w[1, 2] = np.random.uniform(0.001, 0.01)
        # w[1, 3] = np.random.uniform(0.001, 0.01)

        # premotor->motor: fully connected
        w[2, 6] = np.random.uniform(0.001, 0.01)
        w[2, 7] = np.random.uniform(0.001, 0.01)
        w[3, 6] = np.random.uniform(0.001, 0.01)
        w[3, 7] = np.random.uniform(0.001, 0.01)

        # lateral inhibition between DMS units
        w[0, 1] = -0.2
        w[1, 0] = -0.2

        # lateral inhibition between DLS units
        w[4, 5] = -0.2
        w[5, 4] = -0.2

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
            x = ds["x"][trl]
            y = ds["y"][trl]
            cat = ds["cat"][trl]

            # visual input
            xg, yg = np.meshgrid(np.arange(0, vis_dim, 1), np.arange(0, vis_dim, 1))

            vis = np.exp(-(((xg - x) ** 2 + (yg - y) ** 2) / (2 * vis_sigma**2)))
            vis_rec.append(vis)

            # NOTE why no index by sim and trl? 
            vis_act_A = np.dot(vis.flatten(), w_vis_dms_A.flatten())
            vis_act_B = np.dot(vis.flatten(), w_vis_dms_B.flatten())

            # # normalising - turning off for the moment
            # total = dms_A[sim, trl] + dms_B[sim, trl]
            # dms_A[sim, trl] /= total
            # dms_B[sim, trl] /= total

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
                if (g[6, i] - g[7, i]) > resp_thresh:
                    resp[sim, trl] = 0
                    rt[sim, trl] = i
                    break
                elif (g[7, i] - g[6, i]) > resp_thresh:
                    resp[sim, trl] = 1
                    rt[sim, trl] = i
                    break

            # pick a response if it hasn't happened already
            if rt[sim, trl] == 0:
                rt[sim, trl] = i
                if g[6, i] != g[7, i]:
                    resp[sim, trl] = np.argmax(g[6:9, i])
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

            # NOTE: 3-factor vis-dms
            # NOTE WEIRD FUKY CHANGES HERE
            
            for ii in range(vis_dim): 
                for jj in range(vis_dim): 

                    pre_activity = 
                    post_activity = g[0, :].sum(axis=1)
                    
                    dw_1_vis_dms_A = alpha_w_vis_dms * pre_activity * np.clip( post_activity - nmda_thresh, 0, None) * np.clip( rpe[sim, trl], 0, None) * (1 - w_vis_dms_A[vis_dim, vis_dim])
                    dw_2_vis_dms_A = beta_w_vis_dms * pre_activity * np.clip( post_activity - nmda_thresh, 0, None) * np.clip( rpe[sim, trl], None, 0) * w_vis_dms_A[vis_dim, vis_dim]
                    dw_3_vis_dms_A = -gamma_w_vis_dms * pre_activity * np.clip( nmda_thresh - post_activity, 0, None) * w_vis_dms_A[vis_dim, vis_dim]

                    # apply the total weight change
                    dw = dw_1_vis_dms_A + dw_2_vis_dms_A + dw_3_vis_dms_A
                    w_vis_dms_A[vis_dim, vis_dim] += dw
                    w_vis_dms_A[vis_dim, vis_dim] = np.clip(w_vis_dms_A[vis_dim, vis_dim], 0, 1)

                    pre_activity = # sum of all vis activity? 
                    post_activity = g[1, :].sum(axis=1)
                    
                    dw_1_vis_dms_B = alpha_w_vis_dms * pre_activity * np.clip( post_activity - nmda_thresh, 0, None) * np.clip( rpe[sim, trl], 0, None) * (1 - w_vis_dms_B[vis_dim, vis_dim])
                    dw_2_vis_dms_B = beta_w_vis_dms * pre_activity * np.clip( post_activity - nmda_thresh, 0, None) * np.clip( rpe[sim, trl], None, 0) * w_vis_dms_B[vis_dim, vis_dim]
                    dw_3_vis_dms_B = -gamma_w_vis_dms * pre_activity * np.clip( nmda_thresh - post_activity, 0, None) * w_vis_dms_B[vis_dim, vis_dim]

                    # apply the total weight change
                    dw = dw_1_vis_dms_B + dw_2_vis_dms_B + dw_3_vis_dms_B
                    w_vis_dms_B[vis_dim, vis_dim] += dw
                    w_vis_dms_B[vis_dim, vis_dim] = np.clip(w_vis_dms_B[vis_dim, vis_dim], 0, 1)

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
            dw_1 = alpha_w_premotor_dls * pre_activity * np.clip( post_activity - nmda_thresh, 0, None) * np.clip( rpe[sim, trl], 0, None) * (1 - w[pre_indices, post_indices])
            dw_2 = beta_w_premotor_dls * pre_activity * np.clip( post_activity - nmda_thresh, 0, None) * np.clip( rpe[sim, trl], None, 0) * w[pre_indices, post_indices]
            dw_3 = -gamma_w_premotor_dls * pre_activity * np.clip( nmda_thresh - post_activity, 0, None) * w[pre_indices, post_indices]

            # Apply the total weight change
            dw = dw_1 + dw_2 + dw_3
            w[pre_indices, post_indices] += dw
            w[pre_indices, post_indices] = np.clip(w[pre_indices, post_indices], 0,
                                                   1)

            # NOTE: 2-factor vis-premotor
            # NOTE WEIRD FUNKY SHIT HERE AGAIN BUT WIHTOUT GAMMA

            for ii in range(vis_dim):
                for jj in range(vis_dim):
                    
                    pre_acivity = 
                    post_activity = g[2, :].sum(axis=1)

                    # Vectorized weight update components
                    dw_1_vis_pm_A = alpha_w_vis_premotor * pre_activity * np.clip(post_activity - nmda_thresh, 0, None) * (1 - w_vis_pm_A[vis_dim, vis_dim])
                    dw_2_vis_pm_A = -beta_w_vis_premotor * pre_activity * np.clip(nmda_thresh - post_activity, 0, None) * w_vis_pm_A[vis_dim, vis_dim] 

                    # Apply the total weight change
                    dw = dw_1_vis_pm_A + dw_2_vis_pm_A
                    w_vis_pm_A[vis_dim, vis_dim] += dw
                    w_vis_pm_A[vis_dim, vis_dim] = np.clip(w_vis_pm_A[vis_dim, vis_dim], 0, 1)

                    pre_acivity = 
                    post_activity = g[3, :].sum(axis=1)

                    # Vectorized weight update components
                    dw_1_vis_pm_B = alpha_w_vis_premotor * pre_activity * np.clip(post_activity - nmda_thresh, 0, None) * (1 - w_vis_pm_A[vis_dim, vis_dim])
                    dw_2_vis_pm_B = -beta_w_vis_premotor * pre_activity * np.clip(nmda_thresh - post_activity, 0, None) * w_vis_pm_A[vis_dim, vis_dim]

                    # Apply the total weight change
                    dw = dw_1_vis_pm_B + dw_2_vis_pm_B
                    w_vis_pm_B[vis_dim, vis_dim] += dw
                    w_vis_pm_B[vis_dim, vis_dim] = np.clip(w_vis_pm_B[vis_dim, vis_dim], 0, 1)

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
            w[pre_indices, post_indices] = np.clip(w[pre_indices, post_indices], 0, 1)

            v_rec[:, sim, trl, :] = v
            u_rec[:, sim, trl, :] = u
            g_rec[:, sim, trl, :] = g
            spike_rec[:, sim, trl, :] = spike
            w_rec[:, :, sim, trl] = w

    np.save('../output/model_spiking_' + fig_label + '_v.npy', v_rec)
    np.save('../output/model_spiking_' + fig_label + '_g.npy', g_rec)
    np.save('../output/model_spiking_' + fig_label + '_w.npy', w_rec)
    np.save('../output/model_spiking_' + fig_label + '_rpe.npy', rpe)
    np.save('../output/model_spiking_' + fig_label + '_p.npy', p)
    np.save('../output/model_spiking_' + fig_label + '_resp.npy', resp)
    np.save('../output/model_spiking_' + fig_label + '_cat.npy', cat)
    np.save('../output/model_spiking_' + fig_label + '_rt.npy', rt)

    return v_rec, g_rec, w_rec, rpe, p, resp, cat, rt

def make_stim_cats():

    n_stimuli_per_category = 2000
    
    # Define covariance matrix parameters
    var = 100
    corr = 0.9
    sigma = np.sqrt(var)

    # Rotation matrix
    theta = 45 * np.pi / 180
    rotation_matrix = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )

    # Standard deviations along major and minor axes
    std_major = sigma * np.sqrt(1 + corr)
    std_minor = sigma * np.sqrt(1 - corr)

    def sample_within_ellipse(mean, n_samples):

        # Sample radius
        r = np.sqrt(
            np.random.uniform(0, 9, n_samples)
        )  # 3 standard deviations, squared is 9

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
    category_A_mean = [40, 60]
    category_B_mean = [60, 40]

    stimuli_A = sample_within_ellipse(category_A_mean, n_stimuli_per_category)
    stimuli_B = sample_within_ellipse(category_B_mean, n_stimuli_per_category)

    # Define the labels
    labels_A = np.array([0] * n_stimuli_per_category)
    labels_B = np.array([1] * n_stimuli_per_category)

    # Concatenate the stimuli and labels
    stimuli = np.concatenate([stimuli_A, stimuli_B])
    labels = np.concatenate([labels_A, labels_B])

    # Put the stimuli and labels together into a dataframe
    ds = pd.DataFrame({"x": stimuli[:, 0], "y": stimuli[:, 1], "cat": labels})

    # let xt map x from [0, 100] to [0, 5]
    # let yt map y from [0, 100] to [0, 90]
    ds["xt"] = ds["x"] * 5 / 100
    ds["yt"] = (ds["y"] * 90 / 100) * np.pi / 180

    ds = ds.sample(frac=1).reset_index(drop=True)

    return ds

def load_simulation(fig_label):
    v_rec = np.load('../output/model_spiking_' + fig_label + '_v.npy')
    g_rec = np.load('../output/model_spiking_' + fig_label + '_g.npy')
    w_rec = np.load('../output/model_spiking_' + fig_label + '_w.npy')
    rpe = np.load('../output/model_spiking_' + fig_label + '_rpe.npy')
    p = np.load('../output/model_spiking_' + fig_label + '_p.npy')
    resp = np.load('../output/model_spiking_' + fig_label + '_resp.npy')
    cat = np.load('../output/model_spiking_' + fig_label + '_cat.npy')
    rt = np.load('../output/model_spiking_' + fig_label + '_rt.npy')

    return v_rec, g_rec, w_rec, rpe, p, resp, cat, rt


def plot_simulation(fig_label):

    v_rec, g_rec, w_rec, rpe, p, resp, cat, rt = load_simulation(fig_label)

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
    fig, axes = plt.subplots(5, 4, figsize=(20, 15))
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

    # Adjust layout and show plot
    plt.tight_layout()
    plt.savefig('../figures/model_spiking_' + fig_label  + '.png')
    plt.close()


def plot_simulation_2(fig_label):

    trial_segments = [
        np.arange(0, 8000),
        np.arange(2000, 8000),
        np.arange(4000, 8000),
        np.arange(6000, 8000),
    ]

    lesion_cell_sets = [
        np.array([2, 3]),  # DMS
        np.array([6, 7])   # DLS
    ]

    d_rec = []
    for i, lesioned_trials in enumerate(trial_segments):
        for j, lesion_cell_inds in enumerate(lesion_cell_sets):
            fig_label = 'lesion_trials_' + str(i) + '_cells_' + ['DMS', 'DLS'][j]
            v_rec, g_rec, w_rec, rpe, p, resp, cat, rt = load_simulation(fig_label)

            d = pd.DataFrame({
                'lesioned_trials': i,
                'lesioned_cells': ['DMS', 'DLS'][j],
                'trial': np.arange(cat.mean(axis=0).shape[0]),
                'acc': cat.mean(axis=0) == resp.mean(axis=0),
                'rt': rt.mean(axis=0)})
            d_rec.append(d)

    d = pd.concat(d_rec)

    fig, ax = plt.subplots(4, 2, squeeze=False, figsize=(10, 5))
    for i, lt in enumerate(d.lesioned_trials.unique()):
        for j, lc in enumerate(d.lesioned_cells.unique()):
            dd = d[(d.lesioned_trials == lt) & (d.lesioned_cells == lc)]
            sns.lineplot(data=dd, x='trial', y='acc', hue='lesioned_cells', ax=ax[i, j])
            ax[i, j].set_xlabel('Trial')
            ax[i, j].set_ylabel('Accuracy')
            ax[i, j].set_title(f'Lesion Trials {lt} Cells {lc}')
    plt.tight_layout()
    plt.show()

n_simulations = 1
n_trials = 500

lesioned_trials = []
lesion_cell_inds = []
lesion_mean = 1 # equivalent to no lesion
lesion_sd = 0 # equivalent to no lesion
fig_label = 'cat_learn'
simulate(lesioned_trials, lesion_cell_inds, lesion_mean, lesion_sd, fig_label)
plot_simulation(fig_label)
