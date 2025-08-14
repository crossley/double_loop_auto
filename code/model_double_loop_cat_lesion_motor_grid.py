import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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

    return ds


def softmax(x, x2, beta):
    return np.exp(beta * x) / (np.exp(beta * x) + np.exp(beta * x2))


def simulate_model(
    lesion_dms_mean, lesion_dms_std, lesion_dls_mean, lesion_dls_std, trl_lesion
):
    ds = make_stim_cats()

    n_simulations = 50
    n_trials = 2000

    vis_dim = 100
    vis_sigma = 10
    vis = np.zeros((vis_dim, vis_dim))

    premot_dim = 100
    premot_sigma = 10
    premot = np.zeros((premot_dim, premot_dim))

    dms_A = np.zeros((n_simulations, n_trials))
    dms_B = np.zeros((n_simulations, n_trials))

    dls_A = np.zeros((n_simulations, n_trials))
    dls_B = np.zeros((n_simulations, n_trials))

    resp = np.zeros((n_simulations, n_trials))
    r = np.zeros((n_simulations, n_trials))
    p = np.zeros((n_simulations, n_trials))
    rpe = np.zeros((n_simulations, n_trials))

    alpha_actor_pos_1 = 0.05
    alpha_actor_neg_1 = 0.05
    alpha_actor_pos_2 = 0.05
    alpha_actor_neg_2 = 0.05

    alpha_critic = 0.01

    vis_rec = np.zeros((n_simulations * n_trials, vis_dim, vis_dim))
    premot_rec = np.zeros((n_simulations * n_trials, premot_dim, premot_dim))

    results_rec = {
        "simulations": [],
        "trial": [],
        "cat": [],
        "x": [],
        "y": [],
        "resp": [],
    }

    xg_vis, yg_vis = np.meshgrid(np.arange(0, vis_dim, 1), np.arange(0, vis_dim, 1))
    xg_pm, yg_pm = np.meshgrid(np.arange(0, premot_dim, 1), np.arange(0, premot_dim, 1))

    for sim in range(n_simulations):

        print("Simulation: {}".format(sim))

        w_vis_dms_A = np.random.uniform(0.45, 0.55, (vis_dim, vis_dim))
        w_vis_dms_B = np.random.uniform(0.45, 0.55, (vis_dim, vis_dim))

        w_premot_dls_A = np.random.uniform(0.45, 0.55, (premot_dim, premot_dim))
        w_premot_dls_B = np.random.uniform(0.45, 0.55, (premot_dim, premot_dim))

        w_dls_dls = np.random.uniform(0.05, 0.15, (2, 2))
        w_dls_dls_rec = []

        for trl in range(n_trials - 1):

            # trial info
            x = ds["x"][trl]
            y = ds["y"][trl]
            cat = ds["cat"][trl]

            # visual input
            vis = np.exp(-(((xg_vis - x) ** 2 + (yg_vis - y) ** 2) / (2 * vis_sigma**2)))
            vis_rec[sim * n_trials + trl] = vis

            # stage 1: dms
            dms_A[sim, trl] = np.sum(vis * w_vis_dms_A)
            dms_B[sim, trl] = np.sum(vis * w_vis_dms_B)

            total = dms_A[sim, trl] + dms_B[sim, trl]
            dms_A[sim, trl] /= total
            dms_B[sim, trl] /= total

            # simulate lesion by adding noise
            if trl > trl_lesion:
                dms_A[sim, trl] *= 0.8
                dms_B[sim, trl] *= 0.8
                dms_A[sim, trl] += np.random.normal(lesion_dms_mean, lesion_dms_std)
                dms_B[sim, trl] += np.random.normal(lesion_dms_mean, lesion_dms_std)
                dms_A[sim, trl] = np.clip(dms_A[sim, trl], 0, 1)
                dms_B[sim, trl] = np.clip(dms_B[sim, trl], 0, 1)

            # total = dms_A[sim, trl] + dms_B[sim, trl]
            # dms_A[sim, trl] /= total
            # dms_B[sim, trl] /= total

            # print(dms_A[sim, trl], dms_B[sim, trl])

            # probA = softmax(dms_A[sim, trl], dms_B[sim, trl], 5)
            probA = 1 if dms_A[sim, trl] > dms_B[sim, trl] else 0
            probB = 1 - probA

            if np.random.rand() < probA:
                dms_B[sim, trl] = 0
            else:
                dms_A[sim, trl] = 0

            # premot input
            if dms_A[sim, trl] > dms_B[sim, trl]:
                x = 40
                y = 60
            else:
                x = 60
                y = 40

            premot = np.exp(-(((xg_pm - x) ** 2 + (yg_pm - y) ** 2) / (2 * premot_sigma**2)))
            premot_rec[sim * n_trials + trl] = premot

            # stage 2: dls
            dls_A[sim, trl] = np.sum(premot * w_premot_dls_A)
            dls_B[sim, trl] = np.sum(premot * w_premot_dls_B)

            total = dls_A[sim, trl] + dls_B[sim, trl]
            dls_A[sim, trl] /= total
            dls_B[sim, trl] /= total

            # simulate lesion by adding noise
            if trl > trl_lesion:
                dls_A[sim, trl] *= 0.8
                dls_B[sim, trl] *= 0.8
                dls_A[sim, trl] += np.random.normal(lesion_dls_mean, lesion_dls_std)
                dls_B[sim, trl] += np.random.normal(lesion_dls_mean, lesion_dls_std)
                dls_A[sim, trl] = np.clip(dls_A[sim, trl], 0, 1)
                dls_B[sim, trl] = np.clip(dls_B[sim, trl], 0, 1)

            # total = dls_A[sim, trl] + dls_B[sim, trl]
            # dls_A[sim, trl] /= total
            # dls_B[sim, trl] /= total

            probA = softmax(dls_A[sim, trl], dls_B[sim, trl], 5)
            probB = 1 - probA

            if np.random.rand() < probA:
                resp[sim, trl] = 1
                dls_B[sim, trl] = 0
            else:
                resp[sim, trl] = 2
                dls_A[sim, trl] = 0

            # feedback
            if cat == resp[sim, trl]:
                r[sim, trl] = 1
            else:
                r[sim, trl] = -1

            # learning
            rpe[sim, trl] = r[sim, trl] - p[sim, trl]
            p[sim, trl + 1] = p[sim, trl] + alpha_critic * rpe[sim, trl]

            pos_update_A = (1 - w_vis_dms_A) * vis * dms_A[sim, trl]
            neg_update_A = w_vis_dms_A * vis * dms_A[sim, trl]
            w_vis_dms_A += alpha_actor_pos_1 * rpe[sim, trl] * pos_update_A
            w_vis_dms_A += alpha_actor_neg_1 * rpe[sim, trl] * neg_update_A

            pos_update_B = (1 - w_vis_dms_B) * vis * dms_B[sim, trl]
            neg_update_B = w_vis_dms_B * vis * dms_B[sim, trl]
            w_vis_dms_B += alpha_actor_pos_1 * rpe[sim, trl] * pos_update_B
            w_vis_dms_B += alpha_actor_neg_1 * rpe[sim, trl] * neg_update_B

            pos_update_A = (1 - w_premot_dls_A) * premot * dls_A[sim, trl]
            neg_update_A = w_premot_dls_A * premot * dls_A[sim, trl]
            w_premot_dls_A += alpha_actor_pos_1 * rpe[sim, trl] * pos_update_A
            w_premot_dls_A += alpha_actor_neg_1 * rpe[sim, trl] * neg_update_A

            pos_update_B = (1 - w_premot_dls_B) * premot * dls_B[sim, trl]
            neg_update_B = w_premot_dls_B * premot * dls_B[sim, trl]
            w_premot_dls_B += alpha_actor_pos_1 * rpe[sim, trl] * pos_update_B
            w_premot_dls_B += alpha_actor_neg_1 * rpe[sim, trl] * neg_update_B

#             # plot every 100 trials
#             if trl % 100 == 0:
#                 fig, ax = plt.subplots(2, 3, squeeze=False, figsize=(12, 4))
#                 sns.heatmap(vis, cbar=False, ax=ax[0, 0])
#                 sns.heatmap(w_vis_dms_A, vmin=0, vmax=1, cbar=False, ax=ax[0, 1])
#                 sns.heatmap(w_vis_dms_B, vmin=0, vmax=1, cbar=False, ax=ax[0, 2])
#                 sns.heatmap(premot, cbar=False, ax=ax[1, 0])
#                 sns.heatmap(w_premot_dls_A, vmin=0, vmax=1, cbar=False, ax=ax[1, 1])
#                 sns.heatmap(w_premot_dls_B, vmin=0, vmax=1, cbar=False, ax=ax[1, 2])
#                 [x.invert_yaxis() for x in ax.flat]
#                 [x.set_xticks([]) for x in ax.flat]
#                 [x.set_yticks([]) for x in ax.flat]
#                 [x.set_aspect('equal') for x in ax.flat]
#                 fig.suptitle(f"Trial: {trl}, Lesion Trial: {trl_lesion}, Lesion DMS SDs: {lesion_dms_std}, Lesion DLS: {lesion_dls_std}")
#                 plt.tight_layout()
#                 plt.show()

            results_rec["simulations"].append(sim)
            results_rec["trial"].append(trl)
            results_rec["cat"].append(cat)
            results_rec["x"].append(x)
            results_rec["y"].append(y)
            results_rec["resp"].append(resp[sim, trl])

    d = pd.DataFrame(results_rec)

    return d


def inspect_lesion():

    lesion_sd = [0.3]
    tl_max = 1000
    rec = []

    for tl in [0, tl_max]:
        for lsd in lesion_sd:
            d_dms = simulate_model(0.0, lsd, 0.0, 0.0, trl_lesion=tl)
            d_dls = simulate_model(0.0, 0.0, 0.0, lsd, trl_lesion=tl)
            d_dms["lesion_loc"] = "DMS"
            d_dls["lesion_loc"] = "DLS"
            d_dms["lesion_sd"] = lsd
            d_dls["lesion_sd"] = lsd
            d_dms["lesion_trial"] = tl
            d_dls["lesion_trial"] = tl
            d = pd.concat([d_dms, d_dls])
            rec.append(d)

    d_0 = simulate_model(0.0, 0.0, 0.0, 0.0, trl_lesion=1e6)
    d_0["lesion_loc"] = "Control"
    d_0["lesion_sd"] = 0
    d_0["lesion_trial"] = -1

    d = pd.concat([d_0] + rec)

    d["Accuracy"] = d["cat"] == d["resp"]
    d["lesion_loc"] = d["lesion_loc"].astype("category")
    d["lesion_sd"] = d["lesion_sd"].astype("category")

    sns.set_palette("deep")
    fig, ax = plt.subplots(2, 2, squeeze=False, figsize=(12, 5))
    sns.lineplot(
        data=d[
            ((d["lesion_loc"] == "DMS") | (d["lesion_loc"] == "Control"))
            & ((d["lesion_trial"] == 0) | (d["lesion_trial"] == -1))
        ],
        x="trial",
        y="Accuracy",
        hue="lesion_sd",
        ax=ax[0, 0],
    )
    sns.lineplot(
        data=d[
            ((d["lesion_loc"] == "DLS") | (d["lesion_loc"] == "Control"))
            & ((d["lesion_trial"] == 0) | (d["lesion_trial"] == -1))
        ],
        x="trial",
        y="Accuracy",
        hue="lesion_sd",
        ax=ax[0, 1],
    )
    sns.lineplot(
        data=d[
            ((d["lesion_loc"] == "DMS") | (d["lesion_loc"] == "Control"))
            & ((d["lesion_trial"] == tl_max) | (d["lesion_trial"] == -1))
        ],
        x="trial",
        y="Accuracy",
        hue="lesion_sd",
        ax=ax[1, 0],
    )
    sns.lineplot(
        data=d[
            ((d["lesion_loc"] == "DLS") | (d["lesion_loc"] == "Control"))
            & ((d["lesion_trial"] == tl_max) | (d["lesion_trial"] == -1))
        ],
        x="trial",
        y="Accuracy",
        hue="lesion_sd",
        ax=ax[1, 1],
    )
    [x.axvline(0, linestyle="--", color="black") for x in [ax[0, 0], ax[0, 1]]]
    [x.axvline(tl_max, linestyle="--", color="black") for x in [ax[1, 0], ax[1, 1]]]
    [sns.move_legend(x, "lower right") for x in ax.flat]
    ax[0, 0].set_title("DMS")
    ax[0, 1].set_title("DLS")
    plt.tight_layout()
    plt.savefig("../figures/model_cat_learn_lesion_motor_grid.png")
    plt.close()

    d.to_csv("../output/model_cat_learn_lesion_motor_grid.csv", index=False)

inspect_lesion()
