import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def softmax(x, x2, beta):
    return np.exp(beta * x) / (np.exp(beta * x) + np.exp(beta * x2))


def simulate_model(lesion_dms_mean,
                   lesion_dms_std,
                   lesion_dls_mean,
                   lesion_dls_std,
                   lesion_dls_2_mean,
                   lesion_dls_2_std,
                   lesion_scale,
                   trl_lesion):

    n_simulations = 500
    n_trials = 2000

    vis_dim = 2
    vis_sigma = 10
    vis = np.zeros((vis_dim, 1))

    dms_A = np.zeros((n_simulations, n_trials))
    dms_B = np.zeros((n_simulations, n_trials))

    dls_A = np.zeros((n_simulations, n_trials))
    dls_B = np.zeros((n_simulations, n_trials))

    dls_2_A = np.zeros((n_simulations, n_trials))
    dls_2_B = np.zeros((n_simulations, n_trials))

    resp = np.zeros((n_simulations, n_trials))
    r = np.zeros((n_simulations, n_trials))
    p = np.zeros((n_simulations, n_trials))
    rpe = np.zeros((n_simulations, n_trials))

    alpha_actor_pos_1 = 0.3
    alpha_actor_neg_1 = 0.3

    alpha_actor_pos_2 = 0.3
    alpha_actor_neg_2 = 0.3

    alpha_actor_pos_3 = 0.3
    alpha_actor_neg_3 = 0.3

    alpha_critic = 0.01

    vis_rec = []

    results_rec = {
        "simulations": [],
        "trial": [],
        "cat": [],
        "x": [],
        "resp": [],
    }

    for sim in range(n_simulations):

        print("Simulation: {}".format(sim))

        w_vis_dms_A = np.random.uniform(0.45, 0.55, (vis_dim, 1))
        w_vis_dms_B = np.random.uniform(0.45, 0.55, (vis_dim, 1))
        w_vis_dms_rec = []

        w_dms_dls = np.random.uniform(0.05, 0.15, (2, 2))
        w_dms_dls_rec = []

        w_dls_dls_2 = np.random.uniform(0.05, 0.15, (2, 2))
        w_dls_dls_2_rec = []

        for trl in range(n_trials - 1):

            # trial info
            x = np.random.choice([1, 2])
            cat = x

            # visual input
            if x == 1:
                vis[0, 0] = 1
                vis[1, 0] = 0
            else:
                vis[0, 0] = 0
                vis[1, 0] = 1

            vis_rec.append(vis)

            # NOTE: stage 1: dms
            dms_A[sim, trl] = np.dot(vis.flatten(), w_vis_dms_A.flatten())
            dms_B[sim, trl] = np.dot(vis.flatten(), w_vis_dms_B.flatten())

            # total = dms_A[sim, trl] + dms_B[sim, trl]
            # dms_A[sim, trl] /= total
            # dms_B[sim, trl] /= total

            if trl > trl_lesion:
                dms_A[sim, trl] *= lesion_scale
                dms_B[sim, trl] *= lesion_scale

                dms_A[sim, trl] += np.random.normal(lesion_dms_mean,
                                                    lesion_dms_std)
                dms_B[sim, trl] += np.random.normal(lesion_dms_mean,
                                                    lesion_dms_std)

            # total = dms_A[sim, trl] + dms_B[sim, trl]
            # dms_A[sim, trl] /= total
            # dms_B[sim, trl] /= total

            # probA = softmax(dms_A[sim, trl], dms_B[sim, trl], 5)
            probA = 1 if dms_A[sim, trl] > dms_B[sim, trl] else 0
            probB = 1 - probA
            if np.random.rand() < probA:
                dms_B[sim, trl] = 0
            else:
                dms_A[sim, trl] = 0

            # NOTE: stage 2: dls
            dls_A[sim, trl] = (w_dms_dls[0, 0] * dms_A[sim, trl] +
                               w_dms_dls[0, 1] * dms_B[sim, trl])
            dls_B[sim, trl] = (w_dms_dls[1, 0] * dms_A[sim, trl] +
                               w_dms_dls[1, 1] * dms_B[sim, trl])

            # total = dls_A[sim, trl] + dls_B[sim, trl]
            # dls_A[sim, trl] /= total
            # dls_B[sim, trl] /= total

            if trl > trl_lesion:
                dls_A[sim, trl] *= lesion_scale
                dls_B[sim, trl] *= lesion_scale

                dls_A[sim, trl] += np.random.normal(lesion_dls_mean,
                                                    lesion_dls_std)
                dls_B[sim, trl] += np.random.normal(lesion_dls_mean,
                                                    lesion_dls_std)

            # total = dls_A[sim, trl] + dls_B[sim, trl]
            # dls_A[sim, trl] /= total
            # dls_B[sim, trl] /= total

            # probA = softmax(dls_A[sim, trl], dls_B[sim, trl], 5)
            probA = 1 if dls_A[sim, trl] > dls_B[sim, trl] else 0
            probB = 1 - probA
            if np.random.rand() < probA:
                dls_B[sim, trl] = 0
            else:
                dls_A[sim, trl] = 0

            # stage 3: dls_2
            dls_2_A[sim, trl] = (w_dls_dls_2[0, 0] * dls_A[sim, trl] +
                                 w_dls_dls_2[0, 1] * dls_B[sim, trl])
            dls_2_B[sim, trl] = (w_dls_dls_2[1, 0] * dls_A[sim, trl] +
                                 w_dls_dls_2[1, 1] * dls_B[sim, trl])

            # total = dls_2_A[sim, trl] + dls_2_B[sim, trl]
            # dls_2_A[sim, trl] /= total
            # dls_2_B[sim, trl] /= total

            # simulate lesion by adding noise
            if trl > trl_lesion:
                dls_2_A[sim, trl] *= lesion_scale
                dls_2_B[sim, trl] *= lesion_scale

                dls_2_A[sim, trl] += np.random.normal(lesion_dls_2_mean,
                                                      lesion_dls_2_std)
                dls_2_B[sim, trl] += np.random.normal(lesion_dls_2_mean,
                                                      lesion_dls_2_std)

            # total = dls_2_A[sim, trl] + dls_2_B[sim, trl]
            # dls_2_A[sim, trl] /= total
            # dls_2_B[sim, trl] /= total

            probA = softmax(dls_2_A[sim, trl], dls_2_B[sim, trl], 5)
            probB = 1 - probA
            if np.random.rand() < probA:
                resp[sim, trl] = 1
                dls_2_B[sim, trl] = 0
            else:
                resp[sim, trl] = 2
                dls_2_A[sim, trl] = 0

            # feedback
            if cat == resp[sim, trl]:
                r[sim, trl] = 1
            else:
                r[sim, trl] = -1

            # learning
            rpe[sim, trl] = r[sim, trl] - p[sim, trl]
            p[sim, trl + 1] = p[sim, trl] + alpha_critic * rpe[sim, trl]

            # stage 1 weights
            for ii in range(vis_dim):
                if rpe[sim, trl] > 0:
                    w_vis_dms_A[ii, 0] = (w_vis_dms_A[ii, 0] +
                                          alpha_actor_pos_1 * rpe[sim, trl] *
                                          (1 - w_vis_dms_A[ii, 0]) *
                                          vis[ii, 0] * dms_A[sim, trl])
                    w_vis_dms_B[ii, 0] = (w_vis_dms_B[ii, 0] +
                                          alpha_actor_pos_1 * rpe[sim, trl] *
                                          (1 - w_vis_dms_B[ii, 0]) *
                                          vis[ii, 0] * dms_B[sim, trl])
                else:
                    w_vis_dms_A[ii, 0] = (
                        w_vis_dms_A[ii, 0] +
                        alpha_actor_neg_1 * rpe[sim, trl] *
                        w_vis_dms_A[ii, 0] * vis[ii, 0] * dms_A[sim, trl])
                    w_vis_dms_B[ii, 0] = (
                        w_vis_dms_B[ii, 0] +
                        alpha_actor_neg_1 * rpe[sim, trl] *
                        w_vis_dms_B[ii, 0] * vis[ii, 0] * dms_B[sim, trl])

            # stage 2 weights
            dms = np.array([dms_A[sim, trl], dms_B[sim, trl]])
            dls = np.array([dls_A[sim, trl], dls_B[sim, trl]])

            for ii in range(2):
                for jj in range(2):
                    if rpe[sim, trl] > 0:
                        w_dms_dls[ii, jj] = (
                            w_dms_dls[ii, jj] +
                            alpha_actor_pos_2 * rpe[sim, trl] *
                            (1 - w_dms_dls[ii, jj]) * dms[ii] * dls[jj])
                    else:
                        w_dms_dls[ii,
                                  jj] = (w_dms_dls[ii, jj] +
                                         alpha_actor_neg_2 * rpe[sim, trl] *
                                         w_dms_dls[ii, jj] * dms[ii] * dls[jj])

            # stage 3 weights
            dls = np.array([dls_A[sim, trl], dls_B[sim, trl]])
            dls_2 = np.array([dls_2_A[sim, trl], dls_2_B[sim, trl]])

            for ii in range(2):
                for jj in range(2):
                    if rpe[sim, trl] > 0:
                        w_dls_dls_2[ii, jj] = (
                            w_dls_dls_2[ii, jj] +
                            alpha_actor_pos_2 * rpe[sim, trl] *
                            (1 - w_dls_dls_2[ii, jj]) * dls[ii] * dls_2[jj])
                    else:
                        w_dls_dls_2[ii, jj] = (
                            w_dls_dls_2[ii, jj] +
                            alpha_actor_neg_2 * rpe[sim, trl] *
                            w_dls_dls_2[ii, jj] * dls[ii] * dls_2[jj])

            w_dms_dls_rec.append(w_dms_dls.copy())
            w_dls_dls_2_rec.append(w_dls_dls_2.copy())

            results_rec["simulations"].append(sim)
            results_rec["trial"].append(trl)
            results_rec["cat"].append(cat)
            results_rec["x"].append(x)
            results_rec["resp"].append(resp[sim, trl])

    w_dms_dls_rec = np.array(w_dms_dls_rec)
    w_dls_dls_2_rec = np.array(w_dls_dls_2_rec)

    dms_A = np.mean(dms_A, axis=0)
    dms_B = np.mean(dms_B, axis=0)
    dls_A = np.mean(dls_A, axis=0)
    dls_B = np.mean(dls_B, axis=0)
    dls_2_A = np.mean(dls_2_A, axis=0)
    dls_2_B = np.mean(dls_2_B, axis=0)
    resp = np.mean(resp, axis=0)
    r = np.mean(r, axis=0)
    p = np.mean(p, axis=0)
    rpe = np.mean(rpe, axis=0)

    trials = np.arange(0, n_trials, 1)

#    fig, ax = plt.subplots(2, 2, squeeze=False, figsize=(10, 5))
#    fig.suptitle("Lesion: DMS Mean: {} Std: {} DLS Mean: {} Std: {} DLS_2 Mean: {} Std: {} Trial: {}".format(
#        lesion_dms_mean, lesion_dms_std, lesion_dls_mean, lesion_dls_std, lesion_dls_2_mean, lesion_dls_2_std, trl_lesion))
#
#    ax[0, 0].plot(trials, r, label='r')
#    ax[0, 0].plot(trials, p, label='p')
#    ax[0, 0].plot(trials, rpe, label='rpe')
#    ax[0, 0].set_ylim(-1.1, 1.1)
#    ax[0, 0].legend()
#
#    ax[0, 1].plot(trials, dms_A, label='dms_A')
#    ax[0, 1].plot(trials, dms_B, label='dms_B')
#    ax[0, 1].plot(trials, dls_A, label='dls_A')
#    ax[0, 1].plot(trials, dls_B, label='dls_B')
#    ax[0, 1].plot(trials, dls_2_A, label='dls_2_A')
#    ax[0, 1].plot(trials, dls_2_B, label='dls_2_B')
#    ax[0, 1].legend()
#
#    for i in range(2):
#        for j in range(2):
#            ax[1, 0].plot(trials[:-1],
#                          w_dms_dls_rec[:, i, j],
#                          label='w_dms_dls[{}, {}]'.format(i, j))
#    ax[1, 0].set_ylim(-0.1, 1.1)
#    ax[1, 0].legend()
#
#    for i in range(2):
#        for j in range(2):
#            ax[1, 1].plot(trials[:-1],
#                          w_dls_dls_2_rec[:, i, j],
#                          label='w_dls_dls_2[{}, {}]'.format(i, j))
#    ax[1, 1].set_ylim(-0.1, 1.1)
#    ax[1, 1].legend()
#
#    [x.set_xlim(0, n_trials - 2) for x in ax.flat]
#    plt.tight_layout()
#    plt.show()

    d = pd.DataFrame(results_rec)

    return d


def inspect_lesion():


    rec = []

    lesion_sd = [0.1]
    lesion_scale = 0.99

    lesion_trl = [0, 1000]
    for tl in lesion_trl:
        for lsd in lesion_sd:
            d_dms = simulate_model(0.0, lsd, 0.0, 0.0, 0.0, 0.0, lesion_scale, tl)
            d_dls = simulate_model(0.0, 0.0, 0.0, lsd, 0.0, 0.0, lesion_scale, tl)
            d_dls_2 = simulate_model(0.0, 0.0, 0.0, 0.0, 0.0, lsd, lesion_scale, tl)

            d_dms["lesion_loc"] = "DMS"
            d_dls["lesion_loc"] = "DLS"
            d_dls_2["lesion_loc"] = "DLS_2"

            d_dms["lesion_sd"] = lsd
            d_dls["lesion_sd"] = lsd
            d_dls_2["lesion_sd"] = lsd

            d_dms["lesion_trial"] = tl
            d_dls["lesion_trial"] = tl
            d_dls_2["lesion_trial"] = tl

            d = pd.concat([d_dms, d_dls, d_dls_2])
            rec.append(d)

    d_0 = simulate_model(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, lesion_scale, trl_lesion=1e6)
    d_0["lesion_loc"] = "Control"
    d_0["lesion_sd"] = 0
    d_0["lesion_trial"] = -1

    d = pd.concat([d_0] + rec)

    d["Accuracy"] = d["cat"] == d["resp"]
    d["lesion_loc"] = d["lesion_loc"].astype("category")
    d["lesion_sd"] = d["lesion_sd"].astype("category")

    sns.set_palette("deep")
    fig, ax = plt.subplots(2, 3, squeeze=False, figsize=(12, 5))
    for i, ll in enumerate(["DMS", "DLS", "DLS_2"]):
        for j, tl in enumerate(lesion_trl):
            sns.lineplot(
                data=d[((d["lesion_loc"] == ll) | (d["lesion_loc"] == "Control"))
                       & ((d["lesion_trial"] == tl) | (d["lesion_trial"] == -1))],
                x="trial",
                y="Accuracy",
                hue="lesion_sd",
                ax=ax[j, i],
            )
            ax[j, i].axvline(tl, linestyle="--", color="black")
            ax[j, i].set_title("Lesion: {} Trial: {}".format(ll, tl))

    [sns.move_legend(x, "lower right") for x in ax.flat]
    plt.tight_layout()
    plt.savefig("../figures/model_discrim_lesion_simple_vis_triple.png")
    plt.close()

    d.to_csv("../output/model_discrim_lesion_simple_vis_triple.csv",
             index=False)


inspect_lesion()
