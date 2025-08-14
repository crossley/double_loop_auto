import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns


def softmax(z1, z2, z3, beta):
    z = np.array([z1, z2, z3])
    z = z - np.max(z)
    exp_z = np.exp(beta * z)
    sum_exp_z = np.sum(exp_z)
    softmax_probs = exp_z / sum_exp_z
    return softmax_probs


def simulate_model(
    lesion_dms_mean,
    lesion_dms_std,
    lesion_dls_mean,
    lesion_dls_std,
    n_trials,
    n_simulations,
    lesion_trial,
):

    vis_A = np.zeros((n_simulations, n_trials))
    vis_B = np.zeros((n_simulations, n_trials))
    vis_C = np.zeros((n_simulations, n_trials))

    dms_A = np.zeros((n_simulations, n_trials))
    dms_B = np.zeros((n_simulations, n_trials))
    dms_C = np.zeros((n_simulations, n_trials))

    dls_A = np.zeros((n_simulations, n_trials))
    dls_B = np.zeros((n_simulations, n_trials))
    dls_C = np.zeros((n_simulations, n_trials))

    resp = np.zeros((n_simulations, n_trials))

    r_A = np.zeros((n_simulations, n_trials))
    p_A = np.zeros((n_simulations, n_trials))
    rpe_A = np.zeros((n_simulations, n_trials))

    r_B = np.zeros((n_simulations, n_trials))
    p_B = np.zeros((n_simulations, n_trials))
    rpe_B = np.zeros((n_simulations, n_trials))

    r_C = np.zeros((n_simulations, n_trials))
    p_C = np.zeros((n_simulations, n_trials))
    rpe_C = np.zeros((n_simulations, n_trials))

    alpha_actor_pos_dms = 0.3
    alpha_actor_neg_dms = 0.3

    alpha_actor_pos_dls = 0.3
    alpha_actor_neg_dls = 0.3

    alpha_critic = 0.01

    results_rec = {
        "simulations": [],
        "trial": [],
        "dms_A": [],
        "dms_B": [],
        "dms_C": [],
        "dls_A": [],
        "dls_B": [],
        "dls_C": [],
        "resp": [],
        "r_A": [],
        "p_A": [],
        "rpe_A": [],
        "r_B": [],
        "p_B": [],
        "rpe_B": [],
        "r_C": [],
        "p_C": [],
        "rpe_C": [],
        "w_vis_dms": [],
        "w_vis_dls": [],
        "w_dms_dls": [],
    }

    for sim in range(n_simulations):

        w_vis_dms = np.random.uniform(0.1, 0.2, (1, 3))
        w_vis_dls = np.random.uniform(0.01, 0.05, (1, 3)) * 0.0  # no dls direct pathway
        w_dms_dls = np.random.uniform(0.1, 0.2, (3, 3))

        for trl in range(n_trials - 1):

            # visual input
            vis_A[sim, trl] = 1
            vis_B[sim, trl] = 1
            vis_C[sim, trl] = 1

            # vis projection to dms
            dms_A[sim, trl] = w_vis_dms[0, 0] * vis_A[sim, trl]
            dms_B[sim, trl] = w_vis_dms[0, 1] * vis_B[sim, trl]
            dms_C[sim, trl] = w_vis_dms[0, 2] * vis_C[sim, trl]

            # simulate lesion by adding noise
            if trl > lesion_trial:
                dms_A[sim, trl] = np.random.normal(dms_A[sim, trl], lesion_dms_std)
                dms_B[sim, trl] = np.random.normal(dms_B[sim, trl], lesion_dms_std)
                dms_C[sim, trl] = np.random.normal(dms_C[sim, trl], lesion_dms_std)

                dms_A[sim, trl] *= 0.8
                dms_B[sim, trl] *= 0.8
                dms_C[sim, trl] *= 0.8

            # lateral inhibition in dms
            probs_dms = softmax(dms_A[sim, trl], dms_B[sim, trl], dms_C[sim, trl], 2)
            z = np.random.rand()
            if z < probs_dms[0]:
                dms_B[sim, trl] = 0
                dms_C[sim, trl] = 0

            elif z < probs_dms[0] + probs_dms[1]:
                dms_A[sim, trl] = 0
                dms_C[sim, trl] = 0

            else:
                dms_A[sim, trl] = 0
                dms_B[sim, trl] = 0

            # vis projection to dls
            # dls_A[sim, trl] = w_vis_dls[0, 0] * vis_A[sim, trl]
            # dls_B[sim, trl] = w_vis_dls[0, 1] * vis_B[sim, trl]
            # dls_C[sim, trl] = w_vis_dls[0, 2] * vis_C[sim, trl]

            # dms projection to dls
            dls_A[sim, trl] += (
                w_dms_dls[0, 0] * dms_A[sim, trl]
                + w_dms_dls[1, 0] * dms_B[sim, trl]
                + w_dms_dls[2, 0] * dms_C[sim, trl]
            )
            dls_B[sim, trl] += (
                w_dms_dls[0, 1] * dms_A[sim, trl]
                + w_dms_dls[1, 1] * dms_B[sim, trl]
                + w_dms_dls[2, 1] * dms_C[sim, trl]
            )
            dls_C[sim, trl] += (
                w_dms_dls[0, 2] * dms_A[sim, trl]
                + w_dms_dls[1, 2] * dms_B[sim, trl]
                + w_dms_dls[2, 2] * dms_C[sim, trl]
            )

            # simulate lesion by adding noise
            if trl > lesion_trial:
                dls_A[sim, trl] = np.random.normal(dls_A[sim, trl], lesion_dls_std)
                dls_B[sim, trl] = np.random.normal(dls_B[sim, trl], lesion_dls_std)
                dls_C[sim, trl] = np.random.normal(dls_C[sim, trl], lesion_dls_std)

                dls_A[sim, trl] *= 0.8
                dls_B[sim, trl] *= 0.8
                dls_C[sim, trl] *= 0.8

            # lateral inhibition in dls
            probs_dls = softmax(dls_A[sim, trl], dls_B[sim, trl], dls_C[sim, trl], 2)
            z = np.random.rand()
            if z < probs_dls[0]:
                dls_B[sim, trl] = 0
                dls_C[sim, trl] = 0

            elif z < probs_dls[0] + probs_dls[1]:
                dls_A[sim, trl] = 0
                dls_C[sim, trl] = 0

            else:
                dls_A[sim, trl] = 0
                dls_B[sim, trl] = 0

            # response depends on dms and dls
            # probs_resp = softmax(np.mean([dms_A[sim, trl], dls_A[sim, trl]]),
            #                      np.mean([dms_B[sim, trl], dls_B[sim, trl]]),
            #                      np.mean([dms_C[sim, trl], dls_C[sim, trl]]), 5)

            # resposne depends only on dls
            probs_resp = softmax(dls_A[sim, trl], dls_B[sim, trl], dls_C[sim, trl], 5)

            z = np.random.rand()
            if z < probs_resp[0]:
                resp[sim, trl] = 1
            elif z < probs_resp[0] + probs_resp[1]:
                resp[sim, trl] = 2
            else:
                resp[sim, trl] = 3

            # feedback
            if resp[sim, trl] == 1:
                r_A[sim, trl] = 1
                rpe_A[sim, trl] = r_A[sim, trl] - p_A[sim, trl]
                p_A[sim, trl + 1] = p_A[sim, trl] + alpha_critic * rpe_A[sim, trl]
                p_B[sim, trl + 1] = p_B[sim, trl]
                p_C[sim, trl + 1] = p_C[sim, trl]
                rpe = rpe_A[sim, trl]

            elif resp[sim, trl] == 2:
                r_B[sim, trl] = 1
                rpe_B[sim, trl] = r_B[sim, trl] - p_B[sim, trl]
                p_B[sim, trl + 1] = p_B[sim, trl] + alpha_critic * rpe_B[sim, trl]
                p_A[sim, trl + 1] = p_A[sim, trl]
                p_C[sim, trl + 1] = p_C[sim, trl]
                rpe = rpe_B[sim, trl]

            elif resp[sim, trl] == 3:
                r_C[sim, trl] = 0
                rpe_C[sim, trl] = r_C[sim, trl] - p_C[sim, trl]
                p_C[sim, trl + 1] = p_C[sim, trl] + alpha_critic * rpe_C[sim, trl]
                p_A[sim, trl + 1] = p_A[sim, trl]
                p_B[sim, trl + 1] = p_B[sim, trl]
                rpe = rpe_C[sim, trl]

            # prep for weight update
            vis = np.array([vis_A[sim, trl], vis_B[sim, trl], vis_C[sim, trl]])
            dms = np.array([dms_A[sim, trl], dms_B[sim, trl], dms_C[sim, trl]])
            dls = np.array([dls_A[sim, trl], dls_B[sim, trl], dls_C[sim, trl]])

            # update vis-dms weights
            for ii in range(3):
                if rpe > 0:
                    delta_w = (
                        alpha_actor_pos_dms
                        * rpe
                        * (1 - w_vis_dms[0, ii])
                        * vis[ii]
                        * dms[ii]
                    )
                else:
                    delta_w = (
                        alpha_actor_neg_dms * rpe * w_vis_dms[0, ii] * vis[ii] * dms[ii]
                    )
                w_vis_dms[0, ii] += delta_w

            # # update vis-dls weights
            # for ii in range(3):
            #     if rpe > 0:
            #         delta_w = alpha_actor_pos_dls * rpe * ( 1 - w_vis_dls[0, ii]) * vis[ii] * dls[ii]
            #     else:
            #         delta_w = alpha_actor_neg_dls * rpe * w_vis_dls[ 0, ii] * vis[ii] * dls[ii]
            #     w_vis_dls[0, ii] += delta_w

            # update dms-dls weights
            for ii in range(3):
                for jj in range(3):
                    if rpe > 0:
                        delta_w = (
                            alpha_actor_pos_dls
                            * rpe
                            * (1 - w_dms_dls[ii, jj])
                            * dms[ii]
                            * dls[jj]
                        )
                    else:
                        delta_w = (
                            alpha_actor_neg_dls
                            * rpe
                            * w_dms_dls[ii, jj]
                            * dms[ii]
                            * dls[jj]
                        )
                    w_dms_dls[ii, jj] += delta_w

            results_rec["simulations"].append(sim)
            results_rec["trial"].append(trl)
            results_rec["dms_A"].append(dms_A[sim, trl])
            results_rec["dms_B"].append(dms_B[sim, trl])
            results_rec["dms_C"].append(dms_C[sim, trl])
            results_rec["dls_A"].append(dls_A[sim, trl])
            results_rec["dls_B"].append(dls_B[sim, trl])
            results_rec["dls_C"].append(dls_C[sim, trl])
            results_rec["resp"].append(resp[sim, trl])
            results_rec["r_A"].append(r_A[sim, trl])
            results_rec["p_A"].append(p_A[sim, trl])
            results_rec["rpe_A"].append(rpe_A[sim, trl])
            results_rec["r_B"].append(r_B[sim, trl])
            results_rec["p_B"].append(p_B[sim, trl])
            results_rec["rpe_B"].append(rpe_B[sim, trl])
            results_rec["r_C"].append(r_C[sim, trl])
            results_rec["p_C"].append(p_C[sim, trl])
            results_rec["rpe_C"].append(rpe_C[sim, trl])
            results_rec["w_vis_dms"].append(w_vis_dms.copy())
            results_rec["w_vis_dls"].append(w_vis_dls.copy())
            results_rec["w_dms_dls"].append(w_dms_dls.copy())

    return results_rec


def plot_results(res):

    d = pd.DataFrame(res)
    d = d.groupby(["trial"], observed=True).mean().reset_index()

    d_resp = pd.DataFrame(res)
    d_resp = d_resp[["simulations", "trial", "resp"]]
    d_resp = d_resp.groupby(["trial", "resp"]).size().reset_index(name="count")
    d_resp["total"] = d_resp.groupby("trial")["count"].transform("sum")
    d_resp["proportion"] = d_resp["count"] / d_resp["total"]
    d_resp = d_resp[["trial", "resp", "proportion"]]
    d_resp["resp"] = d_resp["resp"].astype("category")

    fig, ax = plt.subplots(6, 2, squeeze=False, figsize=(10, 5))
    plt.subplots_adjust(hspace=0.75)

    ax[0, 0].plot(d["trial"], d["dms_A"])
    ax[0, 0].plot(d["trial"], d["dms_B"])
    ax[0, 0].plot(d["trial"], d["dms_C"])
    # ax[0, 0].set_ylim([-0.1, 1.1])
    ax[0, 0].legend(
        ["dms_A", "dms_B", "dms_C"],
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 1.5),
    )

    ax[1, 0].plot(d["trial"], d["dls_A"])
    ax[1, 0].plot(d["trial"], d["dls_B"])
    ax[1, 0].plot(d["trial"], d["dls_C"])
    # ax[1, 0].set_ylim([-0.1, 1.1])
    ax[1, 0].legend(
        ["dls_A", "dls_B", "dls_C"],
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 1.5),
    )

    sns.lineplot(
        data=d_resp, x="trial", y="proportion", hue="resp", legend=True, ax=ax[2, 0]
    )
    sns.move_legend(ax[2, 0], "upper center", bbox_to_anchor=(0.5, 1.5), ncol=3)
    ax[2, 0].set_ylabel("")

    ax[3, 0].plot(d["trial"], d["r_A"])
    ax[3, 0].plot(d["trial"], d["p_A"])
    ax[3, 0].plot(d["trial"], d["rpe_A"])
    ax[3, 0].legend(
        ["r_A", "p_A", "rpe_A"], loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.5)
    )

    ax[4, 0].plot(d["trial"], d["r_B"])
    ax[4, 0].plot(d["trial"], d["p_B"])
    ax[4, 0].plot(d["trial"], d["rpe_B"])
    ax[4, 0].legend(
        ["r_B", "p_B", "rpe_B"], loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.5)
    )

    ax[5, 0].plot(d["trial"], d["r_C"])
    ax[5, 0].plot(d["trial"], d["p_C"])
    ax[5, 0].plot(d["trial"], d["rpe_C"])
    ax[5, 0].legend(
        ["r_C", "p_C", "rpe_C"], loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.5)
    )

    w_vis_dms = np.concatenate(d["w_vis_dms"], axis=0)
    ax[0, 1].plot(d["trial"], w_vis_dms[:, 0])
    ax[0, 1].plot(d["trial"], w_vis_dms[:, 1])
    ax[0, 1].plot(d["trial"], w_vis_dms[:, 2])
    ax[0, 1].legend(
        ["w_vis_dms_A", "w_vis_dms_B", "w_vis_dms_C"],
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 1.5),
    )

    w_vis_dls = np.concatenate(d["w_vis_dls"], axis=0)
    ax[1, 1].plot(d["trial"], w_vis_dls[:, 0])
    ax[1, 1].plot(d["trial"], w_vis_dls[:, 1])
    ax[1, 1].plot(d["trial"], w_vis_dls[:, 2])
    ax[1, 1].legend(
        ["w_vis_dls_A", "w_vis_dls_B", "w_vis_dls_C"],
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 1.5),
    )

    w_dms_dls = np.stack(d["w_dms_dls"])
    blues = mpl.colormaps["Blues"]
    oranges = mpl.colormaps["Oranges"]
    greens = mpl.colormaps["Greens"]
    ax[2, 1].plot(d["trial"], w_dms_dls[:, 0, 0], color=blues(0.3))
    ax[2, 1].plot(d["trial"], w_dms_dls[:, 0, 1], color=blues(0.6))
    ax[2, 1].plot(d["trial"], w_dms_dls[:, 0, 2], color=blues(0.9))
    ax[2, 1].plot(d["trial"], w_dms_dls[:, 1, 0], color=oranges(0.3))
    ax[2, 1].plot(d["trial"], w_dms_dls[:, 1, 1], color=oranges(0.6))
    ax[2, 1].plot(d["trial"], w_dms_dls[:, 1, 2], color=oranges(0.9))
    ax[2, 1].plot(d["trial"], w_dms_dls[:, 2, 0], color=greens(0.3))
    ax[2, 1].plot(d["trial"], w_dms_dls[:, 2, 1], color=greens(0.6))
    ax[2, 1].plot(d["trial"], w_dms_dls[:, 2, 2], color=greens(0.9))
    ax[2, 1].legend(
        [
            "w_dms_dls_00",
            "w_dms_dls_01",
            "w_dms_dls_02",
            "w_dms_dls_10",
            "w_dms_dls_11",
            "w_dms_dls_12",
            "w_dms_dls_20",
            "w_dms_dls_21",
            "w_dms_dls_22",
        ],
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.5),
    )

    ax[3, 1].axis("off")
    ax[4, 1].axis("off")
    ax[5, 1].axis("off")
    plt.show()


def inspect_lesion():

    n_trials = 800
    n_simulations = 500

    lesion_trials = [0, 400]

    lesion_sd = [0.05, 0.075, 0.1]
    lesion_sd = [x * 4 for x in lesion_sd]

    d_rec = []

    for lt in lesion_trials:
        for lsd in lesion_sd:

            res = simulate_model(0, 0, 0, 0, n_trials, n_simulations, n_trials)
            res_dms = simulate_model(0, lsd, 0, 0, n_trials, n_simulations, lt)
            res_dls = simulate_model(0, 0, 0, lsd, n_trials, n_simulations, lt)

             #plot_results(res)
             #plot_results(res_dms)
             #plot_results(res_dls)

            d = pd.DataFrame(res)
            d_dms = pd.DataFrame(res_dms)
            d_dls = pd.DataFrame(res_dls)

            d["lesion"] = "Control"
            d_dms["lesion"] = "DMS"
            d_dls["lesion"] = "DLS"

            d = d[["simulations", "trial", "resp"]]
            d = d.groupby(["trial", "resp"]).size().reset_index(name="count")
            d["total"] = d.groupby("trial")["count"].transform("sum")
            d["proportion"] = d["count"] / d["total"]
            d = d[["trial", "resp", "proportion"]]
            d["resp"] = d["resp"].astype("category")

            d_dms = d_dms[["simulations", "trial", "resp"]]
            d_dms = d_dms.groupby(["trial", "resp"]).size().reset_index(name="count")
            d_dms["total"] = d_dms.groupby("trial")["count"].transform("sum")
            d_dms["proportion"] = d_dms["count"] / d_dms["total"]
            d_dms = d_dms[["trial", "resp", "proportion"]]
            d_dms["resp"] = d_dms["resp"].astype("category")

            d_dls = d_dls[["simulations", "trial", "resp"]]
            d_dls = d_dls.groupby(["trial", "resp"]).size().reset_index(name="count")
            d_dls["total"] = d_dls.groupby("trial")["count"].transform("sum")
            d_dls["proportion"] = d_dls["count"] / d_dls["total"]
            d_dls = d_dls[["trial", "resp", "proportion"]]
            d_dls["resp"] = d_dls["resp"].astype("category")

            d["resp"] = pd.Categorical(d["resp"], categories=[1, 2, 3], ordered=False)
            d_dms["resp"] = pd.Categorical(
                d_dms["resp"], categories=[1, 2, 3], ordered=False
            )
            d_dls["resp"] = pd.Categorical(
                d_dls["resp"], categories=[1, 2, 3], ordered=False
            )

            d["resp"] = d["resp"].map({1: "Lever 1", 2: "Lever 2", 3: "Other"})
            d_dms["resp"] = d_dms["resp"].map({1: "Lever 1", 2: "Lever 2", 3: "Other"})
            d_dls["resp"] = d_dls["resp"].map({1: "Lever 1", 2: "Lever 2", 3: "Other"})

            d["lesion_loc"] = "Control"
            d_dms["lesion_loc"] = "DMS"
            d_dls["lesion_loc"] = "DLS"

            d["lesion_trial"] = lt
            d_dms["lesion_trial"] = lt
            d_dls["lesion_trial"] = lt

            d["lesion_sd"] = 0.0
            d_dms["lesion_sd"] = lsd
            d_dls["lesion_sd"] = lsd

            d_rec.append(pd.concat([d, d_dms, d_dls]))

    d = pd.concat(d_rec)
    d["lesion_sd"] = d["lesion_sd"].astype("category")

    sns.set_palette("deep")
    fig, ax = plt.subplots(3, 2, squeeze=False, figsize=(10, 5))

    for ii, lt in enumerate(d.lesion_trial.unique()):

        dd = d[(d.lesion_trial == lt) & (d.lesion_loc == "Control")].copy()
        dd['lesion_sd'] = dd['lesion_sd'].cat.remove_unused_categories()
        for jj in range(3):
            sns.lineplot(
                data=dd,
                x="trial",
                y="proportion",
                style="resp",
                legend=False,
                ax=ax[jj, ii],
            )

        dd = d[(d.lesion_trial == lt) & (d.lesion_loc == "DMS")].copy()
        dd['lesion_sd'] = dd['lesion_sd'].cat.remove_unused_categories()
        sns.lineplot(
            data=dd,
            x="trial",
            y="proportion",
            style="resp",
            hue="lesion_sd",
            legend=False,
            ax=ax[1, ii],
        )

        dd = d[(d.lesion_trial == lt) & (d.lesion_loc == "DLS")].copy()
        dd['lesion_sd'] = dd['lesion_sd'].cat.remove_unused_categories()
        sns.lineplot(
            data=dd,
            x="trial",
            y="proportion",
            style="resp",
            hue="lesion_sd",
            legend=False,
            ax=ax[2, ii],
        )

        ax[0, ii].set_ylabel("Control")
        ax[1, ii].set_ylabel("DMS Lesion")
        ax[2, ii].set_ylabel("DLS Lesion")

        [x.set_ylim([0, 0.6]) for x in ax[:, ii]]
        [x.axvline(lt, color="black", linestyle="--") for x in ax[:, ii]]

    plt.tight_layout()
    plt.savefig("../figures/model_inst_cond.png")
    plt.close()

    d.to_csv("../output/model_inst_cond.csv", index=False)
