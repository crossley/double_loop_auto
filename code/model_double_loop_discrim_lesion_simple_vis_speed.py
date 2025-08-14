import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def softmax(x, x2, beta):
    return np.exp(beta * x) / (np.exp(beta * x) + np.exp(beta * x2))


def simulate_model(lesion_dms_scale, lesion_dms_std, lesion_dls_scale,
                   lesion_dls_std, trl_lesion):

    n_simulations = 100
    n_trials = 1000

    cat = np.zeros((n_simulations, n_trials))
    resp = np.zeros((n_simulations, n_trials))
    r = np.zeros((n_simulations, n_trials))
    p = np.ones((n_simulations, n_trials)) * 0.5
    rpe = np.zeros((n_simulations, n_trials))

    vis = np.zeros((n_simulations, n_trials, 2))
    dms = np.zeros((n_simulations, n_trials, 2))
    premotor = np.zeros((n_simulations, n_trials, 2))
    dls = np.zeros((n_simulations, n_trials, 2))
    motor = np.zeros((n_simulations, n_trials, 2))

    alpha_vis_dms_pos = 0.05
    alpha_vis_dms_neg = 0.05

    alpha_pm_dls_pos = 0.05
    alpha_pm_dls_neg = 0.05

    alpha_vis_pm_pos = 0.000
    alpha_vis_pm_neg = 0.000

    alpha_pm_mot_pos = 0.000
    alpha_pm_mot_neg = 0.000

    alpha_critic = 0.0

    w_vis_dms_rec = np.zeros((n_simulations, n_trials, 2, 2))
    w_premotor_dls_rec = np.zeros((n_simulations, n_trials, 2, 2))
    w_vis_premotor_rec = np.zeros((n_simulations, n_trials, 2, 2))
    w_premotor_motor_rec = np.zeros((n_simulations, n_trials, 2, 2))

    for sim in range(n_simulations):

        print("Simulation: {}".format(sim))

        lb = 0.5
        ub = 0.5
        w_vis_dms = np.random.uniform(lb, ub, (2, 2))
        w_premotor_dls = np.random.uniform(lb, ub, (2, 2))

        w_dms_premotor = np.ones(2)
        w_dls_motor = np.ones(2)

        lb = 0.1 * 0
        ub = 0.1 * 0
        w_vis_premotor = np.random.uniform(lb, ub, (2, 2))
        w_premotor_motor = np.random.uniform(lb, ub, (2, 2))

        for trl in range(n_trials - 1):

            # NOTE: trial info
            x = np.random.choice([0, 1])
            cat[sim, trl] = x

            # NOTE: vis
            vis[sim, trl, x] = 1

            # NOTE: dms
            dms[sim, trl, :] = np.sum(vis[sim, trl, :] * w_vis_dms, axis=1)

            if trl > trl_lesion:
                dms[sim, trl, :] *= lesion_dms_scale
                dms[sim, trl, :] += np.random.normal(0.0, lesion_dms_std, 2)

            if np.random.rand() < softmax(dms[sim, trl, 0], dms[sim, trl, 1], 100):
                dms[sim, trl, 1] = 0
            else:
                dms[sim, trl, 0] = 0

            # NOTE: premotor
            dms_act = dms[sim, trl, :] * w_dms_premotor
            vis_act = np.sum(vis[sim, trl, :] * w_vis_premotor, axis=1) * 0.25

            if dms_act.max() > vis_act.max():
                premotor[sim, trl, :] = dms_act
            else:
                premotor[sim, trl, :] = vis_act

            # NOTE: dls
            dls[sim, trl, :] = np.sum(premotor[sim, trl, :] * w_premotor_dls, axis=1)

            if trl > trl_lesion:
                dls[sim, trl, :] *= lesion_dls_scale
                dls[sim, trl, :] += np.random.normal(0.0, lesion_dls_std, 2)

            if np.random.rand() < softmax(dls[sim, trl, 0], dls[sim, trl, 1], 100):
                dls[sim, trl, 1] = 0
            else:
                dls[sim, trl, 0] = 0

            # NOTE: motor
            dls_act = dls[sim, trl, :] * w_dls_motor
            premotor_act = np.sum(premotor[sim, trl, :] * w_premotor_motor, axis=1) * 0.25

            if dls_act.max() > premotor_act.max():
                motor[sim, trl, :] = dls_act
            else:
                motor[sim, trl, :] = premotor_act

            # NOTE: response
            probA = 1 if motor[sim, trl, 0] > motor[sim, trl, 1] else 0
            if np.random.rand() < probA:
                resp[sim, trl] = 0
            else:
                resp[sim, trl] = 1

            # NOTE: feedback
            if cat[sim, trl] == resp[sim, trl]:
                r[sim, trl] = 1
            else:
                r[sim, trl] = 0

            # NOTE: reward prediction error
            rpe[sim, trl] = r[sim, trl] - p[sim, trl]
            p[sim, trl + 1] = p[sim, trl] + alpha_critic * rpe[sim, trl]

            # NOTE: Three-factor update
            for i in range(2):
                for j in range(2):
                    if rpe[sim, trl] > 0:
                        w_vis_dms[i, j] += alpha_vis_dms_pos * vis[
                            sim, trl, i] * dms[sim, trl, j] * rpe[sim, trl] * (
                                1 - w_vis_dms[i, j])
                        w_premotor_dls[i, j] += alpha_pm_dls_pos * premotor[
                            sim, trl, i] * dls[sim, trl, j] * rpe[sim, trl] * (
                                1 - w_premotor_dls[i, j])
                    else:
                        w_vis_dms[
                            i,
                            j] += alpha_vis_dms_neg * vis[sim, trl, i] * dms[
                                sim, trl, j] * rpe[sim, trl] * w_vis_dms[i, j]
                        w_premotor_dls[i, j] += alpha_pm_dls_neg * premotor[
                            sim, trl,
                            i] * dls[sim, trl,
                                     j] * rpe[sim, trl] * w_premotor_dls[i, j]

            # NOTE: two-factor update
            for i in range(2):
                for j in range(2):
                    w_vis_premotor[i, j] += alpha_vis_pm_pos * vis[
                        sim, trl, i] * premotor[sim, trl,
                                                j] * (1 - w_vis_premotor[i, j])
                    w_premotor_motor[i, j] += alpha_pm_mot_pos * premotor[
                        sim, trl, i] * motor[sim, trl,
                                             j] * (1 - w_premotor_motor[i, j])

            # clip all weight [0, 1]
            w_vis_dms = np.clip(w_vis_dms, 0, 1)
            w_premotor_dls = np.clip(w_premotor_dls, 0, 1)
            w_vis_premotor = np.clip(w_vis_premotor, 0, 1)
            w_premotor_motor = np.clip(w_premotor_motor, 0, 1)

            w_vis_dms_rec[sim, trl] = w_vis_dms.copy()
            w_premotor_dls_rec[sim, trl] = w_premotor_dls.copy()
            w_vis_premotor_rec[sim, trl] = w_vis_premotor.copy()
            w_premotor_motor_rec[sim, trl] = w_premotor_motor.copy()

    results = {
        "vis": vis,
        "dms": dms,
        "premotor": premotor,
        "dls": dls,
        "motor": motor,
        "cat": cat,
        "resp": resp,
        "Accuracy": resp == cat,
        "reward": r,
        "prediction": p,
        "reward_prediction_error": rpe,
        "w_vis_dms_rec": w_vis_dms_rec,
        "w_premotor_dls_rec": w_premotor_dls_rec,
        "w_vis_premotor_rec": w_vis_premotor_rec,
        "w_premotor_motor_rec": w_premotor_motor_rec
    }

    return results


def plot_results():

    fig, ax = plt.subplots(5, 3, squeeze=False, figsize=(18, 12))
    simulations = {
        "Control": d_ctl_0,
        "DMS Lesion": d_dms_0,
        "DLS Lesion": d_dls_0
    }
    for col, (sim_name, results) in enumerate(simulations.items()):
        vis = results["vis"]
        dms = results["dms"]
        premotor = results["premotor"]
        dls = results["dls"]
        motor = results["motor"]
        r = results["reward"]
        p = results["prediction"]
        rpe = results["reward_prediction_error"]
        w_vis_dms_rec = results["w_vis_dms_rec"]
        w_premotor_dls_rec = results["w_premotor_dls_rec"]
        w_vis_premotor_rec = results["w_vis_premotor_rec"]
        w_premotor_motor_rec = results["w_premotor_motor_rec"]
        resp = results["resp"]
        cat = results["cat"]
        ax[0, 0].plot(np.mean(resp == cat, axis=0)[:-1], label="Accuracy")
        ax[0, 1].plot(np.mean(resp == cat, axis=0)[:-1], label="Accuracy")
        ax[0, 2].plot(np.mean(resp == cat, axis=0)[:-1], label="Accuracy")
        ax[0, col].set_ylim(0.5, 1.0)
        for i in range(2):
            for j in range(2):
                ax[1, col].plot(w_vis_dms_rec.mean(0)[:-1, i, j])
                ax[2, col].plot(w_premotor_dls_rec.mean(0)[:-1, i, j])
                ax[3, col].plot(w_vis_premotor_rec.mean(0)[:-1, i, j])
                ax[4, col].plot(w_premotor_motor_rec.mean(0)[:-1, i, j])
                ax[1, col].set_title("Weight: Visual-DMS")
                ax[2, col].set_title("Weight: Premotor-DLS")
                ax[3, col].set_title("Weight: Visual-Premotor")
                ax[4, col].set_title("Weight: Premotor-Motor")
                ax[1, col].set_ylim(-0.1, 1.1)
                ax[2, col].set_ylim(-0.1, 1.1)
                ax[3, col].set_ylim(-0.1, 1.1)
                ax[4, col].set_ylim(-0.1, 1.1)
        ax[0, col].set_title(sim_name)
    for axes in ax.flatten():
        axes.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("../figures/model_discrim_lesion_simple_vis_speed.png")
    plt.close()

    fig, ax = plt.subplots(11, 3, squeeze=False, figsize=(18, 24))
    simulations = {
        "Control": d_ctl_0,
        "DMS Lesion": d_dms_0,
        "DLS Lesion": d_dls_0
    }
    for col, (sim_name, results) in enumerate(simulations.items()):
        vis = results["vis"]
        dms = results["dms"]
        premotor = results["premotor"]
        dls = results["dls"]
        motor = results["motor"]
        r = results["reward"]
        p = results["prediction"]
        rpe = results["reward_prediction_error"]
        w_vis_dms_rec = results["w_vis_dms_rec"]
        w_premotor_dls_rec = results["w_premotor_dls_rec"]
        w_vis_premotor_rec = results["w_vis_premotor_rec"]
        w_premotor_motor_rec = results["w_premotor_motor_rec"]
        resp = results["resp"]
        cat = results["cat"]
        ax[0, col].plot(vis.mean(0), label="Visual")
        ax[1, col].plot(dms.mean(0), label="DMS")
        ax[2, col].plot(premotor.mean(0), label="Premotor")
        ax[3, col].plot(dls.mean(0), label="DLS")
        ax[4, col].plot(motor.mean(0), label="Motor")
        ax[5, col].plot(r.mean(0), label="Reward")
        ax[5, col].plot(p.mean(0), label="Prediction")
        ax[5, col].plot(rpe.mean(0), label="RPE")
        ax[6, col].plot(np.mean(resp == cat, axis=0),
                        label="Response Accuracy")
        ax[6, col].plot(np.mean(resp, axis=0), label="Response")
        for i in range(2):
            for j in range(2):
                ax[7, col].plot(w_vis_dms_rec.mean(0)[:, i, j],
                                label=f"Weight: Visual-DMS [{i},{j}]")
                ax[8, col].plot(w_premotor_dls_rec.mean(0)[:, i, j],
                                label=f"Weight: Premotor-DLS [{i},{j}]")
                ax[9, col].plot(w_vis_premotor_rec.mean(0)[:, i, j],
                                label=f"Weight: Visual-Premotor [{i},{j}]")
                ax[10, col].plot(w_premotor_motor_rec.mean(0)[:, i, j],
                                 label=f"Weight: Premotor-Motor [{i},{j}]")

        ax[0, col].set_title(sim_name)
    for axes in ax.flatten():
        axes.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("../figures/model_discrim_lesion_simple_vis_speed_2.png")
    plt.close()


# NOTE: dms lesions across different stages of learning
# d_ctl_0 = simulate_model(0.01, 0.0, 1.0, 0.0, trl_lesion=100)
# d_dms_0 = simulate_model(0.01, 0.0, 1.0, 0.0, trl_lesion=200)
# d_dls_0 = simulate_model(0.01, 0.0, 1.0, 0.0, trl_lesion=400)
# plot_results()

# NOTE: dls lesions across different stages of learning
# d_ctl_0 = simulate_model(1.0, 0.0, 0.01, 0.0, trl_lesion=100)
# d_dms_0 = simulate_model(1.0, 0.0, 0.01, 0.0, trl_lesion=200)
# d_dls_0 = simulate_model(1.0, 0.0, 0.01, 0.0, trl_lesion=400)
# plot_results()

# NOTE: dms vs dls lesions from the start
# d_ctl_0 = simulate_model(1.0, 0.0, 1.0, 0.0, trl_lesion=0)
# d_dms_0 = simulate_model(0.5, 0.0, 1.0, 0.0, trl_lesion=0)
# d_dls_0 = simulate_model(1.0, 0.0, 0.5, 0.0, trl_lesion=0)
# plot_results()

lsc = 1.0
lsd = 0.1
trl_lesion = [0, 200, 400, 600, 800]
dms_rec = []
dls_rec = []
for tl in trl_lesion:
    dms = simulate_model(lsc, lsd, 1.0, 0.0, trl_lesion=tl)
    dls = simulate_model(1.0, 0.0, lsc, lsd, trl_lesion=tl)

    d_dms = pd.DataFrame({
        'Accuracy':
        dms['Accuracy'].mean(axis=0),
        'w_vis_dms_00':
        dms['w_vis_dms_rec'].mean(axis=0)[:, 0, 0],
        'w_premotor_dls_00':
        dms['w_premotor_dls_rec'].mean(axis=0)[:, 0, 0],
        'w_vis_premotor_00':
        dms['w_vis_premotor_rec'].mean(axis=0)[:, 0, 0],
        'w_premotor_motor_00':
        dms['w_premotor_motor_rec'].mean(axis=0)[:, 0, 0],
        'w_vis_dms_01':
        dms['w_vis_dms_rec'].mean(axis=0)[:, 0, 1],
        'w_premotor_dls_01':
        dms['w_premotor_dls_rec'].mean(axis=0)[:, 0, 1],
        'w_vis_premotor_01':
        dms['w_vis_premotor_rec'].mean(axis=0)[:, 0, 1],
        'w_premotor_motor_01':
        dms['w_premotor_motor_rec'].mean(axis=0)[:, 0, 1],
        'w_vis_dms_10':
        dms['w_vis_dms_rec'].mean(axis=0)[:, 1, 0],
        'w_premotor_dls_10':
        dms['w_premotor_dls_rec'].mean(axis=0)[:, 1, 0],
        'w_vis_premotor_10':
        dms['w_vis_premotor_rec'].mean(axis=0)[:, 1, 0],
        'w_premotor_motor_10':
        dms['w_premotor_motor_rec'].mean(axis=0)[:, 1, 0],
        'w_vis_dms_11':
        dms['w_vis_dms_rec'].mean(axis=0)[:, 1, 1],
        'w_premotor_dls_11':
        dms['w_premotor_dls_rec'].mean(axis=0)[:, 1, 1],
        'w_vis_premotor_11':
        dms['w_vis_premotor_rec'].mean(axis=0)[:, 1, 1],
        'w_premotor_motor_11':
        dms['w_premotor_motor_rec'].mean(axis=0)[:, 1, 1],
    })

    d_dls = pd.DataFrame({
        'Accuracy':
        dls['Accuracy'].mean(axis=0),
        'w_vis_dms_00':
        dls['w_vis_dms_rec'].mean(axis=0)[:, 0, 0],
        'w_premotor_dls_00':
        dls['w_premotor_dls_rec'].mean(axis=0)[:, 0, 0],
        'w_vis_premotor_00':
        dls['w_vis_premotor_rec'].mean(axis=0)[:, 0, 0],
        'w_premotor_motor_00':
        dls['w_premotor_motor_rec'].mean(axis=0)[:, 0, 0],
        'w_vis_dms_01':
        dls['w_vis_dms_rec'].mean(axis=0)[:, 0, 1],
        'w_premotor_dls_01':
        dls['w_premotor_dls_rec'].mean(axis=0)[:, 0, 1],
        'w_vis_premotor_01':
        dls['w_vis_premotor_rec'].mean(axis=0)[:, 0, 1],
        'w_premotor_motor_01':
        dls['w_premotor_motor_rec'].mean(axis=0)[:, 0, 1],
        'w_vis_dms_10':
        dls['w_vis_dms_rec'].mean(axis=0)[:, 1, 0],
        'w_premotor_dls_10':
        dls['w_premotor_dls_rec'].mean(axis=0)[:, 1, 0],
        'w_vis_premotor_10':
        dls['w_vis_premotor_rec'].mean(axis=0)[:, 1, 0],
        'w_premotor_motor_10':
        dls['w_premotor_motor_rec'].mean(axis=0)[:, 1, 0],
        'w_vis_dms_11':
        dls['w_vis_dms_rec'].mean(axis=0)[:, 1, 1],
        'w_premotor_dls_11':
        dls['w_premotor_dls_rec'].mean(axis=0)[:, 1, 1],
        'w_vis_premotor_11':
        dls['w_vis_premotor_rec'].mean(axis=0)[:, 1, 1],
        'w_premotor_motor_11':
        dls['w_premotor_motor_rec'].mean(axis=0)[:, 1, 1],
    })

    d_dms['lesion'] = 'DMS'
    d_dls['lesion'] = 'DLS'

    d_dms['trl_lesion'] = tl
    d_dls['trl_lesion'] = tl

    dms_rec.append(d_dms)
    dls_rec.append(d_dls)

dms_rec = pd.concat(dms_rec)
dls_rec = pd.concat(dls_rec)

d = pd.concat([dms_rec, dls_rec])
d['Trial'] = d.groupby(['lesion', 'trl_lesion']).cumcount() + 1

fig, ax = plt.subplots(5, d.trl_lesion.unique().shape[0],
                       squeeze=False,
                       figsize=(28, 18))
for i, tl in enumerate(d.trl_lesion.unique()):
    dd = d.loc[(d.trl_lesion == tl) & (d.Trial < d.Trial.max() - 1)]
    sns.lineplot(data=dd,
                 x="Trial",
                 y="Accuracy",
                 hue='lesion',
                 errorbar=None,
                 ax=ax[0, i])
    # weights: visual-dms
    for w in ["w_vis_dms_00", "w_vis_dms_01", "w_vis_dms_10", "w_vis_dms_11"]:
        sns.lineplot(
            data=dd,
            x="Trial",
            y=w,
            hue='lesion',
            errorbar=None,
            legend=False,
            ax=ax[1, i],
        )
        ax[1, i].set_title("Weight: Visual-DMS")
    # weights: premotor-dls
    for w in [
            "w_premotor_dls_00", "w_premotor_dls_01", "w_premotor_dls_10",
            "w_premotor_dls_11"
    ]:
        sns.lineplot(
            data=dd,
            x="Trial",
            y=w,
            hue='lesion',
            errorbar=None,
            legend=False,
            ax=ax[2, i],
        )
        ax[2, i].set_title("Weight: PM-DLS")
    # weights: visual-pm
    for w in [
            "w_vis_premotor_00", "w_vis_premotor_01", "w_vis_premotor_10",
            "w_vis_premotor_11"
    ]:
        sns.lineplot(
            data=dd,
            x="Trial",
            y=w,
            hue='lesion',
            errorbar=None,
            legend=False,
            ax=ax[3, i],
        )
        ax[3, i].set_title("Weight: Visual-PM")
    # weights: pm-motor
    for w in [
            "w_premotor_motor_00", "w_premotor_motor_01",
            "w_premotor_motor_10", "w_premotor_motor_11"
    ]:
        sns.lineplot(
            data=dd,
            x="Trial",
            y=w,
            hue='lesion',
            errorbar=None,
            legend=False,
            ax=ax[4, i],
        )
        ax[4, i].set_title("Weight: PM-Motor")
    [x.axvline(tl, color='black', linestyle='--') for x in ax[:, i].flatten()]
    [x.set_ylim(-0.1, 1.1) for x in ax[:, i].flatten()]
plt.tight_layout()
plt.savefig("../figures/model_discrim_lesion_simple_vis_speed_3.png")
plt.close()
