#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>
#include <sys/types.h>

#define N_CELLS 8
#define VIS_DIM 100
#define VIS_SIZE (VIS_DIM * VIS_DIM)

typedef struct {
    double *x;
    double *y;
    int *cat;
    size_t len;
} Dataset;

typedef struct {
    uint64_t state;
} Rng;

typedef struct {
    int rotations[3];
    int n_rotations;
    int n_simulations;
    int n_trials;
    int probe_trial_onsets[16];
    int n_probe_onsets;
    int n_probe_trials;
    uint64_t seed;
    char output_dir[512];
} Config;

static inline uint64_t rng_next_u64(Rng *rng) {
    uint64_t x = rng->state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    rng->state = x;
    return x * 2685821657736338717ULL;
}

static inline double rng_uniform01(Rng *rng) {
    return (rng_next_u64(rng) >> 11) * (1.0 / 9007199254740992.0);
}

static inline double rng_uniform(Rng *rng, double low, double high) {
    return low + (high - low) * rng_uniform01(rng);
}

static double rng_normal(Rng *rng, double mean, double sd) {
    static bool has_spare = false;
    static double spare = 0.0;

    if (sd == 0.0) {
        return mean;
    }

    if (has_spare) {
        has_spare = false;
        return mean + sd * spare;
    }

    double u1 = rng_uniform01(rng);
    double u2 = rng_uniform01(rng);
    if (u1 <= 1e-12) {
        u1 = 1e-12;
    }
    double mag = sqrt(-2.0 * log(u1));
    double z0 = mag * cos(2.0 * M_PI * u2);
    spare = mag * sin(2.0 * M_PI * u2);
    has_spare = true;
    return mean + sd * z0;
}

static inline double clip(double value, double low, double high) {
    if (value < low) {
        return low;
    }
    if (value > high) {
        return high;
    }
    return value;
}

static void ensure_dir(const char *path) {
    struct stat st;
    if (stat(path, &st) == 0) {
        if (!S_ISDIR(st.st_mode)) {
            fprintf(stderr, "%s exists but is not a directory\n", path);
            exit(1);
        }
        return;
    }
    if (mkdir(path, 0755) != 0) {
        perror("mkdir");
        exit(1);
    }
}

static Dataset alloc_dataset(size_t len) {
    Dataset ds;
    ds.x = malloc(len * sizeof(double));
    ds.y = malloc(len * sizeof(double));
    ds.cat = malloc(len * sizeof(int));
    ds.len = len;
    if (!ds.x || !ds.y || !ds.cat) {
        fprintf(stderr, "Allocation failed for dataset of length %zu\n", len);
        exit(1);
    }
    return ds;
}

static void free_dataset(Dataset *ds) {
    free(ds->x);
    free(ds->y);
    free(ds->cat);
    ds->x = NULL;
    ds->y = NULL;
    ds->cat = NULL;
    ds->len = 0;
}

static void shuffle_dataset(Dataset *ds, Rng *rng) {
    for (size_t i = ds->len - 1; i > 0; --i) {
        size_t j = (size_t) floor(rng_uniform01(rng) * (double) (i + 1));
        double tx = ds->x[i];
        double ty = ds->y[i];
        int tc = ds->cat[i];
        ds->x[i] = ds->x[j];
        ds->y[i] = ds->y[j];
        ds->cat[i] = ds->cat[j];
        ds->x[j] = tx;
        ds->y[j] = ty;
        ds->cat[j] = tc;
    }
}

static Dataset copy_dataset(const Dataset *src) {
    Dataset dst = alloc_dataset(src->len);
    memcpy(dst.x, src->x, src->len * sizeof(double));
    memcpy(dst.y, src->y, src->len * sizeof(double));
    memcpy(dst.cat, src->cat, src->len * sizeof(int));
    return dst;
}

static Dataset sample_dataset(const Dataset *src, size_t n, uint64_t seed) {
    if (n > src->len) {
        fprintf(stderr, "Sample size %zu exceeds dataset length %zu\n", n, src->len);
        exit(1);
    }

    size_t *indices = malloc(src->len * sizeof(size_t));
    if (!indices) {
        fprintf(stderr, "Allocation failed for sample indices\n");
        exit(1);
    }
    for (size_t i = 0; i < src->len; ++i) {
        indices[i] = i;
    }

    Rng rng = {.state = seed ? seed : 1ULL};
    for (size_t i = 0; i < n; ++i) {
        size_t j = i + (size_t) floor(rng_uniform01(&rng) * (double) (src->len - i));
        size_t tmp = indices[i];
        indices[i] = indices[j];
        indices[j] = tmp;
    }

    Dataset out = alloc_dataset(n);
    for (size_t i = 0; i < n; ++i) {
        size_t idx = indices[i];
        out.x[i] = src->x[idx];
        out.y[i] = src->y[idx];
        out.cat[i] = src->cat[idx];
    }

    free(indices);
    return out;
}

static void rotate_dataset(Dataset *ds, double theta_rad) {
    double c = cos(theta_rad);
    double s = sin(theta_rad);
    for (size_t i = 0; i < ds->len; ++i) {
        double x = ds->x[i] - 50.0;
        double y = ds->y[i] - 50.0;
        double xr = c * x - s * y;
        double yr = s * x + c * y;
        ds->x[i] = xr + 50.0;
        ds->y[i] = yr + 50.0;
    }
}

static void sample_within_ellipse(Rng *rng,
                                  double mean_x,
                                  double mean_y,
                                  double std_major,
                                  double std_minor,
                                  double rot_c,
                                  double rot_s,
                                  double *out_x,
                                  double *out_y,
                                  size_t n_samples) {
    for (size_t i = 0; i < n_samples; ++i) {
        double r = sqrt(rng_uniform(rng, 0.0, 9.0));
        double angle = rng_uniform(rng, 0.0, 2.0 * M_PI);
        double x = r * cos(angle);
        double y = r * sin(angle);
        double x_scaled = x * std_major;
        double y_scaled = y * std_minor;
        double xr = rot_c * x_scaled - rot_s * y_scaled;
        double yr = rot_s * x_scaled + rot_c * y_scaled;
        out_x[i] = xr + mean_x;
        out_y[i] = yr + mean_y;
    }
}

static void make_stim_cats(int n_stimuli_per_category,
                           Rng *rng,
                           Dataset *ds_out,
                           Dataset *ds_90_out,
                           Dataset *ds_180_out) {
    const double var = 100.0;
    const double corr = 0.9;
    const double sigma = sqrt(var);
    const double theta = 45.0 * M_PI / 180.0;
    const double rot_c = cos(theta);
    const double rot_s = sin(theta);
    const double std_major = sigma * sqrt(1.0 + corr);
    const double std_minor = sigma * sqrt(1.0 - corr);

    size_t total = (size_t) (2 * n_stimuli_per_category);
    Dataset ds = alloc_dataset(total);

    sample_within_ellipse(rng,
                          40.0,
                          60.0,
                          std_major,
                          std_minor,
                          rot_c,
                          rot_s,
                          ds.x,
                          ds.y,
                          (size_t) n_stimuli_per_category);

    sample_within_ellipse(rng,
                          60.0,
                          40.0,
                          std_major,
                          std_minor,
                          rot_c,
                          rot_s,
                          ds.x + n_stimuli_per_category,
                          ds.y + n_stimuli_per_category,
                          (size_t) n_stimuli_per_category);

    for (int i = 0; i < n_stimuli_per_category; ++i) {
        ds.cat[i] = 1;
        ds.cat[i + n_stimuli_per_category] = 2;
    }

    shuffle_dataset(&ds, rng);

    Dataset ds_90 = copy_dataset(&ds);
    Dataset ds_180 = copy_dataset(&ds);
    rotate_dataset(&ds_90, 90.0 * M_PI / 180.0);
    rotate_dataset(&ds_180, 180.0 * M_PI / 180.0);

    *ds_out = ds;
    *ds_90_out = ds_90;
    *ds_180_out = ds_180;
}

static Dataset insert_probe_trials(const Dataset *train,
                                   const Dataset *probe,
                                   const int *onsets,
                                   int n_onsets) {
    size_t total_len = train->len + (size_t) n_onsets * probe->len;
    Dataset out = alloc_dataset(total_len);

    size_t in_pos = 0;
    size_t out_pos = 0;
    int onset_idx = 0;

    while (in_pos < train->len) {
        if (onset_idx < n_onsets && (size_t) onsets[onset_idx] == in_pos) {
            memcpy(out.x + out_pos, probe->x, probe->len * sizeof(double));
            memcpy(out.y + out_pos, probe->y, probe->len * sizeof(double));
            memcpy(out.cat + out_pos, probe->cat, probe->len * sizeof(int));
            out_pos += probe->len;
            onset_idx++;
        }
        out.x[out_pos] = train->x[in_pos];
        out.y[out_pos] = train->y[in_pos];
        out.cat[out_pos] = train->cat[in_pos];
        out_pos++;
        in_pos++;
    }

    while (onset_idx < n_onsets && (size_t) onsets[onset_idx] == train->len) {
        memcpy(out.x + out_pos, probe->x, probe->len * sizeof(double));
        memcpy(out.y + out_pos, probe->y, probe->len * sizeof(double));
        memcpy(out.cat + out_pos, probe->cat, probe->len * sizeof(int));
        out_pos += probe->len;
        onset_idx++;
    }

    out.len = out_pos;
    return out;
}

static void fill_zero(double *arr, size_t len) {
    memset(arr, 0, len * sizeof(double));
}

static void write_trial_summary_csv(const char *path,
                                    const double *cat,
                                    const double *resp,
                                    const double *rt,
                                    const double *r,
                                    const double *p,
                                    const double *rpe,
                                    int n_simulations,
                                    int total_trials) {
    FILE *fp = fopen(path, "w");
    if (!fp) {
        perror(path);
        exit(1);
    }
    fprintf(fp, "simulation,trial,cat,resp,rt,r,p,rpe\n");
    for (int sim = 0; sim < n_simulations; ++sim) {
        for (int trl = 0; trl < total_trials; ++trl) {
            int idx = sim * total_trials + trl;
            fprintf(fp,
                    "%d,%d,%.0f,%.0f,%.10f,%.10f,%.10f,%.10f\n",
                    sim,
                    trl,
                    cat[idx],
                    resp[idx],
                    rt[idx],
                    r[idx],
                    p[idx],
                    rpe[idx]);
        }
    }
    fclose(fp);
}

static void write_dataset_csv(const char *path, const Dataset *ds) {
    FILE *fp = fopen(path, "w");
    if (!fp) {
        perror(path);
        exit(1);
    }
    fprintf(fp, "trial,x,y,cat\n");
    for (size_t i = 0; i < ds->len; ++i) {
        fprintf(fp, "%zu,%.10f,%.10f,%d\n", i, ds->x[i], ds->y[i], ds->cat[i]);
    }
    fclose(fp);
}

static void write_matrix_csv(const char *path,
                             const double *data,
                             int rows,
                             int cols,
                             int stride) {
    FILE *fp = fopen(path, "w");
    if (!fp) {
        perror(path);
        exit(1);
    }
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            if (c > 0) {
                fputc(',', fp);
            }
            fprintf(fp, "%.10f", data[r * stride + c]);
        }
        fputc('\n', fp);
    }
    fclose(fp);
}

static double dot_product(const double *a, const double *b, size_t len) {
    double sum = 0.0;
    for (size_t i = 0; i < len; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

static void parse_args(int argc, char **argv, Config *cfg) {
    cfg->rotations[0] = 90;
    cfg->rotations[1] = 180;
    cfg->n_rotations = 2;
    cfg->n_simulations = 1;
    cfg->n_trials = 600;
    cfg->probe_trial_onsets[0] = 600;
    cfg->n_probe_onsets = 1;
    cfg->n_probe_trials = 200;
    cfg->seed = 1;
    snprintf(cfg->output_dir, sizeof(cfg->output_dir), "output");

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--n-simulations") == 0 && i + 1 < argc) {
            cfg->n_simulations = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--n-trials") == 0 && i + 1 < argc) {
            cfg->n_trials = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--n-probe-trials") == 0 && i + 1 < argc) {
            cfg->n_probe_trials = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            cfg->seed = (uint64_t) strtoull(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--output-dir") == 0 && i + 1 < argc) {
            snprintf(cfg->output_dir, sizeof(cfg->output_dir), "%s", argv[++i]);
        } else if (strcmp(argv[i], "--rotations") == 0) {
            cfg->n_rotations = 0;
            while (i + 1 < argc && argv[i + 1][0] != '-') {
                cfg->rotations[cfg->n_rotations++] = atoi(argv[++i]);
            }
        } else if (strcmp(argv[i], "--probe-trial-onsets") == 0) {
            cfg->n_probe_onsets = 0;
            while (i + 1 < argc && argv[i + 1][0] != '-') {
                cfg->probe_trial_onsets[cfg->n_probe_onsets++] = atoi(argv[++i]);
            }
        } else {
            fprintf(stderr, "Unknown or incomplete argument: %s\n", argv[i]);
            exit(1);
        }
    }
}

static void simulate_rotation(int rotation, const Config *cfg) {
    const double alpha_critic = 0.05;
    const double nmda_thresh = 0.0;
    const double alpha_w_vis_dms = 1e-9;
    const double beta_w_vis_dms = 1e-11;
    const double gamma_w_vis_dms = 0.0;
    const double alpha_w_premotor_dls = 2e-15;
    const double beta_w_premotor_dls = 1e-15;
    const double gamma_w_premotor_dls = 0.0;
    const double alpha_w_vis_premotor = 0.0;
    const double beta_w_vis_premotor = 0.0;
    const double alpha_w_premotor_motor = 0.0;
    const double beta_w_premotor_motor = 0.0;
    const int n_steps = 3000;
    const double psp_amp = 1e5;
    const double psp_decay = 200.0;
    const double resp_thresh = 5e6;

    const double C[N_CELLS] = {50, 50, 100, 100, 50, 50, 100, 100};
    const double vr[N_CELLS] = {-80, -80, -60, -60, -80, -80, -60, -60};
    const double vt[N_CELLS] = {-25, -25, -40, -40, -25, -25, -40, -40};
    const double vpeak[N_CELLS] = {40, 40, 35, 35, 40, 40, 35, 35};
    const double a[N_CELLS] = {0.01, 0.01, 0.03, 0.03, 0.01, 0.01, 0.03, 0.03};
    const double b[N_CELLS] = {-20, -20, -2, -2, -20, -20, -2, -2};
    const double c_reset[N_CELLS] = {-55, -55, -50, -50, -55, -55, -50, -50};
    const double d[N_CELLS] = {150, 150, 100, 100, 150, 150, 100, 100};
    const double k[N_CELLS] = {1, 1, 0.7, 0.7, 1, 1, 0.7, 0.7};

    Rng rng = {.state = cfg->seed ? cfg->seed : 1ULL};
    Dataset ds, ds_90, ds_180;
    make_stim_cats(cfg->n_trials / 2, &rng, &ds, &ds_90, &ds_180);

    Dataset ds_0_probe = sample_dataset(&ds, (size_t) cfg->n_probe_trials, 1);
    Dataset ds_90_probe = sample_dataset(&ds_90, (size_t) cfg->n_probe_trials, 1);
    Dataset ds_180_probe = sample_dataset(&ds_180, (size_t) cfg->n_probe_trials, 1);

    Dataset selected_probe = {0};
    if (rotation == 0) {
        selected_probe = copy_dataset(&ds_0_probe);
    } else if (rotation == 90) {
        selected_probe = copy_dataset(&ds_90_probe);
    } else if (rotation == 180) {
        selected_probe = copy_dataset(&ds_180_probe);
    } else {
        fprintf(stderr, "Unsupported rotation %d\n", rotation);
        exit(1);
    }

    Dataset trial_ds = insert_probe_trials(&ds, &selected_probe, cfg->probe_trial_onsets, cfg->n_probe_onsets);
    int total_trials = (int) trial_ds.len;
    ensure_dir(cfg->output_dir);

    double *cat = calloc((size_t) cfg->n_simulations * total_trials, sizeof(double));
    double *resp = calloc((size_t) cfg->n_simulations * total_trials, sizeof(double));
    double *rt = calloc((size_t) cfg->n_simulations * total_trials, sizeof(double));
    double *r = calloc((size_t) cfg->n_simulations * total_trials, sizeof(double));
    double *p = malloc((size_t) cfg->n_simulations * total_trials * sizeof(double));
    double *rpe = calloc((size_t) cfg->n_simulations * total_trials, sizeof(double));
    if (!cat || !resp || !rt || !r || !p || !rpe) {
        fprintf(stderr, "Allocation failed for summary arrays\n");
        exit(1);
    }
    for (int i = 0; i < cfg->n_simulations * total_trials; ++i) {
        p[i] = 0.5;
    }

    double *I_ext = calloc((size_t) N_CELLS * n_steps, sizeof(double));
    double *I_net = calloc((size_t) N_CELLS * n_steps, sizeof(double));
    double *v = calloc((size_t) N_CELLS * n_steps, sizeof(double));
    double *u = calloc((size_t) N_CELLS * n_steps, sizeof(double));
    double *g = calloc((size_t) N_CELLS * n_steps, sizeof(double));
    double *spike = calloc((size_t) N_CELLS * n_steps, sizeof(double));
    double *vis = calloc(VIS_SIZE, sizeof(double));
    double *xg = calloc(VIS_SIZE, sizeof(double));
    double *yg = calloc(VIS_SIZE, sizeof(double));
    double *w_vis_dms_A = calloc(VIS_SIZE, sizeof(double));
    double *w_vis_dms_B = calloc(VIS_SIZE, sizeof(double));
    double *w_vis_pm_A = calloc(VIS_SIZE, sizeof(double));
    double *w_vis_pm_B = calloc(VIS_SIZE, sizeof(double));
    double *w = calloc((size_t) N_CELLS * N_CELLS, sizeof(double));

    if (!I_ext || !I_net || !v || !u || !g || !spike || !vis || !xg || !yg ||
        !w_vis_dms_A || !w_vis_dms_B || !w_vis_pm_A || !w_vis_pm_B || !w) {
        fprintf(stderr, "Allocation failed for simulation arrays\n");
        exit(1);
    }

    for (int yy = 0; yy < VIS_DIM; ++yy) {
        for (int xx = 0; xx < VIS_DIM; ++xx) {
            int idx = yy * VIS_DIM + xx;
            xg[idx] = (double) xx;
            yg[idx] = (double) yy;
        }
    }

    for (int sim = 0; sim < cfg->n_simulations; ++sim) {
        fill_zero(w, (size_t) N_CELLS * N_CELLS);
        for (int idx = 0; idx < VIS_SIZE; ++idx) {
            w_vis_dms_A[idx] = rng_uniform(&rng, 0.4, 0.6);
            w_vis_dms_B[idx] = rng_uniform(&rng, 0.4, 0.6);
            w_vis_pm_A[idx] = 0.0;
            w_vis_pm_B[idx] = 0.0;
        }

        w[0 * N_CELLS + 2] = 0.04;
        w[1 * N_CELLS + 3] = 0.04;
        w[2 * N_CELLS + 4] = rng_uniform(&rng, 0.49, 0.51);
        w[2 * N_CELLS + 5] = rng_uniform(&rng, 0.49, 0.51);
        w[3 * N_CELLS + 4] = rng_uniform(&rng, 0.49, 0.51);
        w[3 * N_CELLS + 5] = rng_uniform(&rng, 0.49, 0.51);
        w[4 * N_CELLS + 6] = 0.04;
        w[5 * N_CELLS + 7] = 0.04;
        w[0 * N_CELLS + 1] = -0.2;
        w[1 * N_CELLS + 0] = -0.2;
        w[4 * N_CELLS + 5] = -0.5;
        w[5 * N_CELLS + 4] = -0.5;

        for (int trl = 0; trl < total_trials - 1; ++trl) {
            fill_zero(I_ext, (size_t) N_CELLS * n_steps);
            fill_zero(I_net, (size_t) N_CELLS * n_steps);
            fill_zero(v, (size_t) N_CELLS * n_steps);
            fill_zero(u, (size_t) N_CELLS * n_steps);
            fill_zero(g, (size_t) N_CELLS * n_steps);
            fill_zero(spike, (size_t) N_CELLS * n_steps);
            for (int cell = 0; cell < N_CELLS; ++cell) {
                v[cell * n_steps] = vr[cell];
            }

            double x = trial_ds.x[trl];
            double y = trial_ds.y[trl];
            cat[sim * total_trials + trl] = (double) trial_ds.cat[trl];

            for (int idx = 0; idx < VIS_SIZE; ++idx) {
                double dx = xg[idx] - x;
                double dy = yg[idx] - y;
                vis[idx] = 7.0 * exp(-((dx * dx + dy * dy) / (2.0 * 7.0 * 7.0)));
            }

            double vis_dms_act_A = dot_product(vis, w_vis_dms_A, VIS_SIZE);
            double vis_dms_act_B = dot_product(vis, w_vis_dms_B, VIS_SIZE);
            double vis_pm_act_A = dot_product(vis, w_vis_pm_A, VIS_SIZE);
            double vis_pm_act_B = dot_product(vis, w_vis_pm_B, VIS_SIZE);

            int start = n_steps / 3;
            int end = 2 * n_steps / 3;
            for (int i = start; i < end; ++i) {
                I_ext[0 * n_steps + i] = vis_dms_act_A;
                I_ext[1 * n_steps + i] = vis_dms_act_B;
                I_ext[2 * n_steps + i] = vis_pm_act_A;
                I_ext[3 * n_steps + i] = vis_pm_act_B;
            }

            int i;
            for (i = 1; i < n_steps; ++i) {
                for (int dest = 0; dest < N_CELLS; ++dest) {
                    double net = 0.0;
                    for (int src = 0; src < N_CELLS; ++src) {
                        net += w[src * N_CELLS + dest] * g[src * n_steps + (i - 1)];
                    }
                    I_net[dest * n_steps + (i - 1)] = net + I_ext[dest * n_steps + (i - 1)];
                }

                for (int cell = 0; cell < N_CELLS; ++cell) {
                    double mu = 1.0;
                    double sig = (cell == 0 || cell == 1 || cell == 4 || cell == 5) ? 1.0 : 0.0;
                    double noise = rng_normal(&rng, mu, sig);
                    double v_prev = v[cell * n_steps + (i - 1)];
                    double u_prev = u[cell * n_steps + (i - 1)];
                    double g_prev = g[cell * n_steps + (i - 1)];
                    double spike_prev = spike[cell * n_steps + (i - 1)];
                    double dvdt = (k[cell] * (v_prev - vr[cell]) * (v_prev - vt[cell]) -
                                   u_prev + I_net[cell * n_steps + (i - 1)] * noise) / C[cell];
                    double dudt = a[cell] * (b[cell] * (v_prev - vr[cell]) - u_prev);
                    double dgdt = (-g_prev + psp_amp * spike_prev) / psp_decay;

                    double v_new = v_prev + dvdt;
                    double u_new = u_prev + dudt;
                    double g_new = g_prev + dgdt;

                    if (v_new < -100.0) {
                        v_new = -100.0;
                    }
                    if (v_new >= vpeak[cell]) {
                        v[cell * n_steps + (i - 1)] = vpeak[cell];
                        v_new = c_reset[cell];
                        u_new += d[cell];
                        spike[cell * n_steps + i] = 1.0;
                    }

                    v[cell * n_steps + i] = v_new;
                    u[cell * n_steps + i] = u_new;
                    g[cell * n_steps + i] = g_new;
                }

                if ((g[6 * n_steps + i] - g[7 * n_steps + i]) > resp_thresh) {
                    resp[sim * total_trials + trl] = 1.0;
                    rt[sim * total_trials + trl] = (double) i;
                    break;
                }
                if ((g[7 * n_steps + i] - g[6 * n_steps + i]) > resp_thresh) {
                    resp[sim * total_trials + trl] = 2.0;
                    rt[sim * total_trials + trl] = (double) i;
                    break;
                }
            }

            if (rt[sim * total_trials + trl] == 0.0) {
                rt[sim * total_trials + trl] = (double) i;
                double g6_sum = 0.0;
                double g7_sum = 0.0;
                for (int step = 0; step < n_steps; ++step) {
                    g6_sum += g[6 * n_steps + step];
                    g7_sum += g[7 * n_steps + step];
                }
                if (g6_sum > g7_sum) {
                    resp[sim * total_trials + trl] = 1.0;
                } else if (g7_sum > g6_sum) {
                    resp[sim * total_trials + trl] = 2.0;
                } else {
                    resp[sim * total_trials + trl] = rng_uniform01(&rng) < 0.5 ? 1.0 : 2.0;
                }
            }

            r[sim * total_trials + trl] =
                (cat[sim * total_trials + trl] == resp[sim * total_trials + trl]) ? 1.0 : 0.0;
            rpe[sim * total_trials + trl] = r[sim * total_trials + trl] - p[sim * total_trials + trl];
            p[sim * total_trials + trl + 1] =
                p[sim * total_trials + trl] + alpha_critic * rpe[sim * total_trials + trl];

            double dms_A = 0.0;
            double dms_B = 0.0;
            double pm_A = 0.0;
            double pm_B = 0.0;
            for (int step = 0; step < n_steps; ++step) {
                dms_A += g[0 * n_steps + step];
                dms_B += g[1 * n_steps + step];
                pm_A += g[2 * n_steps + step];
                pm_B += g[3 * n_steps + step];
            }

            double rpe_val = rpe[sim * total_trials + trl];
            double dms_A_pos = fmax(dms_A - nmda_thresh, 0.0);
            double dms_A_neg = fmax(nmda_thresh - dms_A, 0.0);
            double dms_B_pos = fmax(dms_B - nmda_thresh, 0.0);
            double dms_B_neg = fmax(nmda_thresh - dms_B, 0.0);
            double rpe_pos = fmax(rpe_val, 0.0);
            double rpe_neg = fmin(rpe_val, 0.0);

            for (int idx = 0; idx < VIS_SIZE; ++idx) {
                double dw1 = alpha_w_vis_dms * vis[idx] * dms_A_pos * rpe_pos * (1.0 - w_vis_dms_A[idx]);
                double dw2 = beta_w_vis_dms * vis[idx] * dms_A_pos * rpe_neg * w_vis_dms_A[idx];
                double dw3 = -gamma_w_vis_dms * vis[idx] * dms_A_neg * w_vis_dms_A[idx];
                w_vis_dms_A[idx] = clip(w_vis_dms_A[idx] + dw1 + dw2 + dw3, 0.0, 1.0);

                dw1 = alpha_w_vis_dms * vis[idx] * dms_B_pos * rpe_pos * (1.0 - w_vis_dms_B[idx]);
                dw2 = beta_w_vis_dms * vis[idx] * dms_B_pos * rpe_neg * w_vis_dms_B[idx];
                dw3 = -gamma_w_vis_dms * vis[idx] * dms_B_neg * w_vis_dms_B[idx];
                w_vis_dms_B[idx] = clip(w_vis_dms_B[idx] + dw1 + dw2 + dw3, 0.0, 1.0);
            }

            int pre_indices_1[4] = {2, 2, 3, 3};
            int post_indices_1[4] = {4, 5, 4, 5};
            for (int syn = 0; syn < 4; ++syn) {
                double pre_activity = 0.0;
                double post_activity = 0.0;
                int pre = pre_indices_1[syn];
                int post = post_indices_1[syn];
                for (int step = 0; step < n_steps; ++step) {
                    pre_activity += g[pre * n_steps + step];
                    post_activity += g[post * n_steps + step];
                }
                double dw1 = alpha_w_premotor_dls * pre_activity *
                             fmax(post_activity - nmda_thresh, 0.0) * fmax(rpe_val, 0.0) *
                             (1.0 - w[pre * N_CELLS + post]);
                double dw2 = beta_w_premotor_dls * pre_activity *
                             fmax(post_activity - nmda_thresh, 0.0) * fmin(rpe_val, 0.0) *
                             w[pre * N_CELLS + post];
                double dw3 = -gamma_w_premotor_dls * pre_activity *
                             fmax(nmda_thresh - post_activity, 0.0) * w[pre * N_CELLS + post];
                w[pre * N_CELLS + post] = clip(w[pre * N_CELLS + post] + dw1 + dw2 + dw3, 0.0, 1.0);
            }

            double pm_A_pos = fmax(pm_A - nmda_thresh, 0.0);
            double pm_A_neg = fmax(nmda_thresh - pm_A, 0.0);
            double pm_B_pos = fmax(pm_B - nmda_thresh, 0.0);
            double pm_B_neg = fmax(nmda_thresh - pm_B, 0.0);
            for (int idx = 0; idx < VIS_SIZE; ++idx) {
                double dw1 = alpha_w_vis_premotor * vis[idx] * pm_A_pos * (1.0 - w_vis_pm_A[idx]);
                double dw2 = -beta_w_vis_premotor * vis[idx] * pm_A_neg * w_vis_pm_A[idx];
                w_vis_pm_A[idx] = clip(w_vis_pm_A[idx] + dw1 + dw2, 0.0, 1.0);

                dw1 = alpha_w_vis_premotor * vis[idx] * pm_B_pos * (1.0 - w_vis_pm_B[idx]);
                dw2 = -beta_w_vis_premotor * vis[idx] * pm_B_neg * w_vis_pm_B[idx];
                w_vis_pm_B[idx] = clip(w_vis_pm_B[idx] + dw1 + dw2, 0.0, 1.0);
            }

            int pre_indices_2[4] = {2, 2, 3, 3};
            int post_indices_2[4] = {6, 7, 6, 7};
            for (int syn = 0; syn < 4; ++syn) {
                double pre_activity = 0.0;
                double post_activity = 0.0;
                int pre = pre_indices_2[syn];
                int post = post_indices_2[syn];
                for (int step = 0; step < n_steps; ++step) {
                    pre_activity += g[pre * n_steps + step];
                    post_activity += g[post * n_steps + step];
                }
                double dw1 = alpha_w_premotor_motor * pre_activity *
                             fmax(post_activity - nmda_thresh, 0.0) *
                             (1.0 - w[pre * N_CELLS + post]);
                double dw2 = -beta_w_premotor_motor * pre_activity *
                             fmax(nmda_thresh - post_activity, 0.0) *
                             w[pre * N_CELLS + post];
                w[pre * N_CELLS + post] = clip(w[pre * N_CELLS + post] + dw1 + dw2, 0.0, 1.0);
            }
        }
    }

    double correct = 0.0;
    double rt_sum = 0.0;
    int counted = 0;
    for (int sim = 0; sim < cfg->n_simulations; ++sim) {
        for (int trl = 0; trl < total_trials - 1; ++trl) {
            if (resp[sim * total_trials + trl] == cat[sim * total_trials + trl]) {
                correct += 1.0;
            }
            rt_sum += rt[sim * total_trials + trl];
            counted++;
        }
    }
    printf("rotation=%d trials=%d mean_acc=%.6f mean_rt=%.3f\n",
           rotation,
           total_trials,
           counted ? correct / counted : 0.0,
           counted ? rt_sum / counted : 0.0);

    char path[1024];
    snprintf(path, sizeof(path), "%s/model_spiking_cat_90vs180_c_%d_trials.csv", cfg->output_dir, rotation);
    write_trial_summary_csv(path, cat, resp, rt, r, p, rpe, cfg->n_simulations, total_trials);

    snprintf(path, sizeof(path), "%s/model_spiking_cat_90vs180_c_%d_ds.csv", cfg->output_dir, rotation);
    write_dataset_csv(path, &trial_ds);

    snprintf(path, sizeof(path), "%s/model_spiking_cat_90vs180_c_%d_w_final.csv", cfg->output_dir, rotation);
    write_matrix_csv(path, w, N_CELLS, N_CELLS, N_CELLS);

    snprintf(path, sizeof(path), "%s/model_spiking_cat_90vs180_c_%d_w_vis_dms_A_final.csv", cfg->output_dir, rotation);
    write_matrix_csv(path, w_vis_dms_A, VIS_DIM, VIS_DIM, VIS_DIM);

    snprintf(path, sizeof(path), "%s/model_spiking_cat_90vs180_c_%d_w_vis_dms_B_final.csv", cfg->output_dir, rotation);
    write_matrix_csv(path, w_vis_dms_B, VIS_DIM, VIS_DIM, VIS_DIM);

    snprintf(path, sizeof(path), "%s/model_spiking_cat_90vs180_c_%d_w_vis_pm_A_final.csv", cfg->output_dir, rotation);
    write_matrix_csv(path, w_vis_pm_A, VIS_DIM, VIS_DIM, VIS_DIM);

    snprintf(path, sizeof(path), "%s/model_spiking_cat_90vs180_c_%d_w_vis_pm_B_final.csv", cfg->output_dir, rotation);
    write_matrix_csv(path, w_vis_pm_B, VIS_DIM, VIS_DIM, VIS_DIM);

    free(cat);
    free(resp);
    free(rt);
    free(r);
    free(p);
    free(rpe);
    free(I_ext);
    free(I_net);
    free(v);
    free(u);
    free(g);
    free(spike);
    free(vis);
    free(xg);
    free(yg);
    free(w_vis_dms_A);
    free(w_vis_dms_B);
    free(w_vis_pm_A);
    free(w_vis_pm_B);
    free(w);
    free_dataset(&ds);
    free_dataset(&ds_90);
    free_dataset(&ds_180);
    free_dataset(&ds_0_probe);
    free_dataset(&ds_90_probe);
    free_dataset(&ds_180_probe);
    free_dataset(&selected_probe);
    free_dataset(&trial_ds);
}

int main(int argc, char **argv) {
    Config cfg;
    parse_args(argc, argv, &cfg);

    for (int i = 0; i < cfg.n_rotations; ++i) {
        simulate_rotation(cfg.rotations[i], &cfg);
    }

    return 0;
}
