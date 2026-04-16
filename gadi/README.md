# Gadi Run Notes

This directory contains the Gadi-specific files for the serial, non-interactive
version of `code/model_spiking_cat_90vs180.py`.

## Files

- `model_spiking_cat_90vs180_gadi.pbs`: PBS job script for Gadi
- `setup_model_spiking_cat_90vs180_env.sh`: create a local virtual environment
- `submit_model_spiking_cat_90vs180_gadi.sh`: convenience wrapper for `qsub`
- `model_spiking_cat_90vs180_gadi_requirements.txt`: Python dependencies

## First-time setup on Gadi

From the repo root:

```bash
bash gadi/setup_model_spiking_cat_90vs180_env.sh
```

This creates:

```bash
.venv-model_spiking_cat_90vs180_gadi
```

## Submit the job

From the repo root:

```bash
bash gadi/submit_model_spiking_cat_90vs180_gadi.sh
```

Or directly:

```bash
qsub gadi/model_spiking_cat_90vs180_gadi.pbs
```

## Outputs

- `output/model_spiking_cat_90vs180_gadi_*.npy`
- `output/model_spiking_cat_90vs180_gadi_*.csv`
- `logs/model_spiking_cat_90vs180_gadi.out`
- `logs/model_spiking_cat_90vs180_gadi.err`

## Optional saved figures

The PBS job currently runs simulation only. To also write PNG figures, edit
`gadi/model_spiking_cat_90vs180_gadi.pbs` and add:

```bash
--plot
```

to the Python command.
