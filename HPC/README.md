# HPC Job Scripts

PBS job scripts for training ML models on HPC cluster.

## Prerequisites

1. **TensorFlow Module**: `TensorFlow/2.11.0-foss-2022a-CUDA-11.7.0`
2. **Python Virtual Environment**: `$HOME/.venvs/finsight-tf211-cuda117`
3. **Project Path**: `$HOME/projects-sam/FinSight-QuantLab`

## Setup Virtual Environment (One-time)

```bash
# Load TensorFlow module
module purge
module load TensorFlow/2.11.0-foss-2022a-CUDA-11.7.0

# Create virtual environment
python -m venv $HOME/.venvs/finsight-tf211-cuda117

# Activate and install dependencies
source $HOME/.venvs/finsight-tf211-cuda117/bin/activate
pip install --upgrade pip
pip install pandas pyarrow  # Add other dependencies as needed
```

## Siamese LSTM Training

### Submit Job

```bash
# Basic submission (use default config)
qsub src/hpc/siamese_lstm.pbs

# Custom epochs
qsub -v EPOCHS=20 src/hpc/siamese_lstm.pbs

# Custom batch size and learning rate
qsub -v BATCH_SIZE=256,LR=0.0005 src/hpc/siamese_lstm.pbs

# Resume from checkpoint
qsub -v RESUME=auto src/hpc/siamese_lstm.pbs

# Resume from specific checkpoint
qsub -v RESUME=/path/to/model.keras src/hpc/siamese_lstm.pbs
```

### Monitor Job

```bash
# Check job status
qstat -u $USER

# Monitor live output (while job is running)
tail -f /scratch/$USER/<JOB_ID>/pbs.live.<JOB_ID>.out

# View logs after completion
ls -lh results/logs/
cat results/logs/pbs.live.<JOB_ID>.out
```

### Outputs

- **Checkpoints**: `results/deep/siamese_lstm_best.keras`, `siamese_lstm_last.keras`
- **Training Log**: `results/deep/train_log.csv`
- **Config**: `results/deep/train_config.json`
- **Job Logs**: `results/logs/pbs.live.<JOB_ID>.out`

## Customization

### Edit PBS Script

Key sections to customize in `siamese_lstm.pbs`:

1. **Resources** (line 10):
   ```bash
   #PBS -l select=1:ncpus=36:mem=240gb:ngpus=1
   ```

2. **Walltime** (line 8):
   ```bash
   #PBS -l walltime=12:00:00
   ```

3. **Training Params** (lines 28-32):
   ```bash
   export EPOCHS="${EPOCHS:-8}"
   export BATCH_SIZE="${BATCH_SIZE:-128}"
   export LEARNING_RATE="${LR:-0.001}"
   ```

4. **Threading** (lines 22-28):
   ```bash
   export OMP_NUM_THREADS=36
   export TF_NUM_INTRAOP_THREADS=36
   export TF_NUM_INTEROP_THREADS=4
   ```

### Use Config File

Create `my_config.json`:
```json
{
  "window": 128,
  "stride": 10,
  "holdout_frac": 0.2,
  "ticker_limit": 200,
  "pairs_per_window": 5000,
  "thresh_r": 0.7,
  "hidden": 64,
  "proj": 64,
  "batch_size": 128,
  "epochs": 10,
  "learning_rate": 0.001,
  "seed": 42
}
```

Then submit:
```bash
qsub -v CONFIG=my_config.json src/hpc/siamese_lstm.pbs
```

## Troubleshooting

### No GPUs Visible

Check CUDA module is loaded:
```bash
module list | grep -i cuda
nvidia-smi
```

### Out of Memory

Reduce batch size or ticker limit:
```bash
qsub -v BATCH_SIZE=64 src/hpc/siamese_lstm.pbs
```

### Slow Training

- Verify GPU utilization: `nvidia-smi dmon`
- Enable XLA JIT in PBS script (line 29)
- Reduce data loading overhead: increase `ticker_limit` stride

### Job Killed Early

Increase walltime:
```bash
#PBS -l walltime=24:00:00
```

## Best Practices

1. **Test Locally First**: Run `python src/ml_train/train_siamese.py --epochs 1` on login node
2. **Use Resume**: Long training? Use `--resume auto` to checkpoint and continue
3. **Monitor Early**: Check first few epochs, kill if training diverges
4. **Log Everything**: Live logs in `/scratch` let you debug without waiting for job completion
5. **Clean Scratch**: Old scratch dirs auto-purge after ~60 days, but clean manually if disk-limited
