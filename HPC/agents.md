# HPC-Related Transferable Knowledge (Extracted from FinSight-QuantLab)

This document contains consolidated high-performance computing (HPC) patterns, workflows, and technical configurations extracted from the FinSight-QuantLab project. These are intended to be transferable to the HumanoidEmbodiedAgent project.

## 1. HPC Infrastructure & Job Management (PBS/Slurm)

### 1.1 Resource Requests & Orchestration
- **Standard Research Node Request:**
  - 36-48 CPU cores.
  - 240-384 GB RAM.
  - 1x NVIDIA A40 GPU (or equivalent).
  - 12-24h walltime.
- **Job Orchestration:** Use `.pbs` or `.slurm` scripts for batch processing.
- **Dynamic Overrides:** Use `qsub -v` (for PBS) to pass environment variables (e.g., `EPOCHS`, `BATCH_SIZE`, `RESUME`) without modifying scripts.

### 1.2 Scratch I/O Workflow (Performance Critical)
To avoid hammering the shared network filesystem (NFS) and achieve 10-100x faster I/O (local SSD/NVMe speeds of 1-5 GB/s):

1. **Setup Scratch Directory:**
   ```bash
   SCRATCH=/scratch/${USER}/${PBS_JOBID}
   mkdir -p $SCRATCH
   ```

2. **Stage-In (Local Terminal / Job Script):**
   Copy code and data from persistent home/project storage to the node-local scratch.
   ```bash
   # Inside PBS script
   rsync -av ~/projects/FinSight-QuantLab/data/panel_live.parquet $SCRATCH/
   rsync -av ~/projects/FinSight-QuantLab/src $SCRATCH/
   ```

3. **Execution on Scratch:**
   Always `cd $SCRATCH` before running training to ensure all temporary files and checkpoints are written to the fast local disk.
   ```bash
   cd $SCRATCH
   python -m src.ml_train.train_siamese --data_path $SCRATCH/data.parquet --out_dir $SCRATCH/results
   ```

4. **Stage-Out (Job Script):**
   Sync results back to persistent storage before the job ends.
   ```bash
   rsync -av $SCRATCH/results/ ~/projects/FinSight-QuantLab/results/
   ```

5. **Cleanup:**
   ```bash
   rm -rf $SCRATCH
   ```

### 1.3 Local-to-HPC Sync (Local Terminal Workflow)
To keep your local development environment in sync with the HPC cluster:
- **Pushing Code/Data:**
  ```bash
  rsync -avz --exclude='.git' --exclude='__pycache__' ./path/to/local/src user@hpc-cluster:~/projects/target/src
  ```
- **Pulling Results:**
  ```bash
  rsync -avz user@hpc-cluster:~/projects/target/results/ ./local_results/
  ```

### 1.4 Threading & Environment Configuration
- **Thread Tuning (Python/TF/OMP):**
  Manage thread contention to avoid system hang and maximize throughput.
  ```bash
  export TF_NUM_INTRAOP_THREADS=36  # Match requested ncpus
  export TF_NUM_INTEROP_THREADS=1   # Reduce context switching
  export OMP_NUM_THREADS=36
  ```
- **Module Loading:**
  ```bash
  module load TensorFlow/2.11.0-foss-2021b-CUDA-11.7.0
  module load Python/3.9.6-GCCcore-11.2.0
  ```

---

## 2. Training Stability & Troubleshooting

### 2.1 Common HPC Issues & Solutions
- **Out-of-Memory (OOM):**
  - **Symptoms:** Job killed with "Exceeded memory limit".
  - **Fixes:** Reduce `batch_size`, use `pd.read_parquet(columns=[...])` to save RAM, or request more memory (`#PBS -l mem=384gb`).
- **Numerical Instability (NaN Loss):**
  - **Symptoms:** Loss becomes NaN after few epochs.
  - **Fixes:** Use `Adam(clipnorm=1.0)`, add `TerminateOnNaN` callback, and ensure input data is winsorized (e.g., ±20% return clipping).

### 2.2 Local Testing (Pre-Submission)
Always run a "smoke test" locally before submitting to the cluster to save queue time:
1. **Data Load Test:** Verify parquet files load correctly.
2. **Architecture Test:** Run 1 epoch on a small CPU-based sample.
3. **Checkpoints:** Verify `best_model.keras` and `last_model.keras` are created.

---

## 3. Parallelism Patterns

### 2.1 CPU Parallelism (Joblib/Multiprocessing)
- **Embarrassingly Parallel Tasks:** Use `joblib` for scoring, backtesting, or feature engineering across large sets of independent pairs/tasks.
  ```python
  from joblib import Parallel, delayed
  results = Parallel(n_jobs=16)(delayed(process_task)(item) for item in items)
  ```
- **Configuration:** Set `n_jobs` based on the requested cores (e.g., 16-48).

### 2.2 Deep Learning Parallelism (TensorFlow/PyTorch)
- **Thread Tuning:** Manage thread contention by explicitly setting intra/inter-op threads.
  - `TF_NUM_INTRAOP_THREADS`: Set to number of requested cores.
  - `TF_NUM_INTEROP_THREADS`: Typically set to 1 to reduce context switching overhead.
- **GPU Optimization:** Use mixed precision training and `tf.data` pipelines with `prefetch(tf.data.AUTOTUNE)` and `cache()` to prevent data loading bottlenecks.

---

## 3. Training Stability & Resume Capability

### 3.1 Checkpoint Management
- **Dual Checkpointing:**
  - `best_model.keras`: Saved based on validation metrics (e.g., max `val_AUC` or min `val_loss`).
  - `last_model.keras`: Saved at the end of every epoch to enable resuming after pre-emption or crashes.
- **Resumption Logic:**
  - Check for existing checkpoints and `train_log.csv`.
  - Load the last checkpoint + optimizer state.
  - Set `initial_epoch` in `model.fit()` to ensure learning rate schedulers and logging remain continuous.

### 3.2 Stability Guards
- **Gradient Clipping:** Use `clipnorm=1.0` in optimizers to prevent exploding gradients in recurrent or transformer architectures.
- **TerminateOnNaN:** Always include a callback to fail fast if numerical instability occurs, preventing wasted compute hours.
- **Safe Returns:** Implement winsorization or clipping for input features and labels (e.g., clipping returns at ±20-30%) to handle outliers in financial or noisy sensor data.

---

## 4. Environment & Reproducibility
- **Deterministic Execution:** Always set fixed seeds for `random`, `numpy`, and `tensorflow/pytorch`.
- **Environment Snapshots:** Maintain a `requirements.txt` or `environment.yml` specific to the HPC module versions (e.g., TF 2.11 + CUDA 11.7).
- **Auditability:** Every experiment should generate:
  - A metrics summary (`*_metrics.csv`).
  - A training log (`train_log.csv`).
  - Checkpoint artifacts.

---

## 5. KDB+/Q Integration (Planned Performance)
- **High-Speed Hot Paths:** Migrate data-intensive components to `kdb+` for 10-100x speedups compared to Python/Pandas for large time-series datasets.
- **Interop:** Use `qpython` or IPC bridges for Python-KDB communication.
