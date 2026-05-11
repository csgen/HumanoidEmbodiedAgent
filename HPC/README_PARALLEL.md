# Parallel ML Backtesting - Fast Execution Guide

## Overview

This directory contains a **massively parallel** backtesting system that runs **8 strategies simultaneously** on HPC resources.

### Key Features
- ✅ **True Parallelism**: All strategies run in separate processes
- ✅ **Resource Efficient**: 36 cores / 8 strategies = 4-5 cores per strategy
- ✅ **GPU Sharing**: Batched inference allows all models to share 1 GPU
- ✅ **Fault Tolerant**: Each strategy is independent; failures don't block others
- ✅ **Fast Execution**: Estimated **15-20 minutes** total (vs 2+ hours sequential)

## Quick Start

### 1. On HPC, navigate to project directory:
```bash
cd /nfs/home/svu/e1538626/projects-sam/FinSight-QuantLab
```

### 2. Submit the parallel job:
```bash
bash src/hpc/submit_backtest.sh
```

### 3. Monitor progress:
```bash
# Overall job status
qstat -u $USER

# Main job log (aggregated)
tail -f logs/ml_backtest_parallel.log

# Individual strategy logs
tail -f logs/backtest_siamese.log
tail -f logs/backtest_xgboost.log
tail -f logs/backtest_clustering.log
```

## Architecture

### File Structure
```
src/ml_backtest/
├── backtest_single_strategy.py  # Single strategy runner (for parallel exec)
├── run_ml_backtest.py           # Sequential runner (backup)
└── README_PARALLEL.md           # This file

src/hpc/
├── ml_backtest_parallel.pbs     # PBS job script (parallel execution)
└── submit_backtest.sh           # Quick submission script

results/ml_backtest/              # Output directory
├── distance_pairs.csv
├── distance_metrics.csv
├── distance_portfolio_pnl.csv
├── siamese_lstm_pairs.csv
├── siamese_lstm_metrics.csv
├── siamese_lstm_portfolio_pnl.csv
├── ... (all 8 strategies)
├── summary_table.csv            # Combined performance metrics
└── execution_summary.txt        # Job execution report
```

### Strategies Executed

| # | Strategy | Type | Pair Selection | Signal Generation |
|---|----------|------|----------------|-------------------|
| 1 | Distance | Statistical | SSD | Z-score threshold |
| 2 | Cointegration | Statistical | ADF test | Z-score threshold |
| 3 | TimeSeries | Statistical | OU half-life | Z-score threshold |
| 4 | Clustering | ML | DBSCAN | Z-score threshold |
| 5 | Siamese LSTM | Deep Learning | Neural network | Z-score threshold |
| 6 | XGBoost | ML | Gradient boosting | Z-score threshold |
| 7 | LSTM Spread | Deep Learning | Distance | LSTM forecast |
| 8 | Transformer | Deep Learning | Distance | Transformer forecast |

### Resource Allocation

**Total Resources** (from PBS):
- 36 CPU cores
- 1 A40 GPU (48GB VRAM)
- 240GB RAM
- 2 hour walltime

**Per-Strategy Allocation**:
- ~4-5 CPU cores (via `n_jobs=4` in config)
- Shared GPU access (batched inference)
- ~30GB RAM budget per strategy
- Independent process space

### Parallel Execution Flow

```
PBS Job Start
│
├── Load modules (CUDA, Conda, GCC)
├── Activate venv
├── Stage code to /scratch
│
├── Launch Strategy 1 (distance)      → Background PID 12345
├── Launch Strategy 2 (cointegration) → Background PID 12346
├── Launch Strategy 3 (timeseries)    → Background PID 12347
├── Launch Strategy 4 (clustering)    → Background PID 12348
├── Launch Strategy 5 (siamese)       → Background PID 12349
├── Launch Strategy 6 (xgboost)       → Background PID 12350
├── Launch Strategy 7 (lstm_spread)   → Background PID 12351
└── Launch Strategy 8 (transformer)   → Background PID 12352
    │
    ├── Wait for all PIDs
    ├── Collect exit codes
    ├── Copy results from /scratch to /nfs
    ├── Generate summary_table.csv
    └── Cleanup
```

## Expected Runtime

| Strategy | Estimated Time | Bottleneck |
|----------|----------------|------------|
| Distance | 2-3 min | SSD computation (parallelized) |
| Cointegration | 1-2 min | ADF tests (parallelized) |
| TimeSeries | 1-2 min | OU fitting (parallelized) |
| Clustering | 3-5 min | Correlation matrix + DBSCAN |
| Siamese LSTM | 5-10 min | Batched GPU inference |
| XGBoost | 5-7 min | Feature engineering (parallelized) |
| LSTM Spread | 3-5 min | Batched GPU inference |
| Transformer | 3-5 min | Batched GPU inference |

**Total (parallel)**: **~10-15 minutes** (all run simultaneously)
**Total (sequential)**: **~2-3 hours** (if run one-by-one)

**Speedup**: **8-12x faster**

## Output Files

Each strategy generates 3 files:

### 1. `{strategy}_pairs.csv`
```csv
a,b,beta
RELIANCE,TCS,1.234
INFY,WIPRO,0.987
...
```

### 2. `{strategy}_metrics.csv`
```csv
metric,value
sharpe,0.9221
cagr,0.2190
ann_vol,0.2481
max_dd,0.5114
turnover,63.0
```

### 3. `{strategy}_portfolio_pnl.csv`
```csv
date,portfolio_ret
2020-01-01,0.0023
2020-01-02,-0.0015
...
```

### 4. `summary_table.csv` (aggregated)
```csv
Strategy,Sharpe,CAGR,AnnVol,MaxDD,Turnover
Distance,0.9221,0.2190,0.2481,0.5114,63.0
Siamese Lstm,0.8543,0.1987,0.2301,0.4523,71.0
...
```

## Troubleshooting

### Job fails immediately
```bash
# Check PBS output
cat logs/ml_backtest_parallel.log

# Common issues:
# 1. Virtual environment not found
source ~/.venvs/finsight-tf211-cuda117/bin/activate

# 2. Missing data
ls -lh data/panel_live.parquet

# 3. Missing models
ls -lh results/ml_results/*/
```

### Individual strategy fails
```bash
# Check strategy-specific log
cat logs/backtest_siamese.log

# Test strategy locally
python src/ml_backtest/backtest_single_strategy.py --strategy siamese
```

### GPU out of memory
```bash
# Reduce batch size in siamese_lstm.py line 622
# Change from 512 to 256 or 128
```

### No progress for long time
```bash
# Check if GPU is being used
ssh to HPC node
nvidia-smi

# Check CPU utilization
htop
```

## Model Path Configuration

The script expects models in:
```
results/ml_results/
├── siamese/
│   └── siamese_lstm_best.keras
├── lstm_spread/
│   └── lstm_spread_best.keras
├── transformer/
│   └── transformer_spread_best.keras
└── xgboost/
    └── xgboost_model.json
```

If models are in different locations, update `CONFIG` in `backtest_single_strategy.py` lines 89-92.

## Performance Tuning

### To reduce runtime further:

1. **Reduce `top_n_pairs`** (line 72):
   - `40 → 20` pairs: ~2x faster
   - Trade-off: Less diversification

2. **Reduce `formation_days`** (line 71):
   - `252 → 126` days: ~1.5x faster
   - Trade-off: Less data for pair selection

3. **Increase `n_jobs`** per strategy (line 74):
   - `4 → 6` cores: ~1.3x faster
   - Trade-off: More contention (only 36 cores total)

4. **Skip slow strategies**:
   - Comment out `xgboost` or `transformer` in PBS script
   - Focus on best performers: Distance, Siamese, Clustering

## Next Steps After Completion

1. **View Results**:
   ```bash
   cat results/ml_backtest/summary_table.csv
   ```

2. **Generate Visualizations**:
   ```bash
   python src/visualization/generate_ml_html.py
   ```

3. **Analyze in Jupyter**:
   ```bash
   jupyter notebook notebooks/ml_results_analysis.ipynb
   ```

4. **Update Dissertation**:
   - Copy metrics from `summary_table.csv`
   - Use PnL curves from `*_portfolio_pnl.csv`
   - Reference pairs from `*_pairs.csv`

## Emergency: Cancel Job

```bash
# Get job ID
qstat -u $USER

# Cancel
qdel <JOB_ID>
```

## Emergency: Run Sequential (Backup)

If parallel execution has issues, fallback to sequential:

```bash
python src/ml_backtest/run_ml_backtest.py
```

This runs all strategies one-by-one (slower but more stable).

---

**Last Updated**: 2025-10-31
**Author**: Claude Code + Sam
**Contact**: Check logs in `logs/ml_backtest_parallel.log`
