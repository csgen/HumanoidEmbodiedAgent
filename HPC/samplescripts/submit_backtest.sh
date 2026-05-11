#!/bin/bash
################################################################################
# Quick Submit Script for ML Backtesting
################################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PBS_SCRIPT="${SCRIPT_DIR}/ml_backtest_parallel.pbs"

echo "========================================================================"
echo "Submitting ML Backtest Job"
echo "========================================================================"
echo "PBS Script: ${PBS_SCRIPT}"
echo ""

# Submit job
JOB_ID=$(qsub "${PBS_SCRIPT}")

echo "✓ Job submitted: ${JOB_ID}"
echo ""
echo "Monitor with:"
echo "  qstat -u \$USER"
echo "  tail -f logs/ml_backtest_parallel.log"
echo "  tail -f logs/backtest_siamese.log"
echo ""
echo "Check status:"
echo "  qstat -f ${JOB_ID}"
echo ""
echo "Cancel if needed:"
echo "  qdel ${JOB_ID}"
echo "========================================================================"
