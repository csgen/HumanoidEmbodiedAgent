#!/bin/bash
#
# recover_results.sh - Recover ML training results from scratch directories
#
# This script finds all completed job directories in /scratch/$USER/
# and copies the results + logs back to the home directory.
#
# Usage:
#   bash recover_results.sh
#   OR
#   bash recover_results.sh --dry-run  # Preview what will be copied
#

set -e

# Configuration
SCRATCH_BASE="/scratch/$USER"
HOME_BASE="/home/svu/$USER/projects-sam/FinSight-QuantLab"
RESULTS_HOME="$HOME_BASE/results"
LOGS_HOME="$HOME_BASE/results/logs"

# Check if dry-run mode
DRY_RUN=0
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=1
    echo "[DRY RUN MODE] No files will be copied"
    echo ""
fi

# Ensure home directories exist
mkdir -p "$RESULTS_HOME" "$LOGS_HOME"

echo "========================================"
echo "ML Training Results Recovery"
echo "========================================"
echo "Searching for job directories in: $SCRATCH_BASE"
echo "Target home directory: $HOME_BASE"
echo ""

# Find all job directories (format: JOBID.pbs01)
# Use find instead of globbing for safer handling of special characters
found_jobs=0
recovered_jobs=0

while IFS= read -r -d '' jobdir; do
    found_jobs=$((found_jobs + 1))
    jobid=$(basename "$jobdir")

    echo "----------------------------------------"
    echo "Job: $jobid"
    echo "Path: $jobdir"

    # Check if results directory exists
    if [ -d "$jobdir/results" ]; then
        echo "  [FOUND] Results directory"

        # Show what will be copied
        echo "  Results contents:"
        find "$jobdir/results" -type f 2>/dev/null | head -5 | sed 's/^/    /'
        file_count=$(find "$jobdir/results" -type f 2>/dev/null | wc -l)
        if [ "$file_count" -gt 5 ]; then
            echo "    ... and $((file_count - 5)) more files"
        fi

        # Copy results to home
        if [ $DRY_RUN -eq 0 ]; then
            rsync -av "$jobdir/results/" "$RESULTS_HOME/"
            echo "  [COPIED] Results → $RESULTS_HOME"
        else
            echo "  [DRY-RUN] Would copy: $jobdir/results/ → $RESULTS_HOME/"
        fi

        recovered_jobs=$((recovered_jobs + 1))
    else
        echo "  [SKIP] No results directory found"
    fi

    # Check for log files
    log_files_found=0

    # Copy train.log
    if [ -f "$jobdir/train.log" ]; then
        log_files_found=$((log_files_found + 1))
        echo "  [FOUND] train.log ($(wc -l < "$jobdir/train.log") lines)"

        if [ $DRY_RUN -eq 0 ]; then
            cp "$jobdir/train.log" "$LOGS_HOME/train_${jobid}.log"
            echo "  [COPIED] train.log → $LOGS_HOME/train_${jobid}.log"
        else
            echo "  [DRY-RUN] Would copy: train.log → $LOGS_HOME/train_${jobid}.log"
        fi
    fi

    # Copy pbs.live.*.out
    for pbslog in "$jobdir"/pbs.live.*.out; do
        if [ -f "$pbslog" ]; then
            log_files_found=$((log_files_found + 1))
            logfile=$(basename "$pbslog")
            echo "  [FOUND] $logfile ($(wc -l < "$pbslog") lines)"

            if [ $DRY_RUN -eq 0 ]; then
                cp "$pbslog" "$LOGS_HOME/$logfile"
                echo "  [COPIED] $logfile → $LOGS_HOME/"
            else
                echo "  [DRY-RUN] Would copy: $logfile → $LOGS_HOME/"
            fi
        fi
    done

    if [ $log_files_found -eq 0 ]; then
        echo "  [SKIP] No log files found"
    fi

    echo ""

done < <(find "$SCRATCH_BASE" -mindepth 1 -maxdepth 1 -type d -print0 2>/dev/null)

# Summary
echo "========================================"
echo "Recovery Summary"
echo "========================================"
echo "Jobs found: $found_jobs"
echo "Jobs with results: $recovered_jobs"

if [ $DRY_RUN -eq 1 ]; then
    echo ""
    echo "[DRY RUN] No files were copied."
    echo "Run without --dry-run to actually copy files:"
    echo "  bash recover_results.sh"
else
    echo ""
    echo "Results copied to: $RESULTS_HOME"
    echo "Logs copied to: $LOGS_HOME"
    echo ""
    echo "Verify results:"
    echo "  ls -lh $RESULTS_HOME/deep/*/*.keras"
    echo "  ls -lh $RESULTS_HOME/ml/*/*.json"
    echo "  ls -lh $LOGS_HOME/*.log"
fi
echo "========================================"
