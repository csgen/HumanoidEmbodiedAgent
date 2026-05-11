#!/bin/bash
# Download trained ML models from HPC to local machine

# Configuration
HPC_USER="your_username"  # Replace with your HPC username
HPC_HOST="stdct-mgmt-02"  # Or your HPC login node
HPC_PROJECT_DIR="~/projects-sam/FinSight-QuantLab"
LOCAL_RESULTS_DIR="./results"

echo "================================================"
echo "Downloading ML Models from HPC"
echo "================================================"

# Create local directories
mkdir -p "$LOCAL_RESULTS_DIR/deep/siamese_lstm"
mkdir -p "$LOCAL_RESULTS_DIR/deep/lstm_spread"
mkdir -p "$LOCAL_RESULTS_DIR/deep/transformer_spread"
mkdir -p "$LOCAL_RESULTS_DIR/ml/pair_scoring"

# Download Siamese LSTM
echo -e "\n[1/4] Downloading Siamese LSTM..."
scp "$HPC_USER@$HPC_HOST:$HPC_PROJECT_DIR/results/deep/siamese_lstm_best.keras" \
    "$LOCAL_RESULTS_DIR/deep/siamese_lstm/" || echo "  Warning: siamese_lstm_best.keras not found"

scp "$HPC_USER@$HPC_HOST:$HPC_PROJECT_DIR/results/deep/train_log.csv" \
    "$LOCAL_RESULTS_DIR/deep/siamese_lstm/" || echo "  Warning: train_log.csv not found"

# Download LSTM Spread
echo -e "\n[2/4] Downloading LSTM Spread..."
scp "$HPC_USER@$HPC_HOST:$HPC_PROJECT_DIR/results/deep/lstm_spread/lstm_spread_best.keras" \
    "$LOCAL_RESULTS_DIR/deep/lstm_spread/" || echo "  Warning: lstm_spread_best.keras not found"

scp "$HPC_USER@$HPC_HOST:$HPC_PROJECT_DIR/results/deep/lstm_spread/training_history.csv" \
    "$LOCAL_RESULTS_DIR/deep/lstm_spread/" || echo "  Warning: training_history.csv not found"

# Download Transformer Spread
echo -e "\n[3/4] Downloading Transformer Spread..."
scp "$HPC_USER@$HPC_HOST:$HPC_PROJECT_DIR/results/deep/transformer_spread/transformer_spread_best.keras" \
    "$LOCAL_RESULTS_DIR/deep/transformer_spread/" || echo "  Warning: transformer_spread_best.keras not found"

scp "$HPC_USER@$HPC_HOST:$HPC_PROJECT_DIR/results/deep/transformer_spread/training_history.csv" \
    "$LOCAL_RESULTS_DIR/deep/transformer_spread/" || echo "  Warning: training_history.csv not found"

# Download XGBoost (if it completed)
echo -e "\n[4/4] Downloading XGBoost Pair Scoring..."
scp "$HPC_USER@$HPC_HOST:$HPC_PROJECT_DIR/results/ml/pair_scoring/xgboost_model.json" \
    "$LOCAL_RESULTS_DIR/ml/pair_scoring/" || echo "  Warning: xgboost_model.json not found (may not have completed)"

scp "$HPC_USER@$HPC_HOST:$HPC_PROJECT_DIR/results/ml/pair_scoring/feature_importance.csv" \
    "$LOCAL_RESULTS_DIR/ml/pair_scoring/" || echo "  Warning: feature_importance.csv not found"

scp "$HPC_USER@$HPC_HOST:$HPC_PROJECT_DIR/results/ml/pair_scoring/training_config.json" \
    "$LOCAL_RESULTS_DIR/ml/pair_scoring/" || echo "  Warning: training_config.json not found"

echo -e "\n================================================"
echo "Download Complete!"
echo "================================================"
echo "Models saved to: $LOCAL_RESULTS_DIR"
echo ""
echo "Next steps:"
echo "1. Verify models: ls -lh results/deep/*/*.keras"
echo "2. Run backtesting: python src/ml_backtest/backtest_all_ml.py"
echo "3. Generate visualizations: jupyter notebook notebooks/ml_results_analysis.ipynb"
echo "================================================"
