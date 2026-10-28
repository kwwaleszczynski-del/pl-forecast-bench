#!/usr/bin/env bash
# Rebuild the benchmark data and every reference result from scratch.
# Data go to $PLFB_DATA (default ./data). Expect about 2 hours for the weather download (846 requests to the
# Open-Meteo Single Runs API, about 6,800 of the 10,000 daily API units), a few minutes for PSE and about 1.5 hours
# of CPU for the reference models.
set -euo pipefail

python -m plfb pse
python -m plfb nwp
python -m plfb build
python -m plfb verify          # compares your download with benchmark/checksums.json, month by month

python -m plfb baselines --splits val test
for m in persistence naive_day naive_week pse_plan ridge gbm; do
  python -m plfb evaluate --forecast "${PLFB_DATA:-data}/forecasts/$m/test.parquet" --name "$m" \
    --meta "submissions/$m/meta.json" --out "submissions/$m"
done

# optional: zero-shot foundation models (pip install -e ".[foundation]"; downloads PyTorch and model weights)
# python -m plfb foundation --models chronos_bolt chronos2 chronos2_nwp timesfm toto --device cuda
# for m in chronos_bolt chronos2 chronos2_nwp timesfm toto; do
#   python -m plfb evaluate --forecast "${PLFB_DATA:-data}/forecasts/$m/test.parquet" --name "$m" \
#     --meta "submissions/$m/meta.json" --out "submissions/$m"
# done

python -m plfb leaderboard
