#!/bin/bash
# Touches all files in your SURREAL scratch directory to reset their
# 30-day access timer and prevent OSCAR from purging them.
#
# Run manually when you haven't used the data in a while:
#   bash oscar/keep_alive.sh
#
# Or add to crontab (runs every 20 days automatically):
#   crontab -e
#   0 9 */20 * * bash ~/virtual-tryon/oscar/keep_alive.sh >> ~/scratch/keep_alive.log 2>&1

DATA_DIR="${HOME}/scratch/surreal_data"

if [[ ! -d "$DATA_DIR" ]]; then
    echo "No data directory found at $DATA_DIR — nothing to do."
    exit 0
fi

echo "[$(date)] Touching files in $DATA_DIR ..."
find "$DATA_DIR" -type f -exec touch {} +
COUNT=$(find "$DATA_DIR" -type f | wc -l)
echo "[$(date)] Touched $COUNT files. Purge timer reset."
