#!/bin/bash

# The remaining hybrid topologies for the AblationNet study
PATTERNS=("GGGGGG" "GGGAAA" "AAAGGG" "GAGAGA" "AGAGAG")

# We keep the optimized chunk size for the Delta layers
# Trial and error for the MPS M2 Pro GPU
CHUNK_SIZE=64

echo "🧬 Starting AblationNet Batch Run..."

for pattern in "${PATTERNS[@]}"; do
    echo "=================================================="
    echo "🚀 Launching Experiment: tiny-$pattern"
    echo "=================================================="
    
    # Run the training script for the current pattern
    uv run python ../training.py --pattern "$pattern" --chunksize "$CHUNK_SIZE"
    
    echo "✅ Finished $pattern!"
    echo "Sleeping for 30 seconds to let the GPU cool and WandB sync..."
    sleep 30
done

echo "🎉 All AblationNet hybrid runs are complete!"