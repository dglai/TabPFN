#!/bin/bash
# Script to run WandB sweep agents in parallel across multiple GPUs using tmux
#
# Usage:
#   ./examples/run_sweep_agents.sh <sweep-id>
#
# Example:
#   ./examples/run_sweep_agents.sh your-entity/your-project/abc123xyz
#
# This will create a tmux session named "wandb-sweep" with 8 windows,
# each running a WandB agent on a different GPU (CUDA_VISIBLE_DEVICES=0-7)

set -e  # Exit on error

# Check if sweep ID is provided
if [ -z "$1" ]; then
    echo "Error: Sweep ID is required"
    echo "Usage: $0 <sweep-id>"
    echo "Example: $0 your-entity/your-project/abc123xyz"
    exit 1
fi

SWEEP_ID="$1"
SESSION_NAME="wandb-sweep"
NUM_WINDOWS=8

# Check if tmux session already exists
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Error: tmux session '$SESSION_NAME' already exists"
    echo "Please attach to it with: tmux attach -t $SESSION_NAME"
    echo "Or kill it with: tmux kill-session -t $SESSION_NAME"
    exit 1
fi

echo "Creating tmux session '$SESSION_NAME' with $NUM_WINDOWS windows..."
echo "Sweep ID: $SWEEP_ID"

# Create new tmux session (detached)
tmux new-session -d -s "$SESSION_NAME" -n "agent-0"

# Run first agent in the initial window (GPU 0)
tmux send-keys -t "$SESSION_NAME:0" "conda activate tabpfn" C-m
tmux send-keys -t "$SESSION_NAME:0" "cd ~/tabpfn" C-m
tmux send-keys -t "$SESSION_NAME:0" "CUDA_VISIBLE_DEVICES=0 wandb agent $SWEEP_ID" C-m

# Create additional windows and run agents (GPU 1-7)
for i in $(seq 1 $((NUM_WINDOWS - 1))); do
    tmux new-window -t "$SESSION_NAME:$i" -n "agent-$i"
    tmux send-keys -t "$SESSION_NAME:$i" "conda activate tabpfn" C-m
    tmux send-keys -t "$SESSION_NAME:$i" "cd ~/tabpfn" C-m
    tmux send-keys -t "$SESSION_NAME:$i" "CUDA_VISIBLE_DEVICES=$i wandb agent $SWEEP_ID" C-m
done

echo ""
echo "✓ Created tmux session '$SESSION_NAME' with $NUM_WINDOWS windows"
echo "✓ Each window is running a WandB agent on a different GPU (0-7)"
echo ""
echo "To attach to the session:"
echo "  tmux attach -t $SESSION_NAME"
echo ""
echo "To navigate between windows:"
echo "  Ctrl+b, then 0-7  (switch to window 0-7)"
echo "  Ctrl+b, n         (next window)"
echo "  Ctrl+b, p         (previous window)"
echo "  Ctrl+b, w         (list all windows)"
echo ""
echo "To detach from session:"
echo "  Ctrl+b, d"
echo ""
echo "To kill the session:"
echo "  tmux kill-session -t $SESSION_NAME"
echo ""
