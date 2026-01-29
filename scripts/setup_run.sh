#!/bin/bash
# Usage: ./scripts/setup_run.sh <commit_sha>
#
# Creates worktree and commit-level structure for a production run
# Example: ./scripts/setup_run.sh $(git rev-parse HEAD)
# Example: ./scripts/setup_run.sh a1b2c3d

set -e

if [ $# -ne 1 ]; then
    echo "Usage: $0 <commit_sha>"
    echo "Example: $0 \$(git rev-parse HEAD)"
    echo "Example: $0 a1b2c3d"
    exit 1
fi

COMMIT_SHA="$1"
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

# Resolve to full SHA if needed, then take first 7 chars
cd "$PROJECT_ROOT/SocialTransformer"
FULL_SHA=$(git rev-parse "$COMMIT_SHA")
SHORT_SHA="${FULL_SHA:0:7}"

echo "Setting up production run for commit: $SHORT_SHA"
echo "Full SHA: $FULL_SHA"
echo ""

# 1. Create worktree
WORKTREE_DIR="$PROJECT_ROOT/SocialTransformer_runs/run_$SHORT_SHA"
if [ -d "$WORKTREE_DIR" ]; then
    echo "Worktree already exists: $WORKTREE_DIR"
else
    echo "Creating worktree..."
    git worktree add "$WORKTREE_DIR" "$FULL_SHA"
    echo "✓ Created worktree: $WORKTREE_DIR"
fi

# 2. Create commit directory structure and commit.yaml
COMMIT_DIR="$PROJECT_ROOT/experiments/by_commit/$SHORT_SHA"
COMMIT_YAML="$COMMIT_DIR/commit.yaml"

# Create runs/ directory (Python scripts will create specific run dirs inside)
mkdir -p "$COMMIT_DIR/runs"
echo "✓ Created commit directory: $COMMIT_DIR"

if [ ! -f "$COMMIT_YAML" ]; then
    echo "Creating commit.yaml..."
    
    # Get commit metadata
    COMMIT_MSG=$(git log -1 --format=%s "$FULL_SHA")
    COMMIT_DATE=$(git log -1 --format=%ci "$FULL_SHA")
    AUTHOR=$(git log -1 --format="%an <%ae>" "$FULL_SHA")
    
    # Write commit.yaml
    cat > "$COMMIT_YAML" << EOF
full_sha: $FULL_SHA
short_sha: $SHORT_SHA
commit_message: |
  $COMMIT_MSG
commit_date: $COMMIT_DATE
author: $AUTHOR
created: $(date -Iseconds)
EOF
    echo "✓ Created commit.yaml"
else
    echo "commit.yaml already exists"
fi

# 3. Create symlinks in worktree
echo "Setting up symlinks..."
cd "$WORKTREE_DIR"

# Data symlink
if [ ! -L "data" ]; then
    ln -s ../../data data
    echo "✓ Created data symlink"
fi

# Current_run symlink (points to commit-level directory)
if [ -L "current_run" ] || [ -e "current_run" ]; then
    rm -f current_run
fi
ln -s ../../experiments/by_commit/$SHORT_SHA current_run
echo "✓ Created current_run symlink -> experiments/by_commit/$SHORT_SHA"

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo "Worktree: $WORKTREE_DIR"
echo "Experiments: $COMMIT_DIR"
echo ""
echo "Next steps:"
echo "  cd $WORKTREE_DIR"
echo "  ./scripts/run_training.sh"
echo ""
