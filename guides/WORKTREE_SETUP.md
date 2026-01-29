# Git Worktree Setup for Social Bandits Project

This document explains the git worktree-based workflow for managing experiments in the Social Bandits Project.

## Directory Structure

```
SocialBanditsProject/
  data/                              # Shared datasets
  
  SocialTransformer/                 # Main worktree for development
    data -> ../data/                 # Symlink to shared data
    current_run -> ../experiments/dev/  # Symlink to dev experiments
    src/
    scripts/
    
  SocialTransformer_runs/            # Production run worktrees
    run_<short_sha>/                 # Code snapshot at specific commit
      data -> ../../data/
      current_run -> ../../experiments/by_commit/<short_sha>/
      
  experiments/
    dev/                             # Development runs
      runs/
        wandb_<run_id>/              # or uuid_<10char>
          checkpoints/
          wandb/
          logs/
          results/
          config.yaml
          run.yaml
    by_commit/                       # Production runs
      <short_sha>/                   # e.g., a1b2c3d
        commit.yaml                  # Commit-level metadata
        runs/
          wandb_<run_id>/            # or uuid_<10char>
            checkpoints/
            wandb/
            logs/
            results/
            config.yaml
            run.yaml
```

## Quick Start

### Development Workflow

For iterating on code and running quick experiments:

```bash
cd SocialTransformer

# Run any script - it automatically creates run directories
python src/prior_experience_analysis.py --quick_test
# Creates: experiments/dev/runs/uuid_<id>/

./scripts/run_training.sh
# Creates: experiments/dev/runs/wandb_<id>/

# All outputs automatically organized in current_run/runs/<run_id>/
```

### Production Workflow

For reproducible experiments with code snapshots:

```bash
# 1. Create worktree for specific commit
./scripts/setup_run.sh $(git rev-parse HEAD)
# or use short SHA: ./scripts/setup_run.sh a1b2c3d

# This creates:
#   - SocialTransformer_runs/run_a1b2c3d/
#   - experiments/by_commit/a1b2c3d/commit.yaml
#   - Symlinks in the worktree

# 2. Run experiment in the worktree
cd SocialTransformer_runs/run_a1b2c3d
./scripts/run_training.sh

# Creates: experiments/by_commit/a1b2c3d/runs/wandb_<id>/
# All outputs go to current_run/runs/<run_id>/
```

## Concepts

### Worktrees

Git worktrees allow you to have multiple working directories from the same repository. This means:

- `SocialTransformer/` - main development branch
- `SocialTransformer_runs/run_a1b2c3d/` - snapshot at commit a1b2c3d
- Each worktree is isolated - changes in one don't affect the others
- Perfect for running long experiments while continuing development

### Run IDs

Run directories are automatically named based on context:

- **With wandb**: `wandb_<run_id>` (e.g., `wandb_2k5j8x9m`)
  - Used by `train.py` when `--use_wandb` is enabled
  - Run ID from wandb.init()
  
- **Without wandb**: `uuid_<10char>` (e.g., `uuid_a3f7k9m2p1`)
  - Used by `eval.py`, `inference.py`, `prior_experience_analysis.py`
  - Also used by `train.py` when `--use_wandb` is not set
  - Generated UUID for unique identification

### Output Directories

All scripts write to `current_run/runs/<run_id>/`:

- `checkpoints/` - Model checkpoints
- `wandb/` - Wandb logs (if using wandb)
- `logs/` - Training logs, pip freeze, etc.
- `results/` - Evaluation results, figures, analysis outputs
- `derived/` - Derived data or intermediate outputs
- `config.yaml` - Experiment configuration
- `run.yaml` - Run metadata

The `current_run` symlink points to either:
- `experiments/dev/` for development
- `experiments/by_commit/<sha>/` for production

## Detailed Workflows

### Creating a Production Run Worktree

```bash
# Get commit SHA (or use short SHA)
COMMIT_SHA=$(git rev-parse HEAD)
SHORT_SHA="${COMMIT_SHA:0:7}"

# Create worktree and directory structure
./scripts/setup_run.sh $COMMIT_SHA

# The script creates:
# 1. Worktree at SocialTransformer_runs/run_<short_sha>/
# 2. Commit directory at experiments/by_commit/<short_sha>/
# 3. commit.yaml with commit metadata
# 4. Symlinks: data and current_run

# Navigate to worktree
cd SocialTransformer_runs/run_$SHORT_SHA

# Verify symlinks
ls -la data           # -> ../../data/
ls -la current_run    # -> ../../experiments/by_commit/<short_sha>/
```

### Running Multiple Experiments at Same Commit

```bash
cd SocialTransformer_runs/run_a1b2c3d

# First run
./scripts/run_training.sh
# Creates: experiments/by_commit/a1b2c3d/runs/wandb_<id1>/

# Second run (different hyperparameters)
./scripts/run_training.sh
# Creates: experiments/by_commit/a1b2c3d/runs/wandb_<id2>/

# All runs share the same commit.yaml but have separate run directories
```

### Finding and Comparing Experiments

```bash
# All dev experiments
ls experiments/dev/runs/

# All production experiments for a specific commit
ls experiments/by_commit/a1b2c3d/runs/

# All commits with experiments
ls experiments/by_commit/

# View commit metadata
cat experiments/by_commit/a1b2c3d/commit.yaml

# View run metadata
cat experiments/by_commit/a1b2c3d/runs/wandb_2k5j8x9m/run.yaml

# Compare runs from same commit
for run in experiments/by_commit/a1b2c3d/runs/*/; do
  echo "=== $(basename $run) ==="
  cat $run/run.yaml | grep status
done
```

### Cleaning Up Worktrees

```bash
# List all worktrees
git worktree list

# Remove a worktree
git worktree remove SocialTransformer_runs/run_a1b2c3d

# Note: This only removes the code worktree
# Experiment outputs in experiments/by_commit/ are preserved
```

## Metadata Files

### commit.yaml

Created by `setup_run.sh` in `experiments/by_commit/<sha>/`:

```yaml
full_sha: <40-character SHA>
short_sha: <7-character SHA>
commit_message: |
  <commit message>
commit_date: <ISO 8601 timestamp>
author: Name <email>
created: <ISO 8601 timestamp>
```

### run.yaml

Created by Python scripts when run starts:

```yaml
run_id: wandb_<id> or uuid_<id>
script: train.py
command: <full command line>
dataset_path: <path>
wandb_url: <url if using wandb>
wandb_id: <id if using wandb>
start_time: <ISO 8601 timestamp>
end_time: <ISO 8601 timestamp>
status: running|completed|failed|interrupted
error: <error message if failed>
```

### config.yaml

Created by Python scripts with argparse arguments:

```yaml
# All command-line arguments saved
batch_size: 128
learning_rate: 0.0001
epochs: 500
# ... etc
```

## Tips and Best Practices

### Development Tips

1. **Use `SocialTransformer/` for iteration**
   - Fast feedback loop
   - No need to create worktrees
   - All outputs go to `experiments/dev/`

2. **Commit frequently**
   - Each development iteration can be saved
   - Easy to create production runs from any commit

3. **Clean up dev runs periodically**
   ```bash
   # Dev runs are temporary - safe to delete old ones
   rm -rf experiments/dev/runs/uuid_*
   rm -rf experiments/dev/runs/wandb_*
   ```

### Production Tips

1. **Create worktrees for important experiments**
   - Long training runs
   - Hyperparameter sweeps
   - Final evaluation runs
   - Anything you want to reproduce exactly

2. **Reuse worktrees**
   - Same commit can have multiple runs
   - No need to recreate worktree for each run
   - Just run the script again

3. **Document in notes**
   - Add notes to run.yaml manually if needed
   - Helps remember what each run was testing

4. **Organize by commit**
   - All experiments at a commit are grouped together
   - Easy to compare different hyperparameters
   - Clear history of what was tried

### Debugging Tips

1. **Check symlinks**
   ```bash
   ls -la data         # Should point to ../../data/
   ls -la current_run  # Should point to experiments/...
   ```

2. **Verify run directory**
   ```bash
   # After starting a run, check it was created
   ls current_run/runs/
   ```

3. **View run status**
   ```bash
   cat current_run/runs/*/run.yaml | grep status
   ```

## Migration from Old Structure

If you have existing artifacts/ and results/ directories:

1. The old directories have been removed
2. Future runs will use the new structure automatically
3. Old checkpoints can still be loaded from their original locations if needed

## FAQ

**Q: Can I use the old checkpoint paths?**  
A: Yes, old checkpoint paths still work. The new structure only affects new runs.

**Q: What if I don't use wandb?**  
A: The system works with or without wandb. Without wandb, runs get `uuid_<id>` names.

**Q: Can I run experiments in the main `SocialTransformer/` worktree?**  
A: Yes! That's the development workflow. Use worktrees for production runs.

**Q: Do I need to manually create run directories?**  
A: No. Python scripts automatically create all run directories and metadata.

**Q: What's the difference between dev and production?**  
A: Dev is for iteration (main worktree), production is for reproducible runs (worktree snapshots).

**Q: Can I have multiple dev runs?**  
A: Yes, each run gets its own `uuid_<id>` or `wandb_<id>` directory.

**Q: How do I clean up old experiments?**  
A: Just delete the directories in `experiments/dev/` or `experiments/by_commit/`. Git worktrees are separate.

## Support

For issues or questions, see the main repository README or open an issue.
