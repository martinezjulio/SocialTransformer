# Weights & Biases Artifacts Integration - Summary of Changes

## Overview
This document summarizes the changes made to integrate wandb artifacts for model checkpoint management in the SocialBanditGuider project.

## Files Modified

### 1. `src/train.py`
**Changes:**
- Added imports for `os` and `Path` for file management
- Created new `save_checkpoint()` function that:
  - Saves complete checkpoints (model + optimizer + stats)
  - Logs checkpoints as wandb artifacts with versioning
  - Supports artifact aliases (best, latest, epoch_N)
- Added new command-line arguments:
  - `--checkpoint_dir`: Directory for saving checkpoints (default: "checkpoints")
  - `--save_every`: Frequency for periodic checkpoints (default: 0)
  - `--artifact_name`: Name for wandb artifacts (default: "sbi-transformer-checkpoint")
- Modified training loop to:
  - Track best validation loss across epochs
  - Save best model checkpoint when validation improves
  - Save periodic checkpoints (if `--save_every > 0`)
  - Save final model checkpoint at end of training
  - Log all checkpoints as wandb artifacts

**Backward Compatibility:**
- Still saves `sbi_transformer.pt` state dict for compatibility
- Works with and without wandb (checkpoints save locally even if wandb is disabled)

### 2. `scripts/run_training.sh`
**Changes:**
- Added checkpoint configuration variables:
  - `CHECKPOINT_DIR="checkpoints"`
  - `SAVE_EVERY=0`
  - `ARTIFACT_NAME="sbi-transformer-checkpoint"`
- Updated training command to include new checkpoint arguments
- Enhanced success message to show checkpoint locations and wandb artifact URL
- Added checkpoint info to configuration printout

### 3. `src/inference.py`
**Changes:**
- Added support for loading checkpoints from wandb artifacts
- New command-line arguments:
  - `--use_wandb`: Load from wandb artifact instead of local file
  - `--wandb_entity`: wandb entity name
  - `--wandb_project`: wandb project name
  - `--artifact_name`: Artifact name (default: "sbi-transformer-checkpoint")
  - `--artifact_alias`: Artifact alias (default: "best")
- Handles both full checkpoint dicts and plain state dicts
- Displays checkpoint metadata (epoch, validation loss) when loading

## New Files Created

### 1. `src/load_checkpoint.py`
**Purpose:** Standalone utility for loading model checkpoints

**Features:**
- Load checkpoints from local files or wandb artifacts
- Command-line interface for easy checkpoint retrieval
- Support for all artifact aliases (best, latest, epoch_N)
- Displays checkpoint metadata

**Usage:**
```bash
# Load from wandb
python src/load_checkpoint.py --source wandb \
    --wandb_entity "username" \
    --wandb_project "project" \
    --alias "best"

# Load from file
python src/load_checkpoint.py --source file \
    --checkpoint_path "checkpoints/best_model.pt"
```

### 2. `WANDB_ARTIFACTS_GUIDE.md`
**Purpose:** Comprehensive documentation for wandb artifacts feature

**Contents:**
- Overview of artifacts functionality
- Training with artifacts (basic and advanced usage)
- Checkpoint types (best, final, periodic)
- Checkpoint contents and structure
- Loading checkpoints (multiple methods)
- Viewing artifacts in wandb UI
- Artifact aliases explanation
- Best practices
- Troubleshooting guide
- Example workflows

### 3. `CHANGES_SUMMARY.md` (this file)
**Purpose:** Summary of all changes made in this integration

## Updated Files

### 1. `README.md`
**Changes:**
- Added new section "11. Model Checkpoints with Weights & Biases Artifacts"
- Includes quick start examples for training and loading
- Lists checkpoint types and their aliases
- References detailed guide (`WANDB_ARTIFACTS_GUIDE.md`)
- Updated "Project files" section with new files

## Checkpoint Structure

Each checkpoint contains:
```python
{
    'epoch': int,                    # Training epoch
    'model_state_dict': dict,        # Model parameters
    'optimizer_state_dict': dict,    # Optimizer state
    'train_stats': {                 # Training metrics
        'loss': float,
        'loss_theta': float,
        'loss_demo': float
    },
    'val_stats': {                   # Validation metrics
        'loss': float,
        'loss_theta': float,
        'loss_demo': float
    }
}
```

## Artifact Aliases

| Alias | Description | When Created |
|-------|-------------|--------------|
| `best` | Best model by validation loss | When val loss improves |
| `latest` | Most recent checkpoint | Every checkpoint save |
| `epoch_N` | Checkpoint from epoch N | Specific epoch number |

## Usage Examples

### Training with Artifacts
```bash
# Using the script (easiest)
bash scripts/run_training.sh

# Direct command with options
python src/train.py \
    --use_wandb \
    --wandb_project "social-belief-modeling" \
    --wandb_entity "your-username" \
    --checkpoint_dir "checkpoints" \
    --save_every 5 \
    --epochs 20
```

### Loading Checkpoints

**Method 1: Load utility**
```bash
python src/load_checkpoint.py \
    --source wandb \
    --wandb_entity "username" \
    --wandb_project "project" \
    --alias "best"
```

**Method 2: Inference script**
```bash
python src/inference.py \
    --use_wandb \
    --wandb_entity "username" \
    --wandb_project "project" \
    --artifact_alias "best" \
    --K 12
```

**Method 3: Python API**
```python
import torch
import wandb

# Download artifact
api = wandb.Api()
artifact = api.artifact('entity/project/artifact-name:best')
artifact_dir = artifact.download()

# Load checkpoint
checkpoint = torch.load(f'{artifact_dir}/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
```

## Key Features

1. **Automatic Best Model Tracking**: Saves best model based on validation loss
2. **Version Control**: All checkpoints versioned through wandb
3. **Metadata Tracking**: Epoch, losses, and hyperparameters stored with each checkpoint
4. **Flexible Loading**: Load from local files or wandb artifacts
5. **Backward Compatible**: Still saves old `sbi_transformer.pt` format
6. **Periodic Checkpoints**: Optional periodic saving for long runs
7. **Artifact Aliases**: Easy retrieval with meaningful names (best, latest, epoch_N)

## Benefits

- **Reproducibility**: Track exact model versions used for experiments
- **Collaboration**: Easy model sharing through wandb
- **Experimentation**: Compare different checkpoints easily
- **Safety**: Automatic backup of best models
- **Flexibility**: Works with or without wandb
- **Metadata**: Rich metadata for each checkpoint

## Migration Notes

### For Existing Code
If you have existing code that loads models:

**Old way (still works):**
```python
model.load_state_dict(torch.load('sbi_transformer.pt'))
```

**New way (recommended):**
```python
checkpoint = torch.load('checkpoints/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
```

Both methods are supported. The new checkpoints contain additional information (optimizer state, metrics) that can be useful for resuming training or analysis.

## Testing

To test the integration:

1. **Train a model:**
   ```bash
   bash scripts/run_training.sh
   ```

2. **Check local checkpoints:**
   ```bash
   ls checkpoints/
   # Should show: best_model.pt, final_model.pt
   ```

3. **View in wandb:**
   - Navigate to your project on wandb.ai
   - Click "Artifacts" in sidebar
   - Find your artifact (e.g., "sbi-transformer-checkpoint")
   - View versions and metadata

4. **Load a checkpoint:**
   ```bash
   python src/load_checkpoint.py --source file --checkpoint_path checkpoints/best_model.pt
   ```

5. **Run inference with artifact:**
   ```bash
   python src/inference.py --checkpoint checkpoints/best_model.pt --K 12
   ```

## Next Steps

1. Run a training session to generate checkpoints
2. Explore checkpoint metadata in wandb UI
3. Try loading checkpoints using different methods
4. Set `--save_every` for periodic checkpoints on longer runs
5. Use artifact aliases for easy best model retrieval

## Support

For detailed documentation, see:
- `WANDB_ARTIFACTS_GUIDE.md` - Comprehensive usage guide
- `README.md` - Updated project README with quick start
- [wandb Artifacts Documentation](https://docs.wandb.ai/guides/artifacts)

## Notes

- Checkpoints directory is already in `.gitignore`
- wandb artifacts don't count against your git repository size
- Old `sbi_transformer.pt` files are still created for backward compatibility
- All checkpoint features work locally even without wandb enabled

