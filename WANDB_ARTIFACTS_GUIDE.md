# Weights & Biases Artifacts Guide

This guide explains how to use wandb artifacts for saving and loading model checkpoints in the SocialBanditGuider project.

## Overview

The training script (`src/train.py`) now automatically saves model checkpoints as wandb artifacts, providing:
- Version control for your models
- Easy model sharing and collaboration
- Automatic tracking of model metadata (epoch, losses, hyperparameters)
- Best model selection based on validation loss

## Training with Artifacts

### Basic Usage

Run training with wandb artifacts enabled:

```bash
bash scripts/run_training.sh
```

This will save checkpoints to the `checkpoints/` directory and log them as wandb artifacts.

### Command-Line Options

When running `src/train.py` directly, you can customize checkpoint behavior:

```bash
python src/train.py \
    --use_wandb \
    --wandb_project "social-belief-modeling" \
    --wandb_entity "your-username" \
    --checkpoint_dir "checkpoints" \
    --save_every 5 \
    --artifact_name "sbi-transformer-checkpoint"
```

**New Arguments:**
- `--checkpoint_dir`: Directory to save checkpoints (default: `checkpoints`)
- `--save_every`: Save checkpoint every N epochs. Set to 0 to only save best and final models (default: 0)
- `--artifact_name`: Name for the wandb artifact (default: `sbi-transformer-checkpoint`)

## Checkpoint Types

The training script saves three types of checkpoints:

1. **Best Model** (`best_model.pt`): Saved whenever validation loss improves
   - Tagged with alias: `best`
   - Automatically tracked throughout training

2. **Final Model** (`final_model.pt`): Saved at the end of training
   - Tagged with alias: `latest`
   - Contains the model state from the last epoch

3. **Periodic Checkpoints** (`checkpoint_epoch_N.pt`): Saved every N epochs if `--save_every > 0`
   - Tagged with alias: `epoch_N`
   - Useful for analyzing training progression

## Checkpoint Contents

Each checkpoint file contains:
```python
{
    'epoch': int,                    # Training epoch number
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

## Loading Checkpoints

### From Local File

```python
import torch
from models.BetaModel import SBIPosteriorTransformer

# Create model
model = SBIPosteriorTransformer(
    num_arms=12,
    num_agents=4,
    max_bandits=3,
    d_model=256, nhead=8, num_layers=4, 
    dim_feedforward=768, dropout=0.1
)

# Load checkpoint
checkpoint = torch.load('checkpoints/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])

print(f"Loaded model from epoch {checkpoint['epoch']}")
print(f"Validation loss: {checkpoint['val_stats']['loss']:.4f}")
```

### From wandb Artifacts (Command Line)

Use the provided utility script:

```bash
# Load best model
python src/load_checkpoint.py \
    --source wandb \
    --wandb_entity "your-username" \
    --wandb_project "social-belief-modeling" \
    --artifact_name "sbi-transformer-checkpoint" \
    --alias "best"

# Load model from specific epoch
python src/load_checkpoint.py \
    --source wandb \
    --wandb_entity "your-username" \
    --wandb_project "social-belief-modeling" \
    --alias "epoch_5"

# Load from local file
python src/load_checkpoint.py \
    --source file \
    --checkpoint_path "checkpoints/best_model.pt"
```

### From wandb Artifacts (Python API)

```python
import wandb
import torch
from models.BetaModel import SBIPosteriorTransformer

# Initialize wandb API
api = wandb.Api()

# Download artifact
artifact_path = "your-username/social-belief-modeling/sbi-transformer-checkpoint:best"
artifact = api.artifact(artifact_path, type='model')
artifact_dir = artifact.download()

# Load checkpoint
import os
checkpoint_file = [f for f in os.listdir(artifact_dir) if f.endswith('.pt')][0]
checkpoint_path = os.path.join(artifact_dir, checkpoint_file)
checkpoint = torch.load(checkpoint_path)

# Create and load model
model = SBIPosteriorTransformer(
    num_arms=12, num_agents=4, max_bandits=3,
    d_model=256, nhead=8, num_layers=4, 
    dim_feedforward=768, dropout=0.1
)
model.load_state_dict(checkpoint['model_state_dict'])
```

## Viewing Artifacts in wandb UI

1. Navigate to your project: `https://wandb.ai/your-username/social-belief-modeling`
2. Click on "Artifacts" in the left sidebar
3. Select your artifact (e.g., `sbi-transformer-checkpoint`)
4. View all versions with their metadata
5. Use aliases to quickly find:
   - `best`: Best model by validation loss
   - `latest`: Most recent model
   - `epoch_N`: Model from epoch N

## Artifact Aliases

Each checkpoint is tagged with aliases for easy retrieval:

| Alias | Description |
|-------|-------------|
| `best` | Best model based on validation loss |
| `latest` | Most recently saved checkpoint |
| `epoch_N` | Checkpoint from epoch N |

## Best Practices

1. **Always enable wandb** when training important models:
   ```bash
   python src/train.py --use_wandb ...
   ```

2. **Use meaningful artifact names** for different experiments:
   ```bash
   --artifact_name "sbi-transformer-large-lr1e3"
   ```

3. **Save periodic checkpoints** for long training runs:
   ```bash
   --save_every 10  # Save every 10 epochs
   ```

4. **Load the best model** for inference and evaluation:
   ```python
   artifact = api.artifact('entity/project/artifact-name:best')
   ```

5. **Check checkpoint metadata** before loading:
   ```python
   print(f"Epoch: {checkpoint['epoch']}")
   print(f"Val loss: {checkpoint['val_stats']['loss']:.4f}")
   ```

## Troubleshooting

### Issue: Checkpoints not appearing in wandb

**Solution:** Make sure `--use_wandb` flag is set and you're logged in to wandb:
```bash
wandb login
```

### Issue: Out of disk space

**Solution:** Reduce checkpoint frequency with `--save_every` or manually delete old checkpoints:
```bash
rm checkpoints/checkpoint_epoch_*.pt
```

### Issue: Can't download artifact

**Solution:** Verify artifact name and alias:
```python
api = wandb.Api()
runs = api.runs("your-username/social-belief-modeling")
for run in runs:
    for artifact in run.logged_artifacts():
        print(artifact.name, artifact.aliases)
```

## Example Workflow

### Training
```bash
# Train model with artifact logging
bash scripts/run_training.sh
```

### Loading for Inference
```python
# Load best model from wandb
from src.load_checkpoint import load_checkpoint_from_wandb
from models.BetaModel import SBIPosteriorTransformer

model = SBIPosteriorTransformer(...)
checkpoint_path, checkpoint = load_checkpoint_from_wandb(
    entity="your-username",
    project="social-belief-modeling",
    artifact_name="sbi-transformer-checkpoint",
    alias="best",
    model=model
)

# Use model for inference
model.eval()
# ... your inference code ...
```

### Fine-tuning
```python
# Load checkpoint with optimizer state
import torch.optim as optim

model = SBIPosteriorTransformer(...)
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

checkpoint = torch.load('checkpoints/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# Continue training
# ... your training code ...
```

## Additional Resources

- [wandb Artifacts Documentation](https://docs.wandb.ai/guides/artifacts)
- [wandb Python API](https://docs.wandb.ai/ref/python)
- [Model Versioning Best Practices](https://docs.wandb.ai/guides/models)

