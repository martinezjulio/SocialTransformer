# Social Bandit Guider

A transformer-based model for learning demonstrator beliefs and bandit payoffs from sequential observations with multiple demonstrators and rounds.

---

## 1. Generating Dataset

### Overview

The data generation creates **round-based** simulations where multiple demonstrators and a learner interact with bandits across several rounds. Each round contains alternating segments of demonstrator and learner actions.

### Quick Start

Generate a dataset using the default parameters:

```bash
bash scripts/generate_simulation.sh
```

**Default configuration:**
- `num_demonstrators=2`: Two local demonstrators (Demo 0, Demo 1)
- `num_rounds=3`: Three rounds per run
- `change_bandits_with_rounds=true`: Each round uses a different bandit
- `order=2`: Generates both orders (demo-first and learner-first)
- `K=12`: 12 arms per bandit
- `T_demo=6`, `T_learner=6`: 6 timesteps per segment

### Order Patterns

**order=0 (demo-first):** Demo1 → Learner → Demo2 → Learner → ...  
**order=1 (learner-first):** Learner → Demo1 → Learner → Demo2 → ...  
**order=2 (both):** Generates equal mix of both orders

### Data Structure

Each training example contains:

```python
{
  "events": [
    (0, -1, -1, -1),     # BANDIT_SEP for Round 0
    (0, 0, 3, 1),        # Demo 0 action
    (0, 0, 5, 0),        # ...
    (0, 99, 3, 1),       # Learner action
    (0, 1, 2, 1),        # Demo 1 action
    ...
  ],
  "segment_boundaries": [0, 1, 7, 13, 19, 25, ...],  # Token position of each segment
  "round_boundaries": [0, 25, 50, 75],               # Token position of each round
  "round_bandit_map": [0, 1, 2],                     # Which bandit per round
  "labels": {
    "bandits": [
      {
        "bandit_index": 0,
        "theta": [0.54, 0.29, ..., 0.80],  # True payoffs
        "demonstrators": {
          0: {"alpha0": [...], "beta0": [...]},  # Demo 0's pre-episode beliefs
          1: {"alpha0": [...], "beta0": [...]},  # Demo 1's pre-episode beliefs
        }
      },
      ...
    ],
    "K": 12,
    "num_demonstrators": 2
  }
}
```

**Event format:** `(bandit_id, agent_id, action, reward)`
- `bandit_id`: Which bandit (0, 1, 2, ...)
- `agent_id`: 0-N for demonstrators, 99 for learner, -1 for BANDIT_SEP
- `action`: Arm index (0 to K-1)
- `reward`: 0 or 1

---

## 2. Dataloaders

### Automatic Configuration

The dataloader automatically infers all parameters from the manifest:

```python
from dataloader import ShardedExamplesDataset, BanditSequenceDataset, collate_batch
from torch.utils.data import DataLoader

# Load data - parameters auto-detected from manifest
raw_ds = ShardedExamplesDataset("data/train/manifest.json")
print(f"Auto-detected: K={raw_ds.meta['K']}, num_demonstrators={raw_ds.meta['num_demonstrators']}")

# Build dataset and dataloader
ds = BanditSequenceDataset(
    raw_ds,
    num_arms=raw_ds.meta['K'],
    num_agents=raw_ds.meta['num_demonstrators'] + 1,
    num_bandits=raw_ds.meta['num_rounds']
)
loader = DataLoader(ds, batch_size=32, shuffle=True, collate_fn=collate_batch)
```

### Tokenization

Each event is encoded into five channels:

1. **bandit_ids**: [B, T] - Which bandit (0 to num_bandits-1)
2. **agent_ids**: [B, T] - Remapped locally: 0=learner, 1,2,...=demonstrators
3. **action_ids**: [B, T] - Arm index (0 to K-1, K=PAD)
4. **reward_ids**: [B, T] - Reward value (0, 1, or 2=PAD)
5. **special_ids**: [B, T] - Special tokens (0=NONE, 1=QUERY, 2=BANDIT_SEP)

A **QUERY** token is appended at the end for prediction readout.

### Batch Structure

```python
batch = {
    "bandit_ids": [B, T],           # Token sequences
    "agent_ids": [B, T],
    "action_ids": [B, T],
    "reward_ids": [B, T],
    "special_ids": [B, T],
    "pad_mask": [B, T],             # Padding mask
    "segment_boundaries": [B, S],   # Segment positions
    "round_bandit_map": [B, R],     # Bandit per round
    "y_thetas": [B, N, K],          # Target payoffs per bandit
    "y_alpha_demo": [B, N, D, K],   # Target beliefs per bandit, per demo
    "y_beta_demo": [B, N, D, K],
}
```

Where: B=batch, T=time, S=segments, R=rounds, N=num_bandits, D=num_demonstrators, K=num_arms

---

## 3. Training

### What the Model Predicts

The model predicts **Beta distribution parameters** (not point estimates):

1. **Bandit payoffs**: Beta(α_theta, β_theta) for each arm
   - Loss: Negative log-likelihood of true θ under predicted Beta
   
2. **Demonstrator beliefs**: Per-demonstrator Beta(α_demo, β_demo) parameters
   - Loss: Log-space MSE against true pre-episode counts
   - Predicts separately for each demonstrator (no averaging)

**Combined loss:** `loss = loss_theta + demo_loss_weight * loss_demo`

### Single Training Run

All parameters are auto-detected from the manifest:

```bash
bash scripts/run_training.sh
```

Or directly with Python:

```bash
python src/train.py \
    --train_path data/demonstrations_data_01/train/manifest.json \
    --val_path data/demonstrations_data_01/val/manifest.json \
    --batch_size 32 \
    --lr 3e-4 \
    --epochs 10 \
    --device cuda \
    --demo_loss_weight 0.5 \
    --use_wandb
```

**Note:** `num_arms`, `num_bandits`, `num_agents`, and `num_demonstrators` are automatically inferred from the manifest - no manual specification needed!

### Hyperparameter Sweep

Run Bayesian optimization over hyperparameters:

```bash
# Initialize sweep
wandb sweep src/Sweeps/BetaModelSweep.yaml

# Start agent(s) - returns sweep ID like: entity/project/abc123
wandb agent entity/project/abc123

# Run multiple agents in parallel for faster sweeps (in separate terminals)
wandb agent entity/project/abc123
```

**Swept hyperparameters:**
- Learning rate, optimizer, batch size
- Architecture: d_model, nhead, num_layers, dim_feedforward, dropout
- Loss weighting: demo_loss_weight [0.5, 1.0]

---

## 4. Attention Masking During Training

### Segment-Level Masking

During training, the model uses **segment-level attention masking** to learn from variable amounts of context. This enables the model to predict with different amounts of evidence.

### How It Works

1. **Random segment selection**: Each training example randomly selects 1 to N segments to be visible
2. **Attention masking**: Future segments are masked from attention
3. **Target selection**: The target bandit corresponds to the round of the last visible segment
4. **Prediction**: Model predicts from the QUERY token at the end

### Example Scenarios

For a run with 3 rounds (order=0, 2 demonstrators):

```
Segments:               Visible Context:           Target:
----------------------------------------------------------------------
1  (BANDIT_SEP R0)      Just the separator         Bandit 0
2  (Demo1, R0)          Demo1's actions            Bandit 0
3  (Learner, R0)        + Learner's actions        Bandit 0
4  (Demo2, R0)          + Demo2's actions          Bandit 0
5  (Learner, R0)        Round 0 complete           Bandit 0
6  (BANDIT_SEP R1)      + Start of Round 1         Bandit 1
7  (Demo1, R1)          + Demo1 in Round 1         Bandit 1
...
```

### Why Segment Masking?

- **Variable evidence**: Model learns to predict with minimal vs. maximal context
- **Curriculum learning**: Naturally harder predictions with less context
- **Generalization**: Better inference on partial observations
- **Data efficiency**: One example provides multiple training signals

### Implementation

The masking is implemented in `train.py`:

```python
# Randomly select visible segments
visible_segs = random.randint(1, num_segments)

# Determine target bandit from last visible segment
seg_info = segment_info[visible_segs - 1]
target_round = seg_info["round"]
target_bandit = round_bandit_map[target_round]

# Create attention mask and compute loss
attn_mask = create_segment_mask(segment_boundaries, visible_segs, seq_len)
predictions = model(batch, custom_attn_mask=attn_mask)
loss = compute_loss(predictions, targets[target_bandit])
```

---

## Project Structure

```
src/
  ├── generate_simulations.py   # Generate round-based training data
  ├── dataloader.py              # Dataset and batching with auto-config
  ├── train.py                   # Training loop with segment masking
  ├── models/BetaModel.py        # Transformer model
  ├── inference.py               # Thompson Sampling inference
  └── Sweeps/BetaModelSweep.yaml # Hyperparameter sweep config

scripts/
  ├── generate_simulation.sh     # Data generation script
  └── run_training.sh            # Training script

data/
  └── demonstrations_data_*/     # Generated datasets
      ├── train/manifest.json
      ├── val/manifest.json
      └── test/manifest.json
```

---

## Quick Start

```bash
# 1. Generate dataset
bash scripts/generate_simulation.sh

# 2. Train model
bash scripts/run_training.sh

# 3. Or run hyperparameter sweep
wandb sweep src/Sweeps/BetaModelSweep.yaml
wandb agent <sweep-id>
```

---

## Model Architecture

**EventEmbedder** → **CausalTransformer** → **Prediction Heads**

- **Input**: Five-channel tokenized events
- **Transformer**: Causal attention (can't see future)
- **Outputs**: 
  - Beta(α_theta, β_theta) for bandit payoffs [B, K]
  - Beta(α_demo, β_demo) per demonstrator [B, num_demos, K]

---

