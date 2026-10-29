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

#### How each token type fills the five channels

Real events and the two structural tokens (BANDIT_SEP, QUERY) share the same five channels. The structural tokens fill the non-applicable channels with reserved **PAD** indices so they never collide with a real value (`agent_id = num_agents`, `action_id = K`, `reward_id = 2`):

| token       | bandit_ids | agent_ids                          | action_ids   | reward_ids | special_ids |
|-------------|------------|------------------------------------|--------------|------------|-------------|
| real event  | `b`        | `0`=learner / `1,2,…`=demonstrators | `arm` (0…K-1) | `0` or `1` | `0` (NONE)  |
| BANDIT_SEP  | `b`        | `num_agents` (PAD)                 | `K` (PAD)    | `2` (PAD)  | `2`         |
| QUERY       | `b`        | `num_agents` (PAD)                 | `K` (PAD)    | `2` (PAD)  | `1`         |

BANDIT_SEP and QUERY differ **only** in `special_ids` (2 vs. 1); the PAD values in the other channels are identical for both.

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
wandb sweep src/Sweeps/Sweep_Version_C.yaml

# Start agent(s) - returns sweep ID like: entity/project/abc123
wandb agent entity/project/abc123

# Run multiple agents in parallel for faster sweeps (in separate terminals)
wandb agent entity/project/abc123
```

**Swept hyperparameters:**
- Learning rate, optimizer, batch size, weight decay
- Architecture: d_model, nhead, num_layers, dim_feedforward, dropout
- Loss weighting: demo_loss_weight [0.5, 1.0]

### Continue Training from Checkpoint

Resume training from any checkpoint (local or wandb artifact) with optional hyperparameter overrides. Useful for:
- Extending training beyond original epoch count
- Fine-tuning with different learning rates or schedulers
- Recovering from interrupted training runs

**Resume from wandb artifact with more epochs:**

```bash
./scripts/continue_training.sh \
    --checkpoint <run_id> \
    --source wandb \
    --epochs 150
```

**Resume same wandb run (continuous metrics):**

```bash
./scripts/continue_training.sh \
    --checkpoint <run_id> \
    --source wandb \
    --resume_run \
    --epochs 150 \
    --lr 0.00001
```

**Resume from local checkpoint:**

```bash
./scripts/continue_training.sh \
    --checkpoint checkpoints/<run_id>/epoch_050_best.pt \
    --source local \
    --epochs 100
```

**Optional overrides:**
- `--epochs` - Total epochs to train to
- `--lr` - Learning rate
- `--scheduler` - LR scheduler type (plateau, cosine)
- `--optimizer` - Optimizer (adam, adamw, sgd)
- `--weight_decay` - Weight decay for regularization
- `--batch_size` - Batch size

**Behavior:**
- `--resume_run` flag: Continues the same wandb run (metrics append to existing run)
- Without `--resume_run`: Creates new wandb run (clean separation)
- Automatically detects if training is already complete
- Loads model, optimizer, and scheduler state from checkpoint

---

## 4. Complete Sweep Recovery

Resume incomplete runs from a wandb sweep after preemption or crashes.

### Overview

The sweep recovery tool identifies runs that didn't finish successfully (crashed, failed, or killed) and allows you to interactively resume them with their original configurations. This is useful for:
- Recovering from spot instance preemptions
- Completing runs interrupted by cluster maintenance
- Finishing runs after out-of-memory errors

**Important:** This only resumes existing incomplete runs - it does NOT start new hyperparameter configurations. To continue a sweep with new configs, use `wandb agent <sweep-id>` instead.

### Usage

```bash
./scripts/complete_sweep.sh <sweep_id>
```

The script will:
1. Query the sweep for crashed/failed/killed runs
2. Display them in an interactive list with status and progress
3. Let you select which runs to resume (e.g., "1,3,5" or "all")
4. Resume each selected run with its original configuration

### Example Session

```bash
$ ./scripts/complete_sweep.sh abc123xyz

==================================
Sweep Recovery Tool
==================================
Querying sweep: myuser/myproject/abc123xyz
==================================

Found 3 incomplete runs:

[1] vital-sweep-42 (ID: run_abc123)
    State: crashed
    Epoch: 45/100
    Val loss: 0.2341

[2] dashing-firebolt-89 (ID: run_def456)
    State: killed
    Epoch: 78/100
    Val loss: 0.1987

[3] cosmic-dragon-12 (ID: run_ghi789)
    State: failed
    Epoch: 12/100
    Val loss: N/A

Select runs to resume (e.g., '1,3,5' or 'all' or 'q' to quit): 1,2

Proceed with resuming selected runs? (y/n): y

[1/2] Resuming run run_abc123...
✓ Successfully completed run run_abc123

[2/2] Resuming run run_def456...
✓ Successfully completed run run_def456
```

### Run States

The tool filters by wandb run state:
- **crashed** - Terminated unexpectedly (no cleanup)
- **failed** - Explicitly marked as failed
- **killed** - Manually stopped by user
- **finished** - Completed successfully (excluded, even if early stopped)
- **running** - Currently active (excluded)

**Note:** The tool trusts wandb's state tracking. Runs that finished via early stopping are marked as "finished" and won't be resumed.

---

## 5. Attention Masking During Training

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
  ├── complete_sweep.py          # Resume incomplete sweep runs
  ├── models/                    # SocialTransformer + ProfileSocialTransformer + factory
  ├── inference/                 # Thompson Sampling and other inference utilities
  ├── run_inference.py           # Thompson Sampling inference entry point
  └── Sweeps/Sweep_Version_*.yaml # Hyperparameter sweep configs

scripts/
  ├── generate_simulation.sh     # Data generation script
  ├── run_training.sh            # Training script
  ├── continue_training.sh       # Continue training from checkpoint
  └── complete_sweep.sh          # Resume incomplete sweep runs

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
wandb sweep src/Sweeps/Sweep_Version_C.yaml
wandb agent <sweep-id>
```

---

## Model Architecture

**EventEmbedder** → **CausalTransformer** → **Prediction Heads**

- **Input**: Five-channel tokenized events

The embedder is an **unconditional sum** of all five channel embeddings plus a positional embedding — every channel is looked up at every position (there is no "skip"), which is why structural tokens need reserved PAD indices for their non-applicable channels:

```python
x = (
    self.bandit_emb(bandit_ids)
    + self.agent_emb(agent_ids)
    + self.action_emb(action_ids)
    + self.reward_emb(reward_ids)
    + self.special_emb(special_ids)
    + self.pos_emb(pos)
)
```

- **Transformer**: Causal attention (can't see future)
- **Outputs**: 
  - Beta(α_theta, β_theta) for bandit payoffs [B, K]
  - Beta(α_demo, β_demo) per demonstrator [B, num_demos, K]

---


## 6. Analysis on Sherlock

The analysis pipeline (`src/analysis`, driver `scripts/run_analysis.sh`) runs at full size on Sherlock as
SLURM job arrays. One-time setup from a login node inside the clone:

```bash
bash scripts/setup_sherlock.sh     # miniforge + env under $GROUP_HOME/$USER, data roots under $SCRATCH, writes .env
```

Transfer one version's inputs (only the anonymised preprocessed data ever leaves the lab machine; the raw
files and the ID map stay there):

```bash
rsync -av $EXPERIMENT_DATA_DIR/pilot_E/ sherlock:$SCRATCH/SocialTransformer/experiment_data/pilot_E/
rsync -av $OUTPUT_DATA_DIR/eval/pilot_E/ sherlock:$SCRATCH/SocialTransformer/outputs/eval/pilot_E/
rsync -av $OUTPUT_DATA_DIR/training_runs/<run>/checkpoints/<ckpt>.pt sherlock:$SCRATCH/SocialTransformer/outputs/training_runs/<run>/checkpoints/
```

Then submit: `bash scripts/slurm/submit.sh --version pilot_E` lists the cached units (`python src/run_analysis.py
--list-units`), submits one array per level (table blocks and assembly on CPU, fits, simulated datasets and PPC
draws on `owners` GPUs with `--requeue`) and the final run of every stage on `hns`; logs and unit lists land in
`$OUTPUT_DATA_DIR/analysis/<version>/slurm/`. Preempted tasks resume from the atomic caches. `$SCRATCH` purges
files untouched for 90 days, so copy `analysis/<version>/paper/` (and the fits, if wanted) to group storage after a run.
