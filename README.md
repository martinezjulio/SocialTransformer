
# SBI Transformer for Social Bandits

## Data Generation

To Generate trainin/valid/test splits run 
```
python generate_simulations.py 
```

By default data will be saved in `data/` directory. 

### Training examples format
Training example format (from `build_training_examples_from_run`)

Example:

```
{
"context_events": [(-1,-1,-1,-1)],
"current_events": [
 (0,-1,-1,-1),            # BANDIT_SEP for bandit 0
 (0, 0, 1, 1), ... ,      # demo events (agent_id=0)
 (0,99, 1, 1), ...        # learner events (agent_id=99)
],
"labels": {
 "theta": [... K floats ...],
 "alpha_demo0": [... K floats ...],   # alpha0 + S0
 "beta_demo0":  [... K floats ...],   # beta0  + F0
 "demonstrator_id": 0,
 "K": 12,
 "N_tot": 54,
 "N_by_arm": [ ..., K ints ... ]
}
}
```

Each **example** corresponds to one bandit in a multi-episode run. It has two token streams
(context + current) and a label set the model learns to predict.

Each event is a 4-tuple: `(bandit_id, agent_id, action, reward)`

- `bandit_id` – local id of the bandit within the run (0..B-1)
- `agent_id`  – local demonstrator id (0..D-1) or learner (= `99` in the simulator; remapped in the dataloader)
- `action`    – arm index (0..K-1)
- `reward`    – 0 or 1

We insert structural tokens into the streams:
- **NOCTX**: `(-1, -1, -1, -1)` → “no prior bandits”
- **BANDIT_SEP**: `(b, -1, -1, -1)` → marks the start of bandit `b`’s block

`context_events` is a list of blocks for prior bandits (up to `max_context_bandits`), each block begins with a
`BANDIT_SEP` for that prior bandit id, followed by all its events. If there is no prior history,
this list is the single **NOCTX** token.

`current_events` is the entire event stream for the current bandit, starting with its own `BANDIT_SEP` and
followed by all events for this bandit in chronological order (e.g., demo-then-learner).

Other variables:
- `theta` (length `K`) – the true payoff of each arm for current bandit
- `alpha_demo0`, `beta_demo0` (length `K`) – the demonstrator’s latent history Beta shapes
  for this bandit; with `alpha0`, `beta0` the global priors, i.e.:
  - `S0[k] = alpha_demo0[k] - alpha0`   (pre-episode successes)
  - `F0[k] = beta_demo0[k]  - beta0`    (pre-episode failures)
- `demonstrator_id` – the local id of the demo assigned to this bandit
- `K` – number of arms
- `N_tot` – total pre-episode pulls for the demonstrator in this run (fixed per demo)
- `N_by_arm` (length `K`) – integer pre-episode/latent pulls per arm (from the TS prehistory)
  - i.e. : `sum(N_by_arm) == N_tot`

> IDs are local to the run. A `demonstrator_id` of 0 in one run is *not* the same
demonstrator as `demonstrator_id` 0 in another run. The same applies to bandits. 








## 0. What the model sees

Each token is a structured event:
```
(bandit_id, agent_id, action, reward)
```
- **bandit_id**: integers for *segmenting* sequences; the identity is not important—only that bandits are distinct. We also insert **BANDIT_SEP** tokens.
- **agent_id**: `0/1` = demonstrators (support for two), `99` = learner.
- **action**: arm index in `0..K-1`.
- **reward**: `0/1`.
- If there is **no context**, we feed a `NOCTX` special token so the model learns to fall back to the base prior.

The transformer runs **causally** over `context + current_prefix + [QUERY]`. We read out predictions from the hidden state of the final `[QUERY]` token.

---

## 1. What the model predicts

Two per‑arm heads (positive via Softplus):

1. **Theta head**: \((\alpha^\theta_k, \beta^\theta_k)\) — interpreted as a **Beta distribution over \(\theta_k\)**.  
   Training NLL: \(-\log \mathrm{BetaPDF}(\theta_k; \alpha^\theta_k, \beta^\theta_k)\) using the ground‑truth \(\theta_k\) from simulation.

2. **Demonstrator start‑of‑tail head**: \((\alpha^D_{k,0}, \beta^D_{k,0})\) — continuous **pseudo‑counts** that play the role of \(\alpha_0 + S^{D,0}_k,\ \beta_0 + F^{D,0}_k\).  
   Training loss: **log‑space MSE** against the simulator’s true \(\alpha_0+S^{D,0}_k,\ \beta_0+F^{D,0}_k\).

> Why pseudo‑counts? They are continuous, stable to train, and directly combine with observed rewards:  
> \(\alpha^\star_{k,t} = \alpha^D_{k,0} + R^{\text{all}}_{k,1:t}\),  
> \(\beta^\star_{k,t} = \beta^D_{k,0}  + U^{\text{all}}_{k,1:t}\).  
> From \((\alpha^\star,\beta^\star)\) we get the exact Beta posterior over \(\theta_k\) at time \(t\).

**(Optional, advanced)**: add a *behavioral* loss that maximizes the likelihood of the demonstrator’s *actions* under Thompson Sampling using the predicted start‑of‑tail counts (requires a small 1‑D integral or low‑variance MC per demo step).

---

## 2. Thompson Sampling at inference

Given tokens for context and a current prefix, `inference.py` shows how to:

1. Run the model ⇒ predicted \((\alpha^D_{0},\beta^D_{0})\) and \((\alpha^\theta,\beta^\theta)\).
2. Combine with **observed counts** up to now:
   \(\alpha^\star_{k,t}=\alpha^D_{k,0}+R^{\text{all}}_{k,1:t}\), \(\beta^\star_{k,t}=\beta^D_{k,0}+U^{\text{all}}_{k,1:t}\).
3. **Sample** \(\tilde\theta_{k,t}\sim\mathrm{Beta}(\alpha^\star_{k,t},\beta^\star_{k,t})\).
4. Pick \(a_{t+1}=\arg\max_k \tilde\theta_{k,t}\).

This is exactly **TS** under the model’s posterior.

---

## 3. Data simulation (for SBI training)

`generate_simulations.py` produces a list of examples. Each example contains
- **context_events**: tokens from up to two prior bandits (with **BANDIT_SEP** delimiters; **NOCTX** if empty);
- **current_events**: tokens from the current bandit (a mix of demonstrator and learner events), prefixed by **BANDIT_SEP**;
- **labels**: ground‑truth per‑arm \(\theta\) and demonstrator start‑of‑tail shapes (\(\alpha_0+S^{D,0}, \beta_0+F^{D,0}\)).

The simulator creates a clear best arm per bandit and sets a fixed **experience trait** (high vs low) for a demonstrator so that transfer is learnable across bandits.

---

## 4. Dataloader

`dataloader.py` converts the examples into padded tensors with structured embeddings:
- sums of embeddings for `bandit_id`, `agent_id`, `action`, `reward`, and a `special` channel with tokens {NONE, CTX_SEP, NOCTX, QUERY, BANDIT_SEP};
- a **causal mask** (the transformer cannot see the future).  
The last token is always `[QUERY]` for readout.

---

## 5. Training

```
python generate_simulations.py --out data/train.pt --episodes 2000 --K 4 --num_bandits 3
python generate_simulations.py --out data/val.pt   --episodes  400 --K 4 --num_bandits 3

python train.py --train_path data/train.pt --val_path data/val.pt --num_arms 4 --epochs 10
```

**Losses**
- `loss_theta`: Beta NLL of true \(\theta\) under the theta head.
- `loss_demo`: log‑MSE for start‑of‑tail shapes.
- Total: `loss_theta + 0.5 * loss_demo` (tune weights).

> **Extension (recommended):** add an *action likelihood* term that evaluates the demonstrator TS probability at each demo step (Monte‑Carlo or quadrature) using the predicted start‑of‑tail shapes. This brings training closer to “full SBI” without explicit supervision of counts.

---

## 6. Inference / acting

```
python inference.py --checkpoint sbi_transformer.pt --K 4
```
- Build tokens for your **context** (possibly empty → `NOCTX`) and **current prefix** (what you’ve observed so far).  
- The script returns sampled \(\theta\) and the implied \(\alpha^\star,\beta^\star\) you can use for **TS**.

---

## 7. Handling multiple demonstrators and many contexts

- We map `agent_id=99` → **learner** and `0/1` → **demonstrators**. Extend the embedding range if you have more.
- **Bandit boundaries** are made explicit with `BANDIT_SEP` tokens; the actual bandit id carries no semantic meaning other than separating segments.
- If you have **several prior bandits** as context, just concatenate them with `BANDIT_SEP` delimiters; the model is trained on both *transfer* (context present) and *non‑transfer* (only `NOCTX`) cases with a single set of weights.

---

## 8. Why this is “SBI”

We **train on simulations** generated from the same generative story you care about. The transformer learns an **amortized posterior** over the demonstrator’s latent start‑of‑tail pseudo‑counts and an auxiliary Beta over \(\theta\). You can add a **likelihood‑based** loss on demo actions to further align with the true policy (TS).

---

## 9. Next steps / Extensions

- **Behavioral loss**: add differentiable Thompson Sampling action likelihood (MC with common random numbers) at demo steps.
- **Hierarchical transfer**: add a context encoder for prior bandits that outputs a *hyperprior* over experience magnitude and allocation (conditioning on same demonstrator identity).
- **Non‑collapsed explicit \(\theta\)**: if you need \(\theta\) explicitly for downstream tasks, keep the theta head (already present) and optionally include a reward‑predictive loss.
- **Richer tokens**: add time‑since‑last, arm features, per‑bandit IDs embedded via hashing.

---

## 10. Caveats

- This is a **starter**: defaults and losses are simple. You should tune loss weights, model size, and simulation priors to your use case.  
- The theta head is supervised by simulated ground truth; if you want a fully **likelihood‑only** approach, drop that supervision and train only with action and reward likelihoods.  
- For exact TS action likelihood you will need a small 1‑D integral or a low‑variance Monte‑Carlo estimator.

---

## 11. Model Checkpoints with Weights & Biases Artifacts

The training script now supports saving and versioning model checkpoints using **wandb artifacts**. This provides:
- Automatic versioning and tracking of model checkpoints
- Easy model sharing and collaboration
- Metadata tracking (epoch, losses, hyperparameters)
- Best model selection based on validation loss

### Training with Artifacts

```bash
# Run with wandb artifacts (recommended)
bash scripts/run_training.sh

# Or run directly with custom options
python src/train.py \
    --use_wandb \
    --wandb_project "social-belief-modeling" \
    --wandb_entity "your-username" \
    --checkpoint_dir "checkpoints" \
    --save_every 5 \
    --artifact_name "sbi-transformer-checkpoint"
```

### Checkpoint Types

1. **Best Model** (`best_model.pt`): Saved when validation loss improves (alias: `best`)
2. **Final Model** (`final_model.pt`): Saved at end of training (alias: `latest`)
3. **Periodic** (`checkpoint_epoch_N.pt`): Saved every N epochs if `--save_every > 0`

### Loading Checkpoints

**From local file:**
```python
checkpoint = torch.load('checkpoints/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
```

**From wandb artifact:**
```bash
# Load best model
python src/load_checkpoint.py \
    --source wandb \
    --wandb_entity "your-username" \
    --wandb_project "social-belief-modeling" \
    --alias "best"
```

**In inference:**
```bash
python src/inference.py \
    --use_wandb \
    --wandb_entity "your-username" \
    --wandb_project "social-belief-modeling" \
    --artifact_alias "best" \
    --K 12
```

See `WANDB_ARTIFACTS_GUIDE.md` for detailed documentation and examples.

---

## Project files

- `generate_simulations.py` — create synthetic episodes and build training examples.
- `dataloader.py` — dataset + collate to structured token tensors with masks.
- `models/BetaModel.py` — event embedder, causal transformer, per‑arm heads, TS helper.
- `train.py` — losses, training loop, and wandb artifacts integration.
- `inference.py` — how to use a trained model to sample start‑of‑tail counts and a \(\theta\) vector for Thompson Sampling.
- `load_checkpoint.py` — utility for loading checkpoints from local files or wandb artifacts.
- `WANDB_ARTIFACTS_GUIDE.md` — comprehensive guide for using wandb artifacts with this project.

---

**Have fun extending!** The scaffolding is meant to make it easy to plug in your own loss terms (e.g., TS action likelihood), richer context encoders for transfer, and different simulation regimes.
