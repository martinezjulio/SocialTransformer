# SocialTransformer internals

Companion to [`README.md`](README.md), which covers setup and how to run each
part of the pipeline. This document is the in-the-weeds reference: data format,
model architecture, losses, masking, analysis-stage internals and cluster
operations.

Where this document describes a mechanism, the code is the source of truth and
is cited by path. If you change one of those mechanisms, update the section here
in the same commit.

Contents:

1. [Data format](#1-data-format)
2. [The generative model](#2-the-generative-model)
3. [Tokenization and batching](#3-tokenization-and-batching)
4. [Model](#4-model)
5. [Losses and segment masking](#5-losses-and-segment-masking)
6. [Analysis pipeline internals](#6-analysis-pipeline-internals)
7. [Cluster operations](#7-cluster-operations)

---

## 1. Data format

### On disk

A dataset is three HuggingFace Arrow directories plus one root `metadata.json`:

```
$TRAINING_DATA_ROOT_DIR/demonstrations_data_NN/
├── metadata.json      # dataset-level config, read by train.py and eval.py
├── train/             # Arrow dataset (mmap-backed)
├── val/
└── test/
```

There are **no manifest files and no shards**. Earlier versions wrote
`manifest.json` per split alongside `.pt` shards; both are gone
(`src/generate_simulations.py:700`, `src/train.py:651`). Anything referring to
`manifest.json` is pre-Arrow.

`metadata.json` is written by `write_root_metadata` and carries `K`,
`num_demonstrators`, `num_rounds`, `learner_policy` / `learner_policy_set`,
`T_demo`, `T_learner`, `alpha0`, `beta0`, `demo_strategy`, `expert_N_min` /
`expert_N_max`, `bandit_strategy`, `bandit_payoffs`, the per-split run counts,
`seed` and `segment_type_to_id`. `train.py` and `eval.py` read `K`,
`num_demonstrators`, `expert_N_max`, `alpha0` and `beta0` from it, which is why
no CLI flag specifies dataset shape.

### Raw events

Before encoding, a run is a list of `(bandit_id, agent_id, action, reward)`
tuples (`src/generate_simulations.py`):

| field | meaning |
|---|---|
| `bandit_id` | which bandit (= which round, when `change_bandits_with_rounds`) |
| `agent_id` | `0` = learner (`LEARNER_ID`), `1..D` = demonstrators, `-1` = structural token |
| `action` | arm index `0..K-1`; on a structural token, `-1` = BANDIT_SEP and `-2` = QUERY |
| `reward` | `0` or `1`; `-1` on a structural token |

So a QUERY is emitted as `(bandit_idx, -1, -2, -1)` and a BANDIT_SEP as
`(bandit_idx, -1, -1, -1)`: same `agent_id`, distinguished by the action field.

> The learner is agent **0**, not `99`. The old README's `99` sentinel never
> existed in this generator.

### Arrow row schema

`build_features(K, num_demonstrators, num_bandits)` in
[`src/data/encoding.py`](src/data/encoding.py):

- **Token channels**, variable length `Sequence(int64)`: `bandit_ids`,
  `agent_ids`, `action_ids`, `reward_ids`, `special_ids`.
- **Structural index columns**, variable length: `round_boundaries`,
  `round_bandit_map`, `segment_boundaries`.
- **Segment info, flattened into three parallel list columns**:
  `segment_round`, `segment_type_id`, `segment_agent`. This replaced an older
  list-of-dicts `segment_info`. `SEGMENT_TYPE_TO_ID = {bandit_sep: 0, demo: 1,
  learner: 2, query: 3}`. `train.py` reads only `segment_round`; `eval.py` also
  reads `segment_agent` to recover expert-vs-novice ordering.
- **Labels, fixed shape** so they materialize as contiguous tensors under
  `set_format("torch")` without per-row nested-list conversion:
  `y_thetas` `Array2D(num_bandits, K)`, `y_alpha_demo` and `y_beta_demo`
  `Array3D(num_bandits, num_demonstrators, K)`, all float32.
- **Per-row provenance the model never reads**: `learner_policy` (string) and
  `expert_id` (int), used by eval to stratify accuracy by learner style.

`segment_boundaries[i]` is the **token position where segment `i` starts**, with
`num_segments + 1` entries (the last being `T`). This indexing is what makes the
cutoff mask in §5 cheap.

---

## 2. The generative model

`src/generate_simulations.py`, driven with the production settings by
`scripts/generate_simulation.sh`. Pipeline: `simulate_rounds_run()` →
`build_training_example()` → `build_arrow_row()` →
`Dataset.from_generator(...).save_to_disk(...)`.

One run = one training example.

**1. Bandit structure.** `K = 12` Bernoulli arms, `num_rounds = 3`, and
`change_bandits_with_rounds=True`, so one fresh bandit per round and
`num_bandits = num_rounds`. Two payoff strategies:

- `sample` — rejection-sample `Beta(2,2)^K` until `best − second ≥ 0.05` and
  `best ≥ 0.70`.
- `permute` — random permutation of a fixed 12-value set
  `0.0, 0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.75, 0.8`.

The generation script uses **`permute`**, matching the human experiment.

**2. Demonstrator experience.** `num_demonstrators = 2`. `demo_strategy`:

- `uniform_thompson` — each demonstrator's `N_tot ~ U{0..expert_N_max}`.
- `expert_novice_thompson` *(script default)* — expert
  `N_tot ~ U{expert_N_min..expert_N_max}`, with the script pinning **both to
  128**, matching the human expert's 128 shots; novice `N_tot = 0`. Both play
  Thompson.
- `expert_thompson_novice_random` — the novice plays `random_exclusion`:
  uniform over the bottom `K − 6` arms by true payoff.

`expert_id ∈ {0,1}` comes from a balanced pool, not an independent draw per run.

**3. Pre-episode beliefs — these are the labels.** `allocate_prehistory_ts` runs
`N_tot` Thompson pulls on each bandit starting from `Beta(alpha0=1, beta0=1)`,
giving per-arm `N_by_arm, S0, F0`, hence `alpha0 = 1 + S0` and `beta0 = 1 + F0`.
Those Beta parameters **are** `y_alpha_demo` / `y_beta_demo`. The belief task is
therefore "infer each demonstrator's pre-episode posterior from watching them".

**4. Round structure.** Per round: a `BANDIT_SEP`, then demo-first interleaving
`[Demo0, Learner, Demo1, Learner]` with `T_demo = T_learner = 6` steps each,
then a `QUERY`. That is **6 segments per round** (sep, demo, learner, demo,
learner, query) — 18 segments and 78 tokens for the default 3-round config.
Demonstrator beliefs reset at each new bandit; the learner's belief resets at the
start of each round.

**5. Demonstrator in-episode policy.** Thompson sampling (`ts_argmax`: one
`Beta(α,β)` draw per arm, then argmax) updated on its own observations — or
`random_exclusion` for the random novice.

**6. Learner policies.** `LEARNER_POLICIES` holds 9 named policies built from
three mechanisms × per-segment roles `[role_seg0, role_seg1]`:

| mechanism | behavior | policies |
|---|---|---|
| `priors` | Thompson re-initialized from the role's **pre-episode posterior**, updated only within the segment | `expert-priors-always`, `novice-priors-always`, `expert-novice-priors`, `novice-expert-priors` |
| `naive` | resets to uniform at round start, folds in **all** in-round observations (watched demos' actions and rewards plus its own pulls), Thompson | `naive-learner` |
| `similar` | imitates the role's **realized in-round pulls**, i.i.d. from `imitation_probs(counts, K, floor)` with exploration floor `λ = 0.1` — unpulled arms share exactly `floor`, pulled arms share `1 − floor` in proportion to counts | `expert-similar-always`, `novice-similar-always`, `expert-novice-similar`, `novice-expert-similar` |

`allowed_expert_ids(policy)` locks `similar` policies to the `expert_id` that
makes their first target demonstrator play first in the round.
`build_policy_pool(num_runs, learner_policy)` builds a shuffled,
largest-remainder-balanced pool of `(expert_id, policy)` cells;
`learner_policy="mixed"` balances all 9 equally. Non-`naive` policies require
exactly 2 demonstrators and an expert/novice `demo_strategy`.

**7. Splits.** Generator defaults are `6000 / 400 / 400` train/val/test runs; the
production script uses `300000 / 10000 / 10000`. A global `--seed 42` seeds both
`random` and `torch`. `generate_split` **refuses to overwrite** an existing split
directory.

---

## 3. Tokenization and batching

### The five channels

Each event becomes one position across five parallel `[B, T]` int64 channels
(`_encode_stream` in `src/data/encoding.py`):

1. `bandit_ids` — which bandit, `0..num_bandits-1`
2. `agent_ids` — `0` = learner, `1..D` = demonstrators, `num_agents` = PAD
3. `action_ids` — arm index `0..K-1`, `K` = PAD
4. `reward_ids` — `0`, `1`, or `2` = PAD
5. `special_ids` — `SPECIAL = {NONE: 0, QUERY: 1, BANDIT_SEP: 2}`

Real events and the two structural tokens share the same five channels.
Structural tokens fill the non-applicable channels with reserved **PAD** indices
so they never collide with a real value:

| token | `bandit_ids` | `agent_ids` | `action_ids` | `reward_ids` | `special_ids` |
|---|---|---|---|---|---|
| real event | `b` | `0`=learner / `1..D`=demos | `arm` (`0..K-1`) | `0` or `1` | `0` (NONE) |
| BANDIT_SEP | `b` | `num_agents` (PAD) | `K` (PAD) | `2` (PAD) | `2` |
| QUERY | `b` | `num_agents` (PAD) | `K` (PAD) | `2` (PAD) | `1` |

BANDIT_SEP and QUERY differ **only** in `special_ids` (2 vs 1); the PAD values in
the other channels are identical.

The reason the PAD indices must exist: `EventEmbedder`
(`src/models/components.py:19`) is an **unconditional sum** — every channel is
looked up at every position, with no skip —

```python
x = (self.bandit_emb(bandit_ids)
     + self.agent_emb(agent_ids)
     + self.action_emb(action_ids)
     + self.reward_emb(reward_ids)
     + self.special_emb(special_ids)
     + self.pos_emb(pos))
```

followed by LayerNorm, dropout and `masked_fill(pad_mask, 0.0)`. Embedding
sizes: `nn.Embedding(num_agents + 1, …)`, `nn.Embedding(num_actions + 1, …)`,
`nn.Embedding(3, …)` for reward and special, and `nn.Embedding(4096, …)` for
position.

### The batch

After `collate_batch` in [`src/data/dataloader.py`](src/data/dataloader.py)
(**not** a top-level `dataloader` module — that import path is from an earlier
layout). With B = batch, T = padded tokens, S = max segments, R = max rounds,
N = num_bandits, D = demonstrators, K = arms:

```python
batch = {
    "bandit_ids":         [B, T],   # token channels; PAD: action_ids -> num_arms,
    "agent_ids":          [B, T],   #   reward_ids -> 2, others 0
    "action_ids":         [B, T],
    "reward_ids":         [B, T],
    "special_ids":        [B, T],
    "pad_mask":           [B, T],   # bool, True = padded
    "round_boundaries":   [B, R],
    "round_bandit_map":   [B, R-1], # round_boundaries carries one extra entry
    "segment_boundaries": [B, S],
    "segment_round":      [[int]],  # Python lists per example, unpadded,
    "segment_type_id":    [[int]],  #   iterated by index, never fed to the model
    "segment_agent":      [[int]],
    "y_thetas":           [B, N, K],
    "y_alpha_demo":       [B, N, D, K],
    "y_beta_demo":        [B, N, D, K],
}
```

Labels are float32 and may be scaled by the `BeliefNormalizer` (§5).
`load_arrow_split` mmaps via `load_from_disk` and applies
`set_format("torch", columns=[the three label columns], output_all_columns=True)`,
so random access under DDP sample shuffling is paid by the page cache rather
than by per-row conversion.

---

## 4. Model

### The factory boundary

`build_model(model_type, config)` in
[`src/models/__init__.py`](src/models/__init__.py) is the **only** constructor
used by train, eval and inference; checkpoint loading goes through it too, via
`BaseSocialTransformer._build_and_load`.

| `model_type` | class | file |
|---|---|---|
| `social` | `SocialTransformer` | `src/models/social_model.py` |
| `social_profile_learned` | `LearnedProfileTransformer` | `src/models/profile_social_model.py` |
| `social_profile_agent_pooling` | `ProfileSocialTransformer(model_type=…)` | `src/models/profile_social_model.py` |
| `social_profile_cross_attention` | `ProfileSocialTransformer(model_type=…)` | same |

`_LEGACY_ALIASES` maps the four pre-rename strings so old checkpoints and
leftover sweep YAMLs still load: `no_profile → social`,
`learned_profile → social_profile_learned`,
`agent_pooling → social_profile_agent_pooling`,
`cross_attention → social_profile_cross_attention`. `_build_and_load` also
accepts the legacy `model_config["profile_method"]` key, defaulting to
`"social"`.

Required config keys: `num_arms, num_agents, num_bandits, num_demonstrators,
d_model, nhead, num_layers, dim_feedforward, dropout`. Optional, with defaults:
`env_theta_loss='bce'`, `experience_head_loss_type='mse'`,
`theta_head_input='query_only'`,
`demo_beliefs_head_input='query_and_profiles'`,
`experience_head_input='profiles_only'`. The three head-input keys are silently
ignored when `model_type='social'`; extra keys are ignored; an unknown
`model_type` raises `ValueError`.

### Backbone and readout

`CausalTransformerEncoder` wraps
`nn.TransformerEncoder(nn.TransformerEncoderLayer(..., batch_first=True), num_layers)`.
The default attention mask is upper-triangular causal `[T, T]`; a
`custom_attn_mask` (the segment mask, `[B*nhead, T, T]`) overrides it.
`src_key_padding_mask=pad_mask`.

All heads read **QUERY positions** (`special_ids == 1`), one per round/bandit.
`num_queries` is the max QUERY count in the batch, and head outputs are stacked
on a `num_queries` axis.

### What distinguishes the variants

> **Only `social` is in use.** Every published result came from it, and every
> sweep config pins `model_type: "social"`. The three `social_profile_*`
> variants below are deprecated exploratory paths, documented because the code
> is still there and old checkpoints still load through it.

`SocialTransformer` (120 lines) is the plain path: embed → causal transformer →
per-QUERY heads, head inputs hardcoded to `query_only`. It has **no experience
head** (so no `N_tot_pred` key) and no demonstrator profile vectors at all.

The three profile-aware variants add a per-demonstrator profile `[B, D, d_model]`
plus a `DemoExperienceHead`. They differ only in where the profile comes from:

- **`social_profile_agent_pooling`** — `AgentPoolingExtractor`: mean-pools the
  post-backbone hidden states grouped by `agent_ids == demo_id` for
  `demo_id ∈ 1..D`, count clamped at ≥ 1. No parameters.
- **`social_profile_cross_attention`** — `CrossAttentionExtractor`: `D` learned
  query vectors (`nn.Parameter(D, d_model)`) cross-attend over the hidden states
  via `nn.MultiheadAttention(d_model, nhead, batch_first=True)` with
  `key_padding_mask=pad_mask`.
- **`social_profile_learned`** — the only **sequence-mutating** variant. `D`
  learnable profile tokens are spliced into the embedded sequence immediately
  before each QUERY, *pre*-backbone; `pad_mask` is extended in lockstep (profiles
  are never padding); QUERY positions shift to `p + q_idx*D + D`; per-QUERY
  profiles are read back out of the inserted slots `h[:, q_pos-D:q_pos, :]`.

### Two gotchas worth knowing

1. **`LearnedProfileTransformer.forward` discards `custom_attn_mask`.** When one
   is passed it rebuilds a plain causal mask at the new, longer `T`. Segment-cutoff
   masking (§5) is therefore **inert for that variant** — it trains on full
   context regardless of the sampled cutoff.
2. It also assumes QUERY positions are **uniform across the batch** (it reads
   `query_mask[0]` only), and the experience head in both profile classes uses
   only the **last** QUERY's context.

### Heads

All three are 2-layer MLPs with ReLU (`src/models/components.py`). Input size
depends on `input_mode`: `query_only → d_model`,
`profiles_only → d_model * D`, `query_and_profiles → d_model + d_model*D` (the
experience head is per-demonstrator, so it uses `d_model` and `2*d_model`
respectively).

| head | shape | output |
|---|---|---|
| `BanditPayoffsHead` (theta) | `Linear(in, d_model) → ReLU → Linear(d_model, out_dim)`, `out_dim = K` if `env_theta_loss=="bce"` else `2K` | BCE mode: raw `theta_logits [B, num_queries, K]`. Beta mode: reshape `[B, 2, K]`, `alpha_theta = softplus(·)+1e-3`, `beta_theta` likewise → `[B, num_queries, K]` each |
| `DemoBeliefsHead` | `Linear(in, d_model) → ReLU → Linear(d_model, 2*D*K)`, reshaped `[B, D, 2, K]` | `alpha_demo`, `beta_demo` = `softplus(·)+1e-3` → `[B, num_queries, D, K]` each. **Per-demonstrator, never averaged.** |
| `DemoExperienceHead` (profile models only) | `Linear(in, d_model//2) → ReLU → Linear(d_model//2, 1)`, applied per demonstrator in a loop | `N_tot_pred [B, D]`. `output_type="mse"` → `softplus(·)+1e-3` (positive continuous `N_tot`); `"bce"` → raw logits (expert vs novice) |

At inference, `readout()` in
[`src/inference/readout.py`](src/inference/readout.py) returns these with a
leading cutoff axis (`alpha_demo [C, rounds, D, K]`,
`theta_logits [C, rounds, K]`) and denormalizes `alpha_demo` / `beta_demo` when
the checkpoint was trained with normalized beliefs.

---

## 5. Losses and segment masking

### Loss registry

[`src/models/losses.py`](src/models/losses.py). Three selector functions pick a
loss from a CLI string.

**Theta (environment payoffs)** — `get_env_theta_loss(loss_type)`,
`--env_theta_loss`:

- `bce` **(default)** → `environment_BCE_loss`:
  `binary_cross_entropy_with_logits(theta_logits, theta_true)`, mean.
- `beta_nll` → `beta_nll_of_true_theta`: NLL of the true θ ∈ (0,1) under the
  predicted `Beta(α, β)`, closed form via `lgamma`, θ clamped to
  `[1e-6, 1-1e-6]`, mean.

**Demonstrator beliefs** — `get_demo_belief_loss(loss_type)`,
`--demo_belief_loss`, default `log_mse`. All take
`(pred_alpha, pred_beta, target_alpha, target_beta)` → scalar:

- `log_mse` → mean squared difference of `log α` and of `log β`, averaged. The
  only loss for which `requires_normalization(loss_type)` is True.
- `kl_divergence` → closed-form `KL(Beta(target) ‖ Beta(pred))` via `lgamma` /
  `digamma`.
- `expected_value_mse` → MSE on the means `α/(α+β)`, discarding uncertainty.
- `cross_entropy` → closed-form `−E_{θ∼Beta(target)}[log Beta(θ; pred)]`.

**Experience** — two distinct terms:

- `experience_loss_log(...)` derives `N_tot = Σ_k (α − α0) + (β − β0)` from the
  **belief head's** Beta parameters, clamps to ≥ 0, and takes MSE in log space
  (`log(N+1)`). Weight `--experience_loss_weight`, default **0.0** (off).
- `get_experience_head_loss(loss_type)` scores the **dedicated** `N_tot_pred`
  head: `mse → experience_head_mse_loss` (log-space MSE against the target
  `N_tot`) or `bce → experience_head_bce_loss` (label `N_tot > expert_threshold`,
  default 30.0, via `binary_cross_entropy_with_logits`). Weight
  `--experience_head_loss_weight`, default 1.0; contributes 0.0 when the model
  emits no `N_tot_pred`.

### The total

Assembled once in `compute_losses` (`src/train.py:371`) — deliberately a single
copy shared by train and val, after an earlier drift where val scored a
different task:

```python
loss = (loss_theta
        + demo_loss_weight * loss_demo                       # --demo_loss_weight, default 0.5
        + experience_loss_weight * loss_experience           # default 0.0
        + experience_head_loss_weight * loss_experience_head)  # default 1.0
```

**Four weighted terms, and theta defaults to BCE** — not the two-term
`loss_theta + demo_loss_weight * loss_demo` with Beta NLL that older docs show.
`loss_demo` and `loss_experience` are averaged over demonstrators.

Normalization: `BeliefNormalizer`
([`src/data/transforms.py`](src/data/transforms.py)) uses
`norm_factor = max_experience + alpha0 + beta0`, with
`max_experience = expert_N_max` from dataset metadata. It is applied to targets
in the collate and **undone on both sides** inside `compute_losses` before the
belief and experience losses.

### Choosing visible segments

`sample_segment_cutoffs(batch, rng)` (`src/train.py`) draws, per example,
`visible_segs = rng.randint(1, num_segments)` (inclusive both ends) and resolves
the **target bandit** as
`round_bandit_map[b, segment_round[b][visible_segs - 1]]` — the bandit of the
last visible segment.

`rng` is injected rather than taken from the global stream: training passes the
`random` module (seeded once from `--seed`); validation passes a generator
**reseeded at the top of every val pass**, so val draws identical cutoffs each
epoch. That keeps `val/loss` free of resampling jitter, which matters because it
drives the plateau scheduler, best-model selection and the overfitting
early-stop.

With `--use_segment_masking` off, the mask is `None` and the target is
`round_bandit_map[:, -1]` (full context, last bandit).

### The mask

`create_segment_mask(...)` in
[`src/models/masking.py`](src/models/masking.py) — kept out of `train.py` so
inference does not import wandb and the data loaders. Returns a bool
`[B, T, T]` where **True = masked**.

A cutoff of `c` means "the first `c` segments are visible":
`cutoff_pos = segment_boundaries[b, c]`, and everything from `cutoff_pos` onward
becomes unattendable. The function starts from causal `triu(diagonal=1)`
broadcast per example, then sets `batch_mask[b, :, cutoff_pos:] = True`. Two
strategies (`--masking_strategy`):

- `all_bandits` **(default)** — the cutoff truncation only.
- `current_bandit_only` — additionally masks whole segments within the visible
  prefix that belong to a bandit other than the target.

Then the critical last line:

```python
batch_mask.diagonal(dim1=-2, dim2=-1).fill_(False)
```

**No attention row may be fully masked.** A softmax over an all `-inf` row is
NaN; under `current_bandit_only` every position in an earlier round loses all of
its keys once the cutoff passes that round. The NaN is silent in `train()` mode
but poisons the whole sequence in `eval()` via the fused attention path. This is
invariant 1 of
[`tests/test_segment_masking.py`](tests/test_segment_masking.py); invariant 2 is
that the mask must stay **per-example** (`.repeat_interleave(nhead, dim=0)` to
`[B*nhead, T, T]`, never collapsed to a shared `[T, T]`, which would score each
example's target against another example's context); invariant 3 is that no NaN
reaches outputs in `eval()`.

Net effect: the QUERY token for the target round still runs (its diagonal is
forced visible) but can only attend to keys strictly before `cutoff_pos`, so one
episode yields `num_segments` distinct supervised prediction problems.
`compute_losses` indexes both targets and predictions by `target_bandits` on the
bandit/query axis (`out["alpha_demo"][batch_indices, target_bandits_tensor] →
[B, D, K]`), which assumes query `q` ↔ bandit `q` — true when
`change_bandits_with_rounds=True`, i.e. one bandit and one QUERY per round.

The same function is shared with inference: `src/inference/readout.py` expands
one tokenized session to one row per cutoff `1..num_segments` and masks per row.
`src/eval/eval.py` has its own single-example variant and sweeps
`cutoff ∈ 1..num_segments`.

---

## 6. Analysis pipeline internals

Driver: [`src/run_analysis.py`](src/run_analysis.py) holds the Python stages,
registered with an `@stage(name)` decorator into an ordered `STAGES` dict.
The real entry point is [`scripts/run_analysis.sh`](scripts/run_analysis.sh),
which interleaves the R stages. Everything is keyed by `--version`; all stages
write under `$OUTPUT_DATA_DIR/analysis/<version>/<stage>/`.

Paths travel in the frozen `Paths` dataclass
(`version, preprocessed, eval_dir, analysis_dir, checkpoint`). Stages import
their dependencies lazily *inside* the function, so the driver itself never
loads torch or JAX.

### Stages, in pipeline order

| # | Stage | Lang | Computes | Outputs |
|---|---|---|---|---|
| 1 | `games` | Py (`analysis/games.py`) | One row per **play game**: participant, session, session_order 1–3, condition (expert-first / novice-first / solo), round, slot 1–4, play (first/second), demonstrator watched, mean true payoff of chosen arms, mean reward, Shannon entropy (bits) over the 6 choices, `n_shots`, `exploratory` flag (solo slots 1 and 3). Heavily assertion-checked. | `games/play_games.csv` |
| 2 | `descriptive` | **R** (`r/descriptive.R`) | The 12-cell mixed model per outcome (`payoff`, `reward`, `entropy`): `outcome ~ condition * play * round + (1 \| participant)`, sum-to-zero coding, REML, Satterthwaite df (lmerTest/emmeans). Confirmatory contrasts H1, H2, H3, H5 with Holm within hypothesis plus verdicts; reward rows reported without a rule; registered exploratory contrasts; two sensitivity refits. | `descriptive/{cell_means,contrasts,contrast_weights,robustness}.csv`, `numbers.json` |
| 3 | `partner_choice` | **R** (`r/partner_choice.R`) | H4, the end-of-session expert-vs-novice pick: per-session table, Wilson-interval rates overall and by condition, mixed logistic model with participant intercept for `P(choose expert) > 0.5`, plus the exploratory recency term. Reads the preprocessed CSV directly — the choice game is not tokenized, so it is in no inference CSV. | `partner_choice/{choices,rates,model}.csv`, `numbers.json` |
| 4 | `inference-check` | Py (in `run_analysis.py`) | Asserts the inference CSVs came from **this** version's preprocessed file (comparing `inference_config.yaml`'s recorded `experiment_data_path`) and that their session/participant sets equal the social sessions in the preprocessed file. | — (prints only) |
| 5 | `choice` | Py (`policies`/`thompson`/`fitting`/`compare`) | The confirmatory policy-model stage: per-(source, κ, own) log-policy tables, NUTS fits of the whole registry, PSIS-LOO with a Pareto-k leave-one-session-out fallback on the S1/S2 models, the ladder vs HOS, the ladder per round, per-participant ΔELPD, four-outcome verdicts for S1a–S3b (2 SE rule), claims S1–S3 composed from their parts, S4 from δ_N's 95% ETI, and the reported posteriors. | `choice/{ladder,ladder_by_round,elpd_per_participant,verdicts}.csv`, `choice/fits/`, `choice/loso/`, `tables/`, `numbers.json` |
| 6 | `checks` | Py (`analysis/checks.py`, `predictive.py`) | Simulation-based checks, all GPU: δ_N parameter recovery (`recover_delta`, 25 datasets × 100 simulated participants, coverage/sign rule `COVER_AT_LEAST = 22`), model recovery for `COMPARISONS = {S1a: (HOS, recent), S2a: (HOS, naive-recent), S4: (Switch-by-round, Switch)}`, free-running PPC on `PPC_MODELS = (HOS, Switch-by-round)` over 5 targets (`after_novice_rise, entropy_fall, zigzag, stay_win, stay_loss`), and the exploratory split-half generative check. Loads the transformer itself (`inference.readout.load_model`), so simulated sessions re-read the network before every play game. | `checks/{recovery,model_recovery,confusion,ppc,model_recovery_verdict}.csv`, `checks/{recovery,model_recovery,ppc,split_half}/`, `numbers.json` |
| 7 | `unseen` | Py (`analysis/unseen.py`) | Exploratory unseen-expert section: `readout_table` (confidence, Spearman rank-correlation with truth, favourite arm's true payoff, Thompson entropy, by readout × expert_seen × round), `copying_table` (expert readout vs the `Beta(1+wins, 1+losses)` copying belief), `elpd_by_cell` splits reusing the choice stage's cached LOO, and `exploit_table`. | `unseen/{readout_table,copying_table,exploit,elpd_split,copying_elpd}.csv`, `numbers.json` |
| 8 | `figures` | Py (`analysis/figures/`) | Fig 1 inset, Figs 2–4 and the supplement, from the stage outputs and cached fits. Forces the `Agg` backend unless `MPLBACKEND` says otherwise. PDF + PNG per figure with fixed names (`fig2_primary`, `figS3_ppc`, …). | `figures/` |
| 9 | `report` | Py (`analysis/report.py`) | Last: merges every stage's `numbers.json` with the provenance block, derives the check-gated final verdicts, validates against the manifest, writes the paper artifacts. | `paper/{results.tex,results.md}`, `paper/tables/{hypotheses,ladder}.tex`, `paper/figures/` |
| — | `power` | **R** (`r/power_analysis.R`) | Runs **only when named** — not in the default list. Resamples the pilot at N ∈ {200…800}, refits the descriptive model, applies the registered rule, records decision rates, plus a closed-form N reference. An input to the registration, not a result. | `power/{decision_rates,closed_form_n}.csv` |

Default `--stages` in `scripts/run_analysis.sh`:
`games,descriptive,partner_choice,inference-check,choice,checks,unseen,figures,report`.
The shell script dispatches `descriptive|partner_choice|power` to
`Rscript src/r/<stage>.R` with `VERSION` exported, and everything else to
`python src/run_analysis.py --version <v> --stages <stage>`. `run_analysis.py`
always runs the requested stages in **registry order**, whatever order they were
listed in.

### The policy-model registry

[`src/analysis/policies.py`](src/analysis/policies.py). **HOS = H habit, O own
experience, S social belief.** Every model is the full HOS model with terms
switched off; a name either lists the letters that are on, or names the ablated
**belief source** instead, or adds a suffixed extension.

`ModelSpec` (frozen dataclass) fields: `name`; `source` ∈
`SOURCES = (expert, novice, recent, naive, vis_last, vis_expert, uniform)`;
`kappa` (hierarchical κ_i scaling the social evidence; False fixes κ = 1); `own`
(λ_w on this play game's own counts, λ_c on the round's earlier play game);
`reset` (λ_c ≡ 0); `habit` (win-stay/lose-shift tilt `ρ0_i + ρwin_i · prev_reward`
on the previous arm); `persev` (group-level bonus on any arm already chosen this
game); `mixture` ∈ `MIXTURES = (none, two, by_round, eight)`; `kappa_trend` (δ_κ
round slope on log κ); `lapse` (fixed 0.02, or `None` to fit ε ~ Beta(1,9));
`role`. Invariants: mixtures require `source="expert"` and `kappa=True`; `reset`
requires `own`.

The 21 registered models:

| name | role | description |
|---|---|---|
| `HOS` | full | The full model: habit + own counts + the expert readout scaled by κ_i. The reference every ΔELPD is measured against. |
| `recent` | S1a | HOS, but the belief is the **most recently watched** demonstrator's readout. |
| `novice` | S1b | HOS, but the belief is the **novice's** readout. |
| `naive` | S2c | Belief = Beta(1,1) + every pull visible this round (demos + own); own terms off, since the pooled count already holds them. |
| `naive-recent` | S2a | Belief = Beta(1,1) + the **last demonstration's** visible pulls — copying, not a readout. |
| `naive-expert` | S2b | Belief = Beta(1,1) + the **expert's** visible pulls this round. |
| `OS` | reported | Own + social, no habit. |
| `HS` | reported | Habit + social, no own counts. |
| `HO` | reported | Habit + own; uniform belief with κ fixed at 1 (no social). |
| `H` | reported | Habit only. |
| `O` | reported | Own counts only. |
| `S` | reported | Social belief only. |
| `Random` | reported | Uniform chance baseline; nothing to sample, so its log-likelihood is taken directly. |
| `Switch` | reported | HOS mixed with the novice-readout policy, weight ω_E,i / ω_N,i per participant by who was watched last. |
| `Switch-by-round` | S4 | `Switch` plus δ_E / δ_N round slopes on the mixing logit. **δ_N is the S4 estimand.** |
| `Switch-by-round+ktrend` | robustness | Adds δ_κ, a round slope on log κ. |
| `Switch-8` | exploratory | Eight group-level weights indexed by (demonstrator watched, round, condition), participants deviating with a shared τ·z_i. Drives Fig 4a. |
| `HOS+persev` | exploratory | HOS plus a perseveration bonus on any arm already chosen this play game. |
| `HOS-reset` | exploratory | HOS with λ_c ≡ 0: the learner resets between the round's two play games. |
| `HOS+eps` | robustness | HOS with the lapse ε fitted (Beta(1,9)) rather than fixed at 0.02. |
| `Switch-by-round+eps` | robustness | `Switch-by-round` with the lapse fitted. |

`belief(T, source)` returns a `(prior, social)` count pair, each `(N, K, 2)`: a
readout (`expert`/`novice`/`recent`) is its own prior with prior 0; the count
sources (`naive`, `vis_last`, `vis_expert`) are Beta(1,1) + visible pulls;
`uniform` is Beta(1,1) alone.

The model body is NumPyro: step 1 posterior
`prior + κ·social + λ_w·own_seg + λ_c·own_prev`, step 2 Thompson (**read from
the interpolated tables, not recomputed**), step 3 habit/perseveration tilt,
step 4 lapse, plus the mixture `logaddexp`. `grids(spec)` gives the
`(κ, λ_w, λ_c)` knots the spec's table needs (a single knot where a term is
off); `cubic` is 4-point Lagrange interpolation along the grid axis;
`hierarchical` is the non-centred `name_i = mu + tau*z_i`; `mixture_logit`
implements the `two` / `by_round` / `eight` weight schemes.

### Fitting

[`src/analysis/fitting.py`](src/analysis/fitting.py): NUTS, `CHAINS = 4`, and a
3-rung `LADDER` of `(num_warmup, num_samples, target_accept_prob)` =
`(1000, 1000, 0.9)` → `(2000, 2000, 0.95)` → `(4000, 4000, 0.99)`, gated by
`GATE = dict(rhat=1.01, ess=400, divergences=0)`. A fit that fails the last rung
is returned with `passed=False` and **never reported**. `run_many` fits several
models in spawned processes, one per JAX device, each memory-mapping its tables.

### Caching

Three independent layers. Every write goes through `atomic_write(path, write)`
in [`src/analysis/data.py`](src/analysis/data.py) — write to a hidden
`.stem.tmp<suffix>` beside the target, then `os.replace` — so a task killed
mid-write (SLURM preemption) never leaves a file that looks complete.

**1. Belief tables, keyed by content hash.** `table_path(...)` in
[`src/analysis/thompson.py`](src/analysis/thompson.py) names the file
`<cache_dir>/<name>_<sha256 of all inputs and grids, 16 hex>.npy`, where `name`
is `<source>_<int(kappa)>_<int(own)>`. Each table is
`(N, |κ|, |λ_w|, |λ_c|, K)` float32 floored log-probabilities on `KAPPA` (29
geomspace knots 0.125–16, ratio 2^¼, with 1.0 on the grid), `LAM_W` (16 knots
0.1–16) and `LAM_C` (0 plus 9 knots 0.25–4, where 0 = reset learner). Builds
split into `N_TABLE_BLOCKS` contiguous row blocks: `block=b` builds, caches and
returns one block; `block=None` assembles from cached blocks, building any
missing, then deletes the block files. The `choice` stage queues table builds on
a single CPU `ThreadPoolExecutor` in registry order while the GPU fits, waiting
only on the tables each model needs. Human fits use the scipy builder
(`p_win_all`, 1e-6 mass gate); simulated datasets use `builder="gpu"`
(`p_win_gpu`, 1e-4 gate) and cache under `<name>_gpu`.

**2. Fits, keyed by the table hashes.** `cache_path(name, key, cache_dir)` →
`choice/fits/<sanitised name>_<hash>[-<hash2>].pkl`, where `key` is the hash of
the tables the fit read (two for a Switch model: expert + novice). Changing the
data or the grid therefore invalidates the fit automatically. The same key names
the Pareto-k fallback refits under `choice/loso/`. `figures/fits.py::load_fit`
resolves a model's pickle by an exact name-plus-hash regex, so `HOS` never
matches `HOS+persev`.

**3. Checks, keyed by unit.** Simulated datasets, fits, PPC draws and split-half
draws are cached per dataset index / per draw index under
`checks/{recovery,model_recovery,ppc,split_half}/` with **seeded** draw lists —
so a reduced run's units are literally the first units of the full run, and are
reused by it.

### `--dev` is a guard, not a mode

The reduced sizes come from environment variables read in
`src/analysis/checks.py`: `N_DATASETS` (default 25), `N_MODEL_RECOVERY` (5 per
generating side), `N_PPC_DRAWS` (50), plus `N_SIMS` for the R power stage.
Because stages overwrite their outputs unconditionally, `scripts/run_analysis.sh`
and `scripts/slurm/submit.sh` **refuse to run** if any of those is set unless
`--dev` is passed (exit 2, "a reduced run overwrites the full outputs; pass
--dev if that is intended"). With `--dev` they print `DEV RUN: reduced sizes
allowed; outputs under analysis/<version> are overwritten`. `--dev` changes
nothing else — no stage reads it.

### Numbers: the manifest contract

[`src/analysis/numbers.py`](src/analysis/numbers.py) is what keeps any paper
number from being typed by hand.

- Each stage calls `write_numbers(stage_dir, numbers)`, which runs `check_keys`
  (every value a plain number, bool or short string; every key yielding a
  **distinct** macro name) and writes `<stage_dir>/numbers.json` in insertion
  order. The five stages that write one are
  `STAGES_WITH_NUMBERS = ("descriptive", "partner_choice", "choice", "checks", "unseen")`
  — the first two from R.
- Keys are free text (`S1a_estimate`, `elpd_HOS+eps`,
  `split_half_zigzag_inside`). `macro_name(key)` keeps letters only: words split
  at every non-letter, digits spelled out, each word capitalised, prefixed
  `res`. So `S1a_estimate` becomes `\resSOneAEstimate`.
- `KINDS` is an ordered list of (regex, quantity type) pairs, first match wins:
  `text`, `pvalue`, `two`, `three`, `integer`, `flag` (yes/no), `count`.
  `kind_of(key)` resolves the type and `fmt(value, kind)` renders it — NaN → em
  dash, `pvalue` → `< .001` below 0.001, negative zero stripped.
- `MANIFEST` maps **block → list of keys**, one block per figure or text
  section: `provenance`, `descriptive (Fig 2, H1-H5)`,
  `partner choice (Fig 2c, H4)`, `policy models (Fig 3, S1-S4)`,
  `unseen expert (Fig 4)`, `checks (Fig S, PPC and recovery)`, and
  `final verdicts (report stage, check-gated)`. It is built from named tuples
  (`OUTCOMES`, `EXPLORATORY`, `EXCLUSION_RULES`, `DEMOGRAPHICS`, `CLAIMS`,
  `POSTERIORS`, `PPC_TARGETS`) plus the **live** `REGISTRY`, so every model
  automatically gets an `elpd_<model>` key.
- `check_manifest(numbers)` raises naming **each missing key by block and each
  extra key**. The pipeline fails rather than shipping an unlisted or absent
  number.

### The report stage

[`src/analysis/report.py`](src/analysis/report.py), in order:

1. `provenance(...)` — git commit (`-dirty` when the tree has changes), the
   `sha256` of the checkpoint the inference run used, run date,
   participants/sessions/trials counts, the exclusion ledger (`exclusions.csv`)
   and the demographics summary (`demographics_summary.csv`).
2. `merge_numbers` — merges every stage's `numbers.json` plus provenance. A key
   written by two stages is an error.
3. `final_verdicts` — derives check-gated claim verdicts via
   `GATES = {S1: (model_recovery_S1a_passed,), S2: (model_recovery_S2a_passed,),
   S3: (), S4: (recovery_passed,)}`. A claim whose gating check failed or is
   missing becomes **not evaluable** regardless of the choice stage's own
   verdict, which stays beside it as the description of the human data. Then
   `check_manifest`.
4. Writes `paper/results.tex` (one `\newcommand` per number, plus a composed
   `<stem>_CI` macro — "est [lo, hi]" — after each estimate/lo/hi triple;
   `tex()` routes minus signs and `<` through `\ensuremath` and the em dash to
   `---`), `paper/results.md` (every macro/key/value grouped by manifest block,
   regenerated each run so a diff shows what changed),
   `paper/tables/hypotheses.tex` and `paper/tables/ladder.tex` (booktabs via a
   pandas Styler), and copies `figures/fig*` into `paper/figures/`.

### Inputs

**`$EXPERIMENT_DATA_DIR/<version>/preprocessed_participant_data.csv`** — written
by [`src/r/preprocess_participant_data.R`](src/r/preprocess_participant_data.R)
from the anonymized export; one row per logged event, ordered by (participant,
session start, timestamp, step). **All exclusion rules live in that script**
(`returned, colour_vision, speed_floor, span_ceiling, no_gameplay,
incomplete_rounds, short_play, reload, restarted, never_moved,
not_three_sessions`), which also writes `exclusions.csv`,
`demographics_summary.csv` and a timestamp-free `..._public.csv`.

Key columns consumed downstream: `anonymous_id`, `session_id`, `session_index`
(1–3), `game_sequence` (`expert-play-novice-play-choice` /
`novice-play-expert-play-choice` / `play-play-play-play` = solo), `game`
(`watch` | `play` | `choice`), `game_index` (densely renumbered to
`4*round_index + slot`), `round_index` (0/1), `shot` (bool; the actual shot
rows), `position` (arm chosen), `reward_generated`, `zone_bandit` (the round's
true payoff vector, a bracketed string parsed with `ast.literal_eval`),
`zone_global_index`, `demonstrator_type` (`expert`/`novice`, on watch rows
only), `choice_demonstrator_type`, `action`, `step`, `timestamp`, plus the
pilot_E-only `demonstrator_index` / `demonstrator_experience` (128 = expert,
0 = novice) / `choice_demonstrator_index` from which the `*_type` labels derive.

**Inference CSVs in `$OUTPUT_DATA_DIR/eval/<version>/`** — written by
`src/run_inference.py`. Three artifacts are consumed:

1. `data/experiment_trial_predictions.csv` — per-game / per-step rows. Loaded by
   `load_sessions` in `src/analysis/data.py`, **filtered to
   `learner_segment_input == "human"`**. Reads `session_id, anonymous_id,
   game_sequence, round_index, game_index, demonstrator_type, position,
   zone_bandit, reward_generated` (→ `reward`), plus `learner_segment_input,
   rollout_id, beta_method, own_experience, theta_betas, theta_predicted,
   expert_betas, novice_betas`.
2. `data/experiment_cutoff_human_predictions.csv` — one row per
   `(session_id, cutoff)` with `cutoff ∈ 1..12` (12 segments per social
   session = 2 rounds × 6). Reads `session_id, cutoff, segment_type,
   round_index, theta_predicted` and the readout columns `expert_betas`,
   `novice_betas`, `recent_betas` (each a `'[(a1,b1), …]'` string parsed by
   `parse_betas` into `(K, 2)`); `recent` is absent at cutoff 1 and kept as
   None. The model-mode twin `experiment_cutoff_model_predictions.csv` is **not**
   read by the analysis.
3. `metadata/inference_config.yaml` — read by `inference-check` (asserts
   `experiment_data_path` matches this version's preprocessed file), by
   `checkpoint_path` (relocates the recorded `checkpoint_path` under the current
   machine's output root, taking `eval_dir`'s grandparent as the old root, so the
   metadata stays a record), and by `checks_setup` (`wandb_artifact`).

Structural facts `load_sessions` asserts: social sessions only; 2 rounds per
session; `N_ROUNDS * GAMES_PER_ROUND * trials_per_game` = 2 × 4 × 6 = 48 shots
per session; one `anonymous_id` per session; `demonstrator_type` present exactly
on watch games (`game_index % 4 ∈ {1,3}` are the play games); cutoffs exactly
`1..12`; all readouts `(K, 2)` with `expert` never missing. Sessions where any
game departs from the modal trial count are dropped and recorded in
`Sessions.dropped`.

`trial_data(Sessions) → TrialData` then expands to one entry per human play shot,
with `read[{expert,novice,recent}] (N, K, 2)` taken at cutoff
`6*round + game_index%4 + 1` (the readout right after the demonstration
preceding that play game), `own_seg` / `own_prev` own-count arrays,
`vis_last` / `vis_expert` / `vis_round` visible-pull arrays, `prev_arm` /
`prev_rew`, `round_pm` (∓0.5), `seg` (1–4), `expert_first`, `last_exp`,
`expert_seen`, `theta (N, K)` and `part` (participant index into the sorted
`people` tuple). The identical builder is reused for simulated data via
`simulated_trial_data` in `src/analysis/predictive.py` — which is why the
fitting code runs unchanged on simulations.

---

## 7. Cluster operations

### Sherlock: analysis at full size

One-time setup from a login node inside the clone:

```bash
sh_dev -c 4 -m 16GB -t 2:00:00     # the installs are too heavy for a login node
bash scripts/setup_sherlock.sh     # miniforge + env beside the clone under
                                   # $GROUP_HOME/$USER/SocialBanditsProject,
                                   # data roots under $SCRATCH/SocialBanditsProject,
                                   # writes .env
```

Sherlock's nodes run CentOS 7 (glibc 2.17), too old for the pinned jax, torch,
numpy and scipy wheels, which want ≥ 2.27. The setup therefore also pulls a
Rocky 9 Apptainer image beside the env, and `.env` names it as `CONTAINER`.
`scripts/run_analysis.sh` and the SLURM tasks re-execute inside it, and
`bash scripts/with_env.sh <command>` (or no command, for a shell) runs anything
else there with the env active:

```bash
bash scripts/with_env.sh python src/run_analysis.py --version pilot_E --list-units
bash scripts/with_env.sh pytest tests/test_numbers.py -q
```

The image is defined by `scripts/sherlock.def` and built with
`apptainer build $ENV_ROOT/rocky9.sif scripts/sherlock.def`. `setup_sherlock.sh`
also runs `renv::restore()` inside the container with
`STRINGI_LDFLAGS="-Wl,--allow-shlib-undefined" MAKEFLAGS=-j$SLURM_CPUS_PER_TASK`
and points `RENV_PATHS_CACHE` at group storage, since Sherlock sets
`$XDG_CACHE_HOME` to `/tmp`.

### Transferring one version's inputs

Only the anonymised preprocessed data ever leaves the lab machine; the raw files
and the ID map stay there.

```bash
S=/scratch/users/<sunet>/SocialBanditsProject   # Sherlock's $SCRATCH, spelled out:
                                                # the variable is not set on the lab machine
ssh sherlock "mkdir -p $S/experiments/pilot_E $S/output/eval/pilot_E \
              $S/output/training_runs/<run>/checkpoints"   # rsync creates only the last component

# never the whole folder: it holds the raw files and the ID map
rsync -av $EXPERIMENT_DATA_DIR/pilot_E/{preprocessed_participant_data.csv,exclusions.csv} \
      sherlock:$S/experiments/pilot_E/
rsync -av $OUTPUT_DATA_DIR/eval/pilot_E/ sherlock:$S/output/eval/pilot_E/
rsync -av $OUTPUT_DATA_DIR/training_runs/<run>/checkpoints/<ckpt>.pt \
      sherlock:$S/output/training_runs/<run>/checkpoints/
```

### Submitting

```bash
bash scripts/slurm/submit.sh --version pilot_E [--dev] [--dry-run]
```

It lists the cached units (`python src/run_analysis.py --list-units`), submits
one dependent array per level, then the final full-stage run. Logs and unit lists
land in `$OUTPUT_DATA_DIR/analysis/<version>/slurm/`.

`.env` knobs: `SLURM_CPU_PARTITION` (default `normal`) for table blocks and
assembly; `SLURM_GPU_PARTITION` (`owners`) for fits, simulated datasets and PPC
draws, one GPU each and preemptible so tasks requeue; `SLURM_FINAL_PARTITION`
(`hns`, a long CPU partition) for the final run of every stage on the filled
caches; `SLURM_ACCOUNT` (optional); `N_TABLE_BLOCKS` (8).

Unit levels, from `units(p)` in `run_analysis.py` — each unit fills exactly one
cache piece, one writer per file, and the stages then run over the full caches:

| level | units | where |
|---|---|---|
| 0 | `table:<source>_<κ>_<own>:<b>/<n>` | CPU, `N_TABLE_BLOCKS` row blocks per table |
| 1 | `assemble:…` | CPU |
| 2 | `fit:<model>` | GPU |
| 3 | `loso:<model>`, `recovery:<d>`, `model_recovery:<comparison>:<side>:<d>`, `split_fit:<model>`, `ppc:<model>:<i>` | GPU |
| 4 | `split_half:<model>:<i>` | GPU |

Run one by hand with `python src/run_analysis.py --version <v> --unit fit:HOS`,
or via the array script:
`SLURM_ARRAY_TASK_ID=<i> UNIT_LIST=<file> VERSION=<v> bash scripts/slurm/units.sbatch`.

Preempted tasks resume from the atomic caches (§6). **`$SCRATCH` purges files
untouched for 90 days**, so copy `analysis/<version>/paper/` — and the fits, if
wanted — to group storage after a run.

### Training on a cluster

Three paths, depending on the machine:

```bash
# Single-node multi-GPU DDP. Uses ENV_FILE=.env.cluster, derives NUM_GPUS from
# the allocation. NOTE: the script still has /PLACEHOLDER/slurm-logs/ output
# paths and commented --account/--partition that must be filled in.
sbatch scripts/run_training.sbatch
sbatch --gpus-per-node=2 --time=4:00:00 scripts/run_training.sbatch

# Marlowe preempt partition: auto-requeue into a single wandb run, runs
# run_training.sh inside Apptainer. Compute nodes have no outbound internet,
# so wandb runs offline and is synced once afterwards from a login node.
sbatch scripts/sbatch_run_training.sh        # env: IMAGE, REPO
scripts/wandb_sync.sh episode-<jobid>        # pushes the run + dataset and
                                             # best-checkpoint artifacts

# One job = one long-running wandb sweep agent.
wandb sweep src/Sweeps/Sweep_Version_EnvThetas.yaml
sbatch scripts/run_sweep_agent.sbatch <entity/project/sweep_id>
sbatch --array=1-50 scripts/run_sweep_agent.sbatch <sweep_id> --count=1
```

`--gpus-per-node` must stay matched to the sweep YAML's `--nproc_per_node`
(currently 2). `.env.cluster.example` differs from `.env.example` only by
`WANDB_MODE=offline` and a commented `WANDB_API_KEY` (set it if `~/.netrc` is not
propagated to compute nodes). Opt in with
`ENV_FILE=.env.cluster bash scripts/<script>.sh`; `run_training.sbatch` sets it
automatically.

### Continuing a training run

```bash
bash scripts/continue_training.sh --checkpoint <path|run_id> --source <local|wandb> [overrides]
```

`--checkpoint` and `--source` are required. `--resume_run` continues the **same**
wandb run so metrics append; without it a new run is created for clean
separation. The script detects an already-complete run, and loads model,
optimizer and scheduler state from the checkpoint.

```bash
# more epochs on a wandb run
bash scripts/continue_training.sh --checkpoint <run_id> --source wandb --epochs 150

# same run, lower LR (metrics stay continuous)
bash scripts/continue_training.sh --checkpoint <run_id> --source wandb \
     --resume_run --epochs 150 --lr 0.00001

# from a local file
bash scripts/continue_training.sh \
     --checkpoint checkpoints/<run_id>/epoch_050_best.pt --source local --epochs 100
```

Pass-through overrides: `--epochs --lr --scheduler --optimizer --weight_decay
--batch_size --warmup_epochs --min_lr_ratio --patience --factor
--early_stopping_patience --dropout --nhead --num_layers --dim_feedforward
--d_model --demo_loss_weight --demo_belief_loss --experience_loss_weight
--dataset_name --device --use_scheduler`.

Checkpoint loading itself goes through `BaseSocialTransformer.load_checkpoint`
(`src/models/base.py`), which takes **exactly one** of `local_path` or
`wandb_artifact` (`entity/project/artifact:alias`). wandb downloads are cached
under `$OUTPUT_DATA_DIR/training_runs/wandb_<id>/checkpoints/` and reused.

### Recovering a sweep

```bash
bash scripts/complete_sweep.sh <sweep_id>
```

Wraps `src/complete_sweep.py` (which needs `--entity` and `--project`
explicitly; the shell wrapper fills them from `.env`). It queries the sweep for
incomplete runs, lists them with status and progress, lets you select which to
resume (`1,3,5` or `all`), and resumes each with its **original** configuration.

It filters on wandb's own run state — `crashed` (terminated unexpectedly),
`failed` (explicitly marked), `killed` (manually stopped) are offered;
`finished` and `running` are excluded. A run that stopped via early stopping is
marked `finished` and will not be resumed.

This only resumes **existing** incomplete runs; it never starts a new
hyperparameter configuration. To continue a sweep with new configs, use
`wandb agent <sweep-id>`. To see what a finished sweep found:
`python src/analyze_sweep.py --sweep_id <entity/project/id> [--top_k N]`.
