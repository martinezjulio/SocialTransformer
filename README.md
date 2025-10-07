
# SBI Transformer for Social Bandits (starter)

This is a **starter** implementation for training a **Transformer** offline (simulation‑based inference, SBI) to infer a posterior over
- the **demonstrator’s start‑of‑tail pseudo‑counts** *(pre‑episode latent history)* per arm, and
- the **environment payoffs** \(\theta_{1:K}\) (via a Beta head),
from **histories of observed actions and rewards** that may include **multiple prior bandits** (context) and a **current bandit** (prefix).

At decision time you can **sample** a coherent latent world hypothesis and act with **Thompson Sampling** (TS).

> ⚠️ This is intentionally minimalist and designed for you to extend. The key architectural choices (structured tokenization, causal masking, per‑arm heads) are in place.

---

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

## Project files

- `generate_simulations.py` — create synthetic episodes and build training examples.
- `dataloader.py` — dataset + collate to structured token tensors with masks.
- `model.py` — event embedder, causal transformer, per‑arm heads, TS helper.
- `train.py` — losses and training loop.
- `inference.py` — how to use a trained model to sample start‑of‑tail counts and a \(\theta\) vector for Thompson Sampling.

---

**Have fun extending!** The scaffolding is meant to make it easy to plug in your own loss terms (e.g., TS action likelihood), richer context encoders for transfer, and different simulation regimes.
