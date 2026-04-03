"""
Loss functions for demonstrator belief predictions and theta payoffs.
"""
import torch
import torch.nn.functional as F


def beta_nll_of_true_theta(alpha_pred: torch.Tensor, beta_pred: torch.Tensor, theta_true: torch.Tensor) -> torch.Tensor:
    """
    Negative log-likelihood of the true theta under the predicted Beta(alpha_pred, beta_pred).
    
    This loss is used for the theta (payoff) predictions, where we want to fit a Beta distribution
    over the true Bernoulli payoff parameters.
    
    Args:
        alpha_pred: Predicted alpha parameters [B, K]
        beta_pred: Predicted beta parameters [B, K]
        theta_true: True theta values in [0, 1] [B, K]
    
    Returns:
        Scalar loss value (mean NLL)
    """
    eps = 1e-6
    theta_true = theta_true.clamp(eps, 1 - eps)
    logB = torch.lgamma(alpha_pred) + torch.lgamma(beta_pred) - torch.lgamma(alpha_pred + beta_pred)
    nll = -((alpha_pred - 1) * torch.log(theta_true) + (beta_pred - 1) * torch.log(1 - theta_true) - logB)
    return nll.mean()


def log_mse_loss(pred_alpha: torch.Tensor, pred_beta: torch.Tensor, 
                 target_alpha: torch.Tensor, target_beta: torch.Tensor) -> torch.Tensor:
    """
    MSE loss in log-space to compare positive parameters across scales.
    
    This loss is scale-dependent and benefits from normalization of the parameters.
    
    Args:
        pred_alpha: Predicted alpha parameters [B, num_demos, K] or [B, K]
        pred_beta: Predicted beta parameters [B, num_demos, K] or [B, K]
        target_alpha: Target alpha parameters (same shape as pred)
        target_beta: Target beta parameters (same shape as pred)
    
    Returns:
        Scalar loss value
    """
    eps = 1e-8
    loss_alpha = ((torch.log(pred_alpha + eps) - torch.log(target_alpha + eps)) ** 2).mean()
    loss_beta = ((torch.log(pred_beta + eps) - torch.log(target_beta + eps)) ** 2).mean()
    return (loss_alpha + loss_beta) / 2.0


def beta_kl_divergence(pred_alpha: torch.Tensor, pred_beta: torch.Tensor,
                       target_alpha: torch.Tensor, target_beta: torch.Tensor) -> torch.Tensor:
    """
    KL divergence between Beta distributions: KL(Beta(target) || Beta(pred)).
    
    Closed-form formula for KL divergence between two Beta distributions:
    KL(Beta(a1,b1) || Beta(a2,b2)) = 
        log[B(a2,b2)/B(a1,b1)] + (a1-a2)*ψ(a1) + (b1-b2)*ψ(b1) + (a2+b2-a1-b1)*ψ(a1+b1)
    
    where B(a,b) = Γ(a)Γ(b)/Γ(a+b) is the Beta function and ψ is the digamma function.
    
    This loss does not require normalization.
    
    Args:
        pred_alpha: Predicted alpha parameters [B, num_demos, K] or [B, K]
        pred_beta: Predicted beta parameters [B, num_demos, K] or [B, K]
        target_alpha: Target alpha parameters (same shape as pred)
        target_beta: Target beta parameters (same shape as pred)
    
    Returns:
        Scalar loss value (mean KL divergence)
    """
    # Compute log Beta function: log B(a,b) = log Γ(a) + log Γ(b) - log Γ(a+b)
    def log_beta(a, b):
        return torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b)
    
    # KL formula components
    log_B_ratio = log_beta(pred_alpha, pred_beta) - log_beta(target_alpha, target_beta)
    
    digamma_target_alpha = torch.digamma(target_alpha)
    digamma_target_beta = torch.digamma(target_beta)
    digamma_target_sum = torch.digamma(target_alpha + target_beta)
    
    kl = (log_B_ratio 
          + (target_alpha - pred_alpha) * digamma_target_alpha
          + (target_beta - pred_beta) * digamma_target_beta
          + (pred_alpha + pred_beta - target_alpha - target_beta) * digamma_target_sum)
    
    return kl.mean()


def expected_value_mse(pred_alpha: torch.Tensor, pred_beta: torch.Tensor,
                       target_alpha: torch.Tensor, target_beta: torch.Tensor) -> torch.Tensor:
    """
    MSE loss on the expected values of Beta distributions.
    
    For Beta(α, β), E[θ] = α/(α+β), which is naturally in [0,1].
    This loss does not require normalization and is simple and interpretable.
    
    Downside: loses information about uncertainty (doesn't distinguish 
    Beta(10,10) from Beta(100,100)).
    
    Args:
        pred_alpha: Predicted alpha parameters [B, num_demos, K] or [B, K]
        pred_beta: Predicted beta parameters [B, num_demos, K] or [B, K]
        target_alpha: Target alpha parameters (same shape as pred)
        target_beta: Target beta parameters (same shape as pred)
    
    Returns:
        Scalar loss value
    """
    eps = 1e-8  # For numerical stability
    pred_mean = pred_alpha / (pred_alpha + pred_beta + eps)
    target_mean = target_alpha / (target_alpha + target_beta + eps)
    return F.mse_loss(pred_mean, target_mean)


def beta_cross_entropy(pred_alpha: torch.Tensor, pred_beta: torch.Tensor,
                       target_alpha: torch.Tensor, target_beta: torch.Tensor) -> torch.Tensor:
    """
    Cross-entropy loss between Beta distributions: -E_{θ~Beta(target)}[log p(θ|pred)].
    
    This is equivalent to:
    H(target, pred) = -∫ Beta(θ; target) * log Beta(θ; pred) dθ
                    = -E_{θ~Beta(target)}[log Beta(θ; pred)]
    
    Closed form:
    CE = log B(pred_α, pred_β) - (pred_α-1)*ψ(target_α) - (pred_β-1)*ψ(target_β) 
         + (pred_α + pred_β - 2)*ψ(target_α + target_β)
    
    where B is Beta function and ψ is digamma.
    
    This loss does not require normalization.
    
    Args:
        pred_alpha: Predicted alpha parameters [B, num_demos, K] or [B, K]
        pred_beta: Predicted beta parameters [B, num_demos, K] or [B, K]
        target_alpha: Target alpha parameters (same shape as pred)
        target_beta: Target beta parameters (same shape as pred)
    
    Returns:
        Scalar loss value
    """
    # log B(a,b) = log Γ(a) + log Γ(b) - log Γ(a+b)
    log_beta_pred = torch.lgamma(pred_alpha) + torch.lgamma(pred_beta) - torch.lgamma(pred_alpha + pred_beta)
    
    digamma_target_alpha = torch.digamma(target_alpha)
    digamma_target_beta = torch.digamma(target_beta)
    digamma_target_sum = torch.digamma(target_alpha + target_beta)
    
    # E_{θ~target}[log Beta(θ; pred)]
    expected_log_prob = (
        -log_beta_pred
        + (pred_alpha - 1) * digamma_target_alpha
        + (pred_beta - 1) * digamma_target_beta
        - (pred_alpha + pred_beta - 2) * digamma_target_sum
    )
    
    # Cross-entropy is the negative of this
    cross_entropy = -expected_log_prob
    
    return cross_entropy.mean()


def experience_loss_log(pred_alpha: torch.Tensor, pred_beta: torch.Tensor,
                       target_alpha: torch.Tensor, target_beta: torch.Tensor,
                       alpha0: float = 1.0, beta0: float = 1.0,
                       eps: float = 1.0) -> torch.Tensor:
    """
    Log-space MSE loss on total experience N_tot.
    
    Computes N_tot = sum over arms of (alpha - alpha0) + (beta - beta0)
    Then applies MSE loss in log space to handle varying scales (novices vs experts).
    
    Args:
        pred_alpha: Predicted alpha [B, K]
        pred_beta: Predicted beta [B, K]
        target_alpha: Target alpha [B, K]
        target_beta: Target beta [B, K]
        alpha0: Prior alpha parameter
        beta0: Prior beta parameter
        eps: Small constant for log stability (default: 1.0)
    
    Returns:
        Scalar loss value
    """
    K = pred_alpha.shape[-1]
    
    # Compute N_tot from predictions
    pred_N_tot = (pred_alpha.sum(dim=-1) - alpha0 * K + 
                  pred_beta.sum(dim=-1) - beta0 * K)
    
    # Compute N_tot from targets
    target_N_tot = (target_alpha.sum(dim=-1) - alpha0 * K + 
                    target_beta.sum(dim=-1) - beta0 * K)
    
    # Clamp to non-negative to avoid log of negative values
    # (can happen with predicted values early in training)
    pred_N_tot = torch.clamp(pred_N_tot, min=0.0)
    target_N_tot = torch.clamp(target_N_tot, min=0.0)
    
    # MSE in log space (add eps to handle N=0 case for novices)
    return F.mse_loss(torch.log(pred_N_tot + eps), 
                      torch.log(target_N_tot + eps))


def get_demo_belief_loss(loss_type: str):
    """
    Get the appropriate loss function for demonstrator belief predictions.
    
    Args:
        loss_type: One of ["log_mse", "kl_divergence", "expected_value_mse", "cross_entropy"]
    
    Returns:
        Loss function that takes (pred_alpha, pred_beta, target_alpha, target_beta) -> scalar
    
    Raises:
        ValueError: If loss_type is not recognized
    """
    losses = {
        "log_mse": log_mse_loss,
        "kl_divergence": beta_kl_divergence,
        "expected_value_mse": expected_value_mse,
        "cross_entropy": beta_cross_entropy,
    }
    
    if loss_type not in losses:
        raise ValueError(
            f"Unknown loss type: {loss_type}. "
            f"Must be one of {list(losses.keys())}"
        )
    
    return losses[loss_type]


def requires_normalization(loss_type: str) -> bool:
    """
    Check if a loss function requires input normalization.
    
    Args:
        loss_type: One of ["log_mse", "kl_divergence", "expected_value_mse", "cross_entropy"]
    
    Returns:
        True if normalization is recommended for this loss type
    """
    # Only log_mse benefits from normalization
    return loss_type == "log_mse"


def experience_head_mse_loss(
    pred_N_tot: torch.Tensor,
    target_alpha: torch.Tensor,
    target_beta: torch.Tensor,
    alpha0: float,
    beta0: float,
    K: int,
    eps: float = 1.0
) -> torch.Tensor:
    """
    Direct MSE loss on N_tot predictions from experience head (continuous).
    
    Args:
        pred_N_tot: [B, num_demonstrators] - continuous N_tot predictions
        target_alpha: [B, num_demonstrators, K]
        target_beta: [B, num_demonstrators, K]
        alpha0: prior alpha parameter
        beta0: prior beta parameter
        K: number of arms
        eps: for log stability
    
    Returns:
        Scalar loss
    """
    # Compute target N_tot
    target_N_tot = (target_alpha.sum(dim=-1) - alpha0 * K + 
                    target_beta.sum(dim=-1) - beta0 * K)
    target_N_tot = torch.clamp(target_N_tot, min=0.0)
    
    # MSE in log space
    return F.mse_loss(torch.log(pred_N_tot + eps), 
                      torch.log(target_N_tot + eps))


def experience_head_bce_loss(
    pred_logits: torch.Tensor,
    target_alpha: torch.Tensor,
    target_beta: torch.Tensor,
    alpha0: float,
    beta0: float,
    K: int,
    expert_threshold: float = 30.0
) -> torch.Tensor:
    """
    Binary cross-entropy loss for expert vs novice classification.
    
    Args:
        pred_logits: [B, num_demonstrators] - logits (before sigmoid)
        target_alpha: [B, num_demonstrators, K]
        target_beta: [B, num_demonstrators, K]
        alpha0: prior alpha parameter
        beta0: prior beta parameter
        K: number of arms
        expert_threshold: N_tot threshold for expert label (default: 30)
    
    Returns:
        Scalar loss
    """
    # Compute target N_tot
    target_N_tot = (target_alpha.sum(dim=-1) - alpha0 * K + 
                    target_beta.sum(dim=-1) - beta0 * K)
    target_N_tot = torch.clamp(target_N_tot, min=0.0)
    
    # Convert to binary labels (1 = expert, 0 = novice)
    is_expert = (target_N_tot > expert_threshold).float()
    
    # Binary cross-entropy with logits
    return F.binary_cross_entropy_with_logits(pred_logits, is_expert)


def get_experience_head_loss(loss_type: str):
    """
    Get the appropriate loss function for experience head predictions.
    
    Args:
        loss_type: "mse" or "bce"
    
    Returns:
        Loss function
    """
    losses = {
        "mse": experience_head_mse_loss,
        "bce": experience_head_bce_loss,
    }
    if loss_type not in losses:
        raise ValueError(f"Unknown experience head loss type: {loss_type}. Must be 'mse' or 'bce'")
    return losses[loss_type]
