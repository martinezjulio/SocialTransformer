"""
ABC contract for the SocialTransformer family.

Every concrete model adheres to this contract:
- Implement ``forward``.
- Inherit ``save_checkpoint`` / ``load_checkpoint_from_local`` /
  ``load_checkpoint_from_wandb_artifact`` for checkpoint I/O.
"""

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn


class BaseSocialTransformer(nn.Module, ABC):
    """Skeleton contract for the SocialTransformer family.

    Cannot be instantiated directly — subclasses must implement ``forward``.
    All checkpoint I/O lives here so every model has identical save/load
    behavior.
    """

    @abstractmethod
    def forward(self, batch: Dict[str, torch.Tensor],
                custom_attn_mask: Optional[torch.Tensor] = None
                ) -> Dict[str, torch.Tensor]:
        ...

    # ---- save -----------------------------------------------------------

    def save_checkpoint(self, path: Union[str, Path], **extras) -> None:
        """Save model weights plus arbitrary extras to a .pt file.

        Args:
            path: Destination .pt path.
            **extras: Additional fields to merge into the saved dict
                (e.g. ``optimizer_state_dict=…``, ``epoch=…``,
                ``model_config=…``). The ``model_state_dict`` key is reserved
                for ``self.state_dict()`` and will be overridden if passed in.
        """
        torch.save({**extras, "model_state_dict": self.state_dict()}, path)

    # ---- load (classmethods) -------------------------------------------

    @classmethod
    def load_checkpoint(
        cls,
        *,
        local_path: Optional[Union[str, Path]] = None,
        wandb_artifact: Optional[str] = None,
        cache_dir: Optional[Union[str, Path]] = None,
    ) -> Tuple["BaseSocialTransformer", Dict]:
        """Router for the two load sources. Exactly one of ``local_path`` or
        ``wandb_artifact`` must be provided. ``cache_dir`` is required when
        loading from a wandb artifact (used as the local download/cache root).

        Returns ``(model, checkpoint_dict)``.
        """
        if (local_path is None) == (wandb_artifact is None):
            raise ValueError(
                "Provide exactly one of `local_path` or `wandb_artifact`."
            )
        if local_path is not None:
            return cls.load_checkpoint_from_local(local_path)
        if cache_dir is None:
            raise ValueError("`cache_dir` is required when loading from a wandb artifact.")
        return cls.load_checkpoint_from_wandb_artifact(wandb_artifact, cache_dir)

    @classmethod
    def load_checkpoint_from_local(
        cls, path: Union[str, Path]
    ) -> Tuple["BaseSocialTransformer", Dict]:
        """Build the right model class from a local .pt checkpoint and load state.

        Returns ``(model, checkpoint_dict)``; the dict carries any extras the
        caller saved (``optimizer_state_dict``, ``epoch``, ``model_config``, …).
        """
        path = str(path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        print(f"Loading checkpoint from local file: {path}")
        checkpoint = torch.load(path, map_location="cpu")
        model = cls._build_and_load(checkpoint)
        return model, checkpoint

    @classmethod
    def load_checkpoint_from_wandb_artifact(
        cls,
        artifact_string: str,
        cache_dir: Union[str, Path],
    ) -> Tuple["BaseSocialTransformer", Dict]:
        """Download a model from a wandb artifact (with local caching) and load it.

        artifact_string format: ``'entity/project/artifact:alias'``.
        Cached under ``cache_dir/wandb_<source_run_id>/checkpoints/``; if a
        ``.pt`` already exists there the download is skipped.
        """
        import wandb
        if artifact_string.count("/") != 2 or ":" not in artifact_string:
            raise ValueError(
                f"Invalid artifact string '{artifact_string}'. "
                "Expected 'entity/project/artifact:alias'."
            )

        print(f"Loading checkpoint from wandb artifact: {artifact_string}")
        api = wandb.Api()
        artifact = api.artifact(artifact_string, type="model")
        source_run = artifact.logged_by()
        run_id = source_run.id if source_run else "unknown_run"

        local_dir = os.path.join(str(cache_dir), f"wandb_{run_id}", "checkpoints")

        checkpoint_path = None
        if os.path.exists(local_dir):
            cached = [f for f in os.listdir(local_dir) if f.endswith(".pt")]
            if cached:
                checkpoint_path = os.path.join(local_dir, cached[0])
                print(f"Using cached checkpoint: {checkpoint_path}")

        if checkpoint_path is None:
            os.makedirs(local_dir, exist_ok=True)
            print(f"Artifact from run: {run_id} ({source_run.name if source_run else 'unknown'})")
            print(f"Downloading to: {local_dir}")
            artifact_dir = artifact.download(root=local_dir)
            files = [f for f in os.listdir(artifact_dir) if f.endswith(".pt")]
            if not files:
                raise FileNotFoundError(f"No .pt file in {artifact_dir}")
            checkpoint_path = os.path.join(artifact_dir, files[0])
            print(f"Downloaded checkpoint: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model = cls._build_and_load(checkpoint)
        return model, checkpoint

    # ---- internal helper ------------------------------------------------

    @staticmethod
    def _build_and_load(checkpoint: Dict) -> "BaseSocialTransformer":
        """Construct the right model from ``checkpoint['model_config']`` and
        load its ``model_state_dict`` (strict). Lazy import of ``build_model``
        avoids a circular dependency with ``models/__init__.py``.

        Accepts both the new ``model_type`` key and the legacy
        ``profile_method`` key for back-compat with pre-rename checkpoints.
        Legacy values are translated to current ones inside ``build_model``.
        """
        from models import build_model  # lazy
        if "model_config" not in checkpoint:
            raise ValueError(
                "Checkpoint is missing 'model_config' — cannot reconstruct the model."
            )
        model_config = checkpoint["model_config"]
        model_type = model_config.get("model_type") or model_config.get("profile_method", "social")
        model = build_model(model_type, model_config)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model
