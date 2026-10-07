"""Train a LeWM world model (the official LeWM training loop), here on multi-agent Push-T data.

Changes from LeWM's train.py, each needed to transfer a single-agent model to the multi-agent data:
  * init.ckpt     start from the official single-agent checkpoint (fine-tuning)
  * widen_action_encoder   load it into a model whose action is [self, partner] (joint-action LeWM)
  * split_by_episode       validation holds out whole scenes (the random window split leaks near-copies)
  * z-scores from assets/pusht_stats.json, the official data's, so inputs keep their meaning

    python train.py                                   # fine-tune on own actions  -> checkpoints/lewm_coop2_ftfull
    python train.py data=coop_joint output_model_name=lewm_coop2_joint
"""
import os
from functools import partial
from pathlib import Path

os.environ.setdefault("STABLEWM_HOME", str(Path(__file__).resolve().parent / "data"))

import h5py
import hydra
import lightning as pl
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from module import SIGReg
from utils import get_img_preprocessor, get_stats_normalizer, SaveCkptCallback


def split_by_episode(dataset, train_split, seed):
    """Train/val split of whole scenes: collect.py writes one episode per agent, and the agents' episodes of a
    scene show the same events, so they go to the same side."""
    with h5py.File(dataset.h5_path, "r") as f:
        offsets = f["ep_offset"][:]
        group = f["scene_idx"][:][offsets] if "scene_idx" in f else np.arange(len(offsets))
    units = np.random.default_rng(seed).permutation(np.unique(group))
    val_units = units[: round(len(units) * (1 - train_split))]
    is_val = np.isin(group[[ep for ep, _ in dataset.clip_indices]], val_units)
    return spt.data.Subset(dataset, np.nonzero(~is_val)[0].tolist()), spt.data.Subset(dataset, np.nonzero(is_val)[0].tolist())


def widen_action_encoder(state, model, frameskip):
    """Load a single-agent action encoder into a joint-action one, keeping its function.

    The Embedder packs a step's frameskip actions time-major through a 1x1 conv and an MLP. Source column
    k*a + d (env step k, coordinate d of the a-dim action) becomes k*A + d for the A-dim joint action. The
    new conv rows keep their random init and the MLP's columns for them start at zero, so the widened model
    computes exactly the source function and gradient still reaches the new path.
    """
    own = model.state_dict()
    conv, lin = "action_encoder.patch_embed", "action_encoder.embed.0"
    if state[f"{conv}.weight"].shape == own[f"{conv}.weight"].shape:
        return state
    w, b, lw = state[f"{conv}.weight"], state[f"{conv}.bias"], state[f"{lin}.weight"]
    s_out, s_in = w.shape[:2]
    a, A = s_in // frameskip, own[f"{conv}.weight"].shape[1] // frameskip
    cols = torch.tensor([k * A + d for k in range(frameskip) for d in range(a)])
    new_w, new_b, new_lw = own[f"{conv}.weight"].clone(), own[f"{conv}.bias"].clone(), own[f"{lin}.weight"].clone()
    new_w[:s_out] = 0.0
    new_w[:s_out, cols] = w
    new_b[:s_out] = b
    new_lw.zero_()
    new_lw[:, :s_out] = lw
    return {**state, f"{conv}.weight": new_w, f"{conv}.bias": new_b, f"{lin}.weight": new_lw}


def lejepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses."""

    ctx_len = cfg.history_size
    n_preds = cfg.num_preds
    lambd = cfg.loss.sigreg.weight

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)

    emb = output["emb"]  # (B, T, D)
    act_emb = output["act_emb"]

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, : ctx_len]

    tgt_emb = emb[:, n_preds:] # label
    pred_emb = self.model.predict(ctx_emb, ctx_act) # pred

    # LeWM loss
    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    output["sigreg_loss"]= self.sigreg(emb.transpose(0, 1))
    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]  

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output

@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    #########################
    ##       dataset       ##
    #########################

    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop("name")
    dataset = swm.data.load_dataset(dataset_name, transform=None, **dataset_cfg)
    transforms = [get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)]

    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            tile = dataset.get_dim(col) // 2 if col == "action" else 1  # joint action [self, partner]: per-agent z-scores
            transforms.append(get_stats_normalizer(col, tile))

        cfg.model.action_encoder.input_dim = cfg.data.dataset.frameskip * dataset.get_dim("action")

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = split_by_episode(dataset, cfg.train_split, cfg.seed)
    print(f"{len(train_set)} train / {len(val_set)} val windows (whole scenes held out)", flush=True)

    train = torch.utils.data.DataLoader(train_set, **cfg.loader, shuffle=True, drop_last=True, generator=rnd_gen)
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)

    ##############################
    ##       model / optim      ##
    ##############################

    world_model = hydra.utils.instantiate(cfg.model)
    if cfg.init_ckpt:
        ckpt = Path(swm.data.utils.get_cache_dir(sub_folder='checkpoints'), cfg.init_ckpt)
        world_model.load_state_dict(widen_action_encoder(torch.load(ckpt, map_location="cpu"), world_model, cfg.data.dataset.frameskip))

    optimizers = {
        'model_opt': {
            "modules": 'model',
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model = world_model,
        sigreg = SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    run_dir = Path(swm.data.utils.get_cache_dir(sub_folder='checkpoints'), cfg.output_model_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[SaveCkptCallback(run_name=cfg.output_model_name, cfg=cfg.model, epoch_interval=cfg.save_every)],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=False,
    )
    spt.set(cache_dir=str(run_dir / "spt"))
    manager = spt.Manager(trainer=trainer, module=world_model, data=data_module)
    manager()


if __name__ == "__main__":
    run()
