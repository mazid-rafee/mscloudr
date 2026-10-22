"""TRAIN-only gradient diagnostic for source-guided DB-CR checkpoints."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from mscloudr.bridge import get_bridge_schedule
from mscloudr.cli import eval as base
from mscloudr.cli.eval_source_guided_dbcr import _checkpoint_config
from mscloudr.data import (REFERENCE_PROTOCOL, build_reference_dataloaders,
    build_reference_datasets, discover_sen12mscr, load_ignored_sample_ids,
    reference_split_audit)
from mscloudr.endpoint_alignment import profile_endpoint_alignment
from mscloudr.models.source_guided_dbcr import SourceGuidedDBCRNet
from mscloudr.reproducibility import seed_everything

def build_parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--data-root",default=None)
    p.add_argument("--ignore-file",default="manifests/known_invalid_samples.txt")
    p.add_argument("--output-dir",required=True)
    p.add_argument("--device",required=True)
    p.add_argument("--batch-size",type=int,default=1)
    p.add_argument("--num-workers",type=int,default=2)
    p.add_argument("--max-batches",type=int,default=4)
    p.add_argument("--num-bins",type=int,default=8)
    p.add_argument("--sampler-seed",type=int,default=42)
    p.add_argument("--overwrite",action="store_true")
    p.add_argument("--allow-smoke-checkpoint",action="store_true")
    return p

def run(args):
    checkpoint=Path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    target=Path(args.output_dir)/"alignment.json"
    if target.exists() and not args.overwrite:
        raise FileExistsError(target)
    payload=base._torch_load_payload(checkpoint)
    metadata=payload["run_metadata"]
    config=_checkpoint_config(metadata)
    if metadata.get("paper_grade") is False and not args.allow_smoke_checkpoint:
        raise ValueError("smoke checkpoint requires --allow-smoke-checkpoint")
    device=torch.device(args.device)
    if device.type=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    train_seed=int(metadata["reproducibility"]["seeds"]["train_seed"])
    seed_everything(train_seed)
    ignored=load_ignored_sample_ids(args.ignore_file)
    discovered=discover_sen12mscr(args.data_root,ignored_sample_ids=ignored,
                                  strict_counterparts=True)
    datasets=build_reference_datasets(discovered.samples,include_sample_id=True,
                    strict_channels=True,verify_frozen_membership=True)
    audit=reference_split_audit({k:v.samples for k,v in datasets.as_dict().items()})
    if metadata.get("split_protocol") != REFERENCE_PROTOCOL or audit != metadata.get("split_audit"):
        raise ValueError("dataset split fingerprint differs from checkpoint")
    loaders=build_reference_dataloaders(datasets,batch_size=args.batch_size,
                    num_workers=args.num_workers,train_seed=train_seed,
                    pin_memory=(device.type=="cuda"))
    model=SourceGuidedDBCRNet(config).to(device)
    model.load_state_dict(payload["model_state"],strict=True)
    if model.architecture_audit()["trainable_parameters"]!=metadata.get("trainable_parameters"):
        raise ValueError("architecture mismatch")
    result=profile_endpoint_alignment(model,loaders.train,
        schedule=get_bridge_schedule("original"),total_steps=config.total_steps,
        device=device,max_batches=args.max_batches,num_bins=args.num_bins,
        sampler_seed=args.sampler_seed)
    result.update({"checkpoint":str(checkpoint),"checkpoint_epoch":int(payload["epoch"]),
          "model_identity":metadata["model_identity"],"split_protocol":REFERENCE_PROTOCOL,
          "split_audit":audit,"schedule_name":"original","train_seed":train_seed,
          "architecture_config":config.audit_config(),"paper_grade":False})
    target.parent.mkdir(parents=True,exist_ok=True)
    temporary=target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    temporary.replace(target)
    print(json.dumps({"output":str(target),"cosines":[round(b["cosine"]["mean"],5)
         for b in result["bins"]]},indent=2))
    return result

def main():
    run(build_parser().parse_args())
if __name__=="__main__":
    main()
