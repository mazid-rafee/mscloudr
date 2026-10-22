"""Unit tests for train-only gradient diagnostic (no dataset required)."""
import copy
import pytest
import torch
from torch import nn
from mscloudr.bridge import get_bridge_schedule
from mscloudr.endpoint_alignment import gradient_alignment,profile_endpoint_alignment

class TinyBridgeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale=nn.Parameter(torch.tensor(.3))
        self.bias=nn.Parameter(torch.tensor(.1))
    def forward(self,x_t,t,sar):
        return self.scale*x_t+self.bias*t[:,None,None,None]/16+.1*sar

def batches():
    g=torch.Generator().manual_seed(12)
    return [{"target":torch.randn(2,1,4,4,generator=g),
             "cloudy":torch.randn(2,1,4,4,generator=g),
             "sar":torch.randn(2,1,4,4,generator=g)} for _ in range(3)]

def test_cosine_and_unused_parameters():
    r=gradient_alignment((torch.tensor([1.,0.]),None),
                         (torch.tensor([0.,2.]),torch.tensor([3.])))
    assert r["cosine"]==0
    assert r["candidate_grad_norm"]==pytest.approx(13**.5)
    assert gradient_alignment((torch.tensor([1.]),),(torch.tensor([-1.]),))["cosine"]==-1
    with pytest.raises(ValueError,match="zero gradient"):
        gradient_alignment((torch.zeros(1),),(torch.ones(1),))

def test_read_only_and_reproducible():
    model=TinyBridgeNet().train()
    before=copy.deepcopy(model.state_dict())
    opts=dict(schedule=get_bridge_schedule("original"),total_steps=16,
              max_batches=2,num_bins=8,sampler_seed=42,device="cpu")
    a=profile_endpoint_alignment(model,batches(),**opts)
    b=profile_endpoint_alignment(model,batches(),**opts)
    assert a==b
    assert a["split_used"]=="train_only"
    assert a["interior_t_includes_endpoint"] is False
    assert a["bins"][-1]["t_end_exclusive"]==16
    assert len(a["bins"])==8 and a["num_batches"]==2
    assert model.training
    assert all(p.grad is None for p in model.parameters())
    assert all(torch.equal(before[k],model.state_dict()[k]) for k in before)

def test_empty_loader_restores_mode():
    model=TinyBridgeNet().eval()
    with pytest.raises(ValueError,match="no batches"):
        profile_endpoint_alignment(model,[],schedule=get_bridge_schedule("original"),
                                   total_steps=16,device="cpu")
    assert not model.training

def test_bad_bin_count():
    with pytest.raises(ValueError,match="num_bins"):
        profile_endpoint_alignment(TinyBridgeNet(),batches(),
                      schedule=get_bridge_schedule("original"),total_steps=4,
                      num_bins=8,device="cpu")
