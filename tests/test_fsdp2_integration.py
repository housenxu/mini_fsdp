"""Two-rank Transformer numerical and DTensor contract test for FSDP2.

Run from the repository root:
    PYTHONPATH=. torchrun --standalone --nproc_per_node=2 \
        tests/test_fsdp2_integration.py
"""

from __future__ import annotations

import copy
import datetime
import os

import torch
import torch.distributed as dist

from toy_fsdp import (
    FSDP2Config,
    TinyTransformerLM,
    TransformerBlock,
    TransformerConfig,
    apply_fsdp2,
    make_lm_batch,
)


def main() -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 2:
        raise RuntimeError("this test expects exactly two processes")
    dist.init_process_group(
        backend="gloo",
        init_method="env://",
        timeout=datetime.timedelta(minutes=2),
    )

    try:
        from torch.distributed.tensor import DTensor, Shard
    except ImportError:
        from torch.distributed._tensor import DTensor, Shard

    torch.manual_seed(2026)
    config = TransformerConfig(
        vocab_size=32,
        max_seq_len=8,
        hidden_dim=16,
        num_layers=2,
        num_heads=4,
        intermediate_dim=32,
        dropout=0.0,
        tie_embeddings=True,
        freeze_token_embedding=True,
    )
    model = TinyTransformerLM(config)
    assert model.token_embedding.weight is model.lm_head.weight
    reference = copy.deepcopy(model)
    runtime = apply_fsdp2(
        model,
        device_type="cpu",
        config=FSDP2Config(reshard_after_forward=True),
        wrap_cls=(TransformerBlock,),
    )

    original_names = tuple(reference.state_dict().keys())
    sharded_state = model.state_dict()
    assert tuple(sharded_state.keys()) == original_names
    assert model.token_embedding.weight is model.lm_head.weight
    assert not model.token_embedding.weight.requires_grad
    assert len(runtime.fsdp_modules) == config.num_layers
    assert runtime.root_has_parameter_group
    assert runtime.communication_group_count == config.num_layers + 1
    assert runtime.dtensor_parameter_count == sum(1 for _ in model.parameters())
    assert runtime.global_parameter_numel == sum(p.numel() for p in reference.parameters())
    assert runtime.local_parameter_numel < runtime.global_parameter_numel
    for parameter in model.parameters():
        assert isinstance(parameter, DTensor)
        assert len(parameter.placements) == 1
        assert isinstance(parameter.placements[0], Shard)

    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=1e-3,
        weight_decay=0.0,
    )
    reference_optimizer = torch.optim.AdamW(
        (parameter for parameter in reference.parameters() if parameter.requires_grad),
        lr=1e-3,
        weight_decay=0.0,
    )
    input_ids, labels = make_lm_batch(
        batch_size=3,
        seq_len=config.max_seq_len,
        vocab_size=config.vocab_size,
        device=torch.device("cpu"),
        seed=10_000 + rank,
    )

    reference_loss = reference(input_ids, labels)
    reference_loss.backward()
    for parameter in reference.parameters():
        if parameter.grad is not None:
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
            parameter.grad.div_(world_size)

    loss = model(input_ids, labels)
    torch.testing.assert_close(loss, reference_loss)
    loss.backward()
    for parameter in model.parameters():
        if parameter.requires_grad:
            assert isinstance(parameter.grad, DTensor)
        else:
            assert parameter.grad is None

    frozen_before = reference.token_embedding.weight.detach().clone()
    reference_optimizer.step()
    optimizer.step()
    torch.testing.assert_close(reference.token_embedding.weight, frozen_before)

    optimizer_state_tensors = [
        value
        for state in optimizer.state.values()
        for value in state.values()
        if isinstance(value, torch.Tensor) and value.numel() > 1
    ]
    assert optimizer_state_tensors
    assert all(isinstance(value, DTensor) for value in optimizer_state_tensors)

    reference_parameters = dict(reference.named_parameters())
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        sharded_optimizer_state = optimizer.state[parameter]
        full_optimizer_state = reference_optimizer.state[reference_parameters[name]]
        for state_name in ("exp_avg", "exp_avg_sq"):
            sharded_value = sharded_optimizer_state[state_name]
            assert isinstance(sharded_value, DTensor)
            torch.testing.assert_close(
                sharded_value.full_tensor(),
                full_optimizer_state[state_name],
                rtol=1e-5,
                atol=1e-6,
            )

    reference_state = reference.state_dict()
    for name, sharded_value in model.state_dict().items():
        assert isinstance(sharded_value, DTensor)
        full_value = sharded_value.full_tensor()
        torch.testing.assert_close(
            full_value,
            reference_state[name],
            rtol=1e-5,
            atol=1e-6,
        )

    layouts = runtime.parameter_layouts()
    assert set(layouts) == {name for name, _ in model.named_parameters()}
    assert len({tuple(value["global_shape"]) for value in layouts.values()}) > 2
    if rank == 0:
        print(
            "Transformer Block groups, tied weights, frozen parameters, DTensor "
            "gradients, optimizer state, and full-state checks passed"
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
