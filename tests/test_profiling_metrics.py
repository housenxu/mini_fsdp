from toy_fsdp.profiling import calculate_transformer_mfu, transformer_flops_per_token


def test_transformer_flops_per_token() -> None:
    assert transformer_flops_per_token(1_000, 2, 16, 8) == 6_000 + 12 * 2 * 16 * 8


def test_transformer_mfu_uses_per_gpu_peak() -> None:
    result = calculate_transformer_mfu(
        parameter_count=1_000,
        num_layers=2,
        hidden_dim=16,
        sequence_length=8,
        local_tokens=100,
        elapsed_seconds=2.0,
        vendor_peak_tflops=1.0,
        practical_peak_tflops=0.5,
    )
    expected_tflops = 50.0 * result.flops_per_token / 1e12
    assert result.model_tflops_per_gpu == expected_tflops
    assert result.formula_mfu_vs_vendor_percent == expected_tflops * 100
    assert result.efficiency_vs_practical_peak_percent == expected_tflops / 0.5 * 100