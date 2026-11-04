"""
Unit and architecture validation tests for CIFAR-10 model implementations.
"""

from typing import List, Tuple
import pytest
import torch
import torch.nn as nn

from edge_vision.models import (
    resnet18_cifar,
    mobilenet_v3_small_cifar,
    count_parameters,
    compute_parameter_memory,
    get_model_summary,
    get_model,
    list_available_models,
    MODEL_REGISTRY,
)


def test_resnet18_cifar_structural_adaptations() -> None:
    """
    Verifies that ResNet-18 has the required CIFAR-10 structural modifications:
      - conv1: 3x3 kernel, stride 1, padding 1, no bias.
      - maxpool: replaced with nn.Identity.
      - fc: 512 input features, num_classes output features.
    """
    model = resnet18_cifar(num_classes=10)

    assert isinstance(model.conv1, nn.Conv2d)
    assert model.conv1.kernel_size == (3, 3)
    assert model.conv1.stride == (1, 1)
    assert model.conv1.padding == (1, 1)
    assert model.conv1.bias is None
    assert model.conv1.in_channels == 3
    assert model.conv1.out_channels == 64

    assert isinstance(model.maxpool, nn.Identity)

    assert isinstance(model.fc, nn.Linear)
    assert model.fc.in_features == 512
    assert model.fc.out_features == 10

    # Ensure standard residual stages are intact
    for stage_name in ["layer1", "layer2", "layer3", "layer4"]:
        stage = getattr(model, stage_name)
        assert isinstance(stage, nn.Sequential)
        assert len(stage) == 2  # Each stage in ResNet18 has 2 BasicBlocks


def test_mobilenet_v3_small_cifar_structural_adaptations() -> None:
    """
    Verifies that MobileNetV3-Small has the required CIFAR-10 structural modifications:
      - features[0][0] (stem conv): stride 1 instead of stride 2.
      - classifier[3]: linear layer projecting 1024 -> num_classes.
    """
    model = mobilenet_v3_small_cifar(num_classes=10)

    # Stem convolution is the first sub-module of features[0]
    stem_conv = model.features[0][0]
    assert isinstance(stem_conv, nn.Conv2d)
    assert stem_conv.stride == (1, 1)
    assert stem_conv.kernel_size == (3, 3)
    assert stem_conv.in_channels == 3
    assert stem_conv.out_channels == 16

    # Verify features container contains 13 modules (stem, 11 inverted residuals, pointwise)
    assert len(model.features) == 13

    # Final projection layer in classifier
    final_linear = model.classifier[3]
    assert isinstance(final_linear, nn.Linear)
    assert final_linear.in_features == 1024
    assert final_linear.out_features == 10


@pytest.mark.parametrize("batch_size", [1, 2, 64])
def test_resnet18_cifar_forward_shapes(batch_size: int) -> None:
    """
    Verifies forward pass output shape and finiteness for ResNet-18 across batch sizes.
    """
    model = resnet18_cifar(num_classes=10)
    model.eval()

    inputs = torch.randn(batch_size, 3, 32, 32)
    with torch.no_grad():
        outputs = model(inputs)

    assert outputs.shape == (batch_size, 10)
    assert outputs.dtype == torch.float32
    assert torch.isfinite(outputs).all()


@pytest.mark.parametrize("batch_size", [1, 2, 64])
def test_mobilenet_v3_small_cifar_forward_shapes(batch_size: int) -> None:
    """
    Verifies forward pass output shape and finiteness for MobileNetV3-Small across batch sizes.
    """
    model = mobilenet_v3_small_cifar(num_classes=10)
    model.eval()

    inputs = torch.randn(batch_size, 3, 32, 32)
    with torch.no_grad():
        outputs = model(inputs)

    assert outputs.shape == (batch_size, 10)
    assert outputs.dtype == torch.float32
    assert torch.isfinite(outputs).all()


@pytest.mark.parametrize("model_fn", [resnet18_cifar, mobilenet_v3_small_cifar])
def test_models_train_and_eval_modes_batch_size_1(model_fn) -> None:
    """
    Verifies that models operate in both train() and eval() modes with batch_size=1
    without raising BatchNorm2d spatial dimension errors.
    """
    model = model_fn(num_classes=10)

    inputs = torch.randn(1, 3, 32, 32)

    # Train mode test (validates that N*H*W > 1 for all BatchNorm2d layers)
    model.train()
    outputs_train = model(inputs)
    assert outputs_train.shape == (1, 10)
    assert torch.isfinite(outputs_train).all()

    # Eval mode test
    model.eval()
    with torch.no_grad():
        outputs_eval = model(inputs)
    assert outputs_eval.shape == (1, 10)
    assert torch.isfinite(outputs_eval).all()


@pytest.mark.parametrize("model_fn", [resnet18_cifar, mobilenet_v3_small_cifar])
def test_models_output_logits_properties(model_fn) -> None:
    """
    Verifies that model outputs are unnormalized logits rather than softmax probabilities.
    """
    model = model_fn(num_classes=10)
    model.eval()

    inputs = torch.randn(4, 3, 32, 32)
    with torch.no_grad():
        logits = model(inputs)

    # Raw logits should not sum to 1.0 per sample
    row_sums = logits.sum(dim=-1)
    assert not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3)


def test_resnet18_intermediate_stage_shapes() -> None:
    """
    Locks and verifies intermediate activation shapes across all ResNet-18 stages on 32x32 inputs.
    """
    model = resnet18_cifar(num_classes=10)
    model.eval()

    batch_size = 2
    x = torch.randn(batch_size, 3, 32, 32)

    with torch.no_grad():
        # Stem
        x_stem = model.relu(model.bn1(model.conv1(x)))
        assert x_stem.shape == (batch_size, 64, 32, 32)

        # Maxpool is Identity
        x_mp = model.maxpool(x_stem)
        assert x_mp.shape == (batch_size, 64, 32, 32)

        # Stage 1
        x_l1 = model.layer1(x_mp)
        assert x_l1.shape == (batch_size, 64, 32, 32)

        # Stage 2 (downsampling to 16x16)
        x_l2 = model.layer2(x_l1)
        assert x_l2.shape == (batch_size, 128, 16, 16)

        # Stage 3 (downsampling to 8x8)
        x_l3 = model.layer3(x_l2)
        assert x_l3.shape == (batch_size, 256, 8, 8)

        # Stage 4 (downsampling to 4x4)
        x_l4 = model.layer4(x_l3)
        assert x_l4.shape == (batch_size, 512, 4, 4)

        # Avgpool (global pooling to 1x1)
        x_ap = model.avgpool(x_l4)
        assert x_ap.shape == (batch_size, 512, 1, 1)

        # Flatten + Linear
        x_flat = torch.flatten(x_ap, 1)
        assert x_flat.shape == (batch_size, 512)

        out = model.fc(x_flat)
        assert out.shape == (batch_size, 10)


def test_mobilenet_v3_small_intermediate_stage_shapes() -> None:
    """
    Locks and verifies intermediate activation shapes across all MobileNetV3-Small stages on 32x32 inputs.
    """
    model = mobilenet_v3_small_cifar(num_classes=10)
    model.eval()

    batch_size = 2
    x = torch.randn(batch_size, 3, 32, 32)

    expected_feature_shapes: List[Tuple[int, int, int, int]] = [
        (batch_size, 16, 32, 32),   # features[0]: stem conv (stride 1)
        (batch_size, 16, 16, 16),   # features[1]: IR 0 (stride 2)
        (batch_size, 24, 8, 8),     # features[2]: IR 1 (stride 2)
        (batch_size, 24, 8, 8),     # features[3]: IR 2 (stride 1)
        (batch_size, 40, 4, 4),     # features[4]: IR 3 (stride 2)
        (batch_size, 40, 4, 4),     # features[5]: IR 4 (stride 1)
        (batch_size, 40, 4, 4),     # features[6]: IR 5 (stride 1)
        (batch_size, 48, 4, 4),     # features[7]: IR 6 (stride 1)
        (batch_size, 48, 4, 4),     # features[8]: IR 7 (stride 1)
        (batch_size, 96, 2, 2),     # features[9]: IR 8 (stride 2)
        (batch_size, 96, 2, 2),     # features[10]: IR 9 (stride 1)
        (batch_size, 96, 2, 2),     # features[11]: IR 10 (stride 1)
        (batch_size, 576, 2, 2),    # features[12]: pointwise conv
    ]

    with torch.no_grad():
        curr = x
        for idx, layer in enumerate(model.features):
            curr = layer(curr)
            assert curr.shape == expected_feature_shapes[idx], (
                f"features[{idx}] shape mismatch: expected {expected_feature_shapes[idx]}, got {curr.shape}"
            )

        pooled = model.avgpool(curr)
        assert pooled.shape == (batch_size, 576, 1, 1)

        flat = torch.flatten(pooled, 1)
        assert flat.shape == (batch_size, 576)

        logits = model.classifier(flat)
        assert logits.shape == (batch_size, 10)


def test_parameter_counts_exact() -> None:
    """
    Verifies exact parameter counts for both models.
    """
    r18 = resnet18_cifar(num_classes=10)
    r18_counts = count_parameters(r18)
    assert r18_counts["total"] == 11_173_962
    assert r18_counts["trainable"] == 11_173_962
    assert r18_counts["non_trainable"] == 0

    mb3 = mobilenet_v3_small_cifar(num_classes=10)
    mb3_counts = count_parameters(mb3)
    assert mb3_counts["total"] == 1_528_106
    assert mb3_counts["trainable"] == 1_528_106
    assert mb3_counts["non_trainable"] == 0


def test_parameter_memory_exact() -> None:
    """
    Verifies exact parameter memory calculations (bytes, decimal MB, binary MiB).
    """
    r18 = resnet18_cifar(num_classes=10)
    r18_mem = compute_parameter_memory(r18)
    assert r18_mem["bytes"] == 44_695_848
    assert pytest.approx(r18_mem["megabytes_decimal"], rel=1e-5) == 44.695848
    assert pytest.approx(r18_mem["mebibytes_binary"], rel=1e-5) == 42.625282

    mb3 = mobilenet_v3_small_cifar(num_classes=10)
    mb3_mem = compute_parameter_memory(mb3)
    assert mb3_mem["bytes"] == 6_112_424
    assert pytest.approx(mb3_mem["megabytes_decimal"], rel=1e-5) == 6.112424
    assert pytest.approx(mb3_mem["mebibytes_binary"], rel=1e-5) == 5.829262


def test_count_parameters_with_frozen_layers() -> None:
    """
    Verifies that freezing layers correctly updates trainable and non-trainable counts.
    """
    model = resnet18_cifar(num_classes=10)

    # Freeze layer1 parameters
    layer1_param_count = sum(p.numel() for p in model.layer1.parameters())
    for p in model.layer1.parameters():
        p.requires_grad = False

    counts = count_parameters(model)
    assert counts["total"] == 11_173_962
    assert counts["non_trainable"] == layer1_param_count
    assert counts["trainable"] == 11_173_962 - layer1_param_count


def test_get_model_summary() -> None:
    """
    Verifies that get_model_summary combines parameter counts and memory correctly.
    """
    model = resnet18_cifar(num_classes=10)
    summary = get_model_summary(model)

    assert "total_parameters" in summary
    assert "trainable_parameters" in summary
    assert "non_trainable_parameters" in summary
    assert "parameter_bytes" in summary
    assert "parameter_mb" in summary
    assert "parameter_mib" in summary
    assert summary["total_parameters"] == 11_173_962
    assert summary["parameter_bytes"] == 44_695_848


def test_model_factory_registry_and_listing() -> None:
    """
    Verifies the model factory registry, available model listing, and instantiation.
    """
    available = list_available_models()
    assert "resnet18_cifar" in available
    assert "mobilenet_v3_small_cifar" in available
    assert available == sorted(list(MODEL_REGISTRY.keys()))

    # Instantiate default num_classes
    m1 = get_model("resnet18_cifar")
    assert isinstance(m1, nn.Module)
    assert m1.fc.out_features == 10

    m2 = get_model("mobilenet_v3_small_cifar")
    assert isinstance(m2, nn.Module)
    assert m2.classifier[3].out_features == 10

    # Instantiate custom num_classes
    m1_custom = get_model("resnet18_cifar", num_classes=100)
    assert m1_custom.fc.out_features == 100

    m2_custom = get_model("mobilenet_v3_small_cifar", num_classes=100)
    assert m2_custom.classifier[3].out_features == 100


def test_model_factory_invalid_name_raises_key_error() -> None:
    """
    Verifies that requesting an unregistered model architecture raises KeyError with a clear message.
    """
    with pytest.raises(KeyError, match="Unknown model architecture 'unsupported_model'"):
        get_model("unsupported_model")


def test_model_initialization_reproducibility() -> None:
    """
    Verifies that setting a manual seed produces identical initial model parameters.
    """
    torch.manual_seed(42)
    m1 = resnet18_cifar(num_classes=10)

    torch.manual_seed(42)
    m2 = resnet18_cifar(num_classes=10)

    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        assert torch.equal(p1, p2)

    torch.manual_seed(999)
    m3 = resnet18_cifar(num_classes=10)
    # Different seed should yield different weights
    assert not torch.equal(m1.conv1.weight, m3.conv1.weight)


def test_mps_device_execution() -> None:
    """
    Verifies execution on MPS device if available.
    """
    if not torch.backends.mps.is_available():
        pytest.skip("MPS device not available on this platform")

    device = torch.device("mps")

    for model_name in ["resnet18_cifar", "mobilenet_v3_small_cifar"]:
        model = get_model(model_name, num_classes=10).to(device)

        for mode in ["eval", "train"]:
            if mode == "eval":
                model.eval()
            else:
                model.train()

            for bs in [1, 64]:
                inputs = torch.randn(bs, 3, 32, 32, device=device)
                outputs = model(inputs)
                assert outputs.device.type == "mps"
                assert outputs.shape == (bs, 10)
                assert torch.isfinite(outputs).all()
