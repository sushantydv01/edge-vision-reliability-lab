import sys
import torch
import edge_vision

def test_python_version():
    assert sys.version_info[:2] == (3, 12)

def test_torch_cpu_tensor():
    x = torch.ones(2, 2)
    assert (x + x).sum().item() == 8.0

def test_package_import():
    assert edge_vision is not None

