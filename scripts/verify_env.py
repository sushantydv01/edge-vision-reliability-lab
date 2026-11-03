#!/usr/bin/env python3
"""
Environment and hardware verification script for Edge Vision Reliability Lab.
Verifies Python version, critical ML dependencies, CPU execution, MPS availability,
and system resources (RAM, disk).
"""

import sys
import shutil
import platform
import psutil

def main():
    print("=" * 60)
    print("Edge Vision Reliability Lab - Environment Verification")
    print("=" * 60)

    # 1. Python executable and version
    print(f"[1] Python Executable: {sys.executable}")
    print(f"    Python Version:    {platform.python_version()} ({platform.python_implementation()})")
    assert sys.version_info[:2] == (3, 12), f"Expected Python 3.12, found {sys.version_info[:2]}"
    print("    -> Python version constraint (3.12.x) PASSED")

    # 2. Dependency imports & versions
    print("\n[2] Checking Dependency Imports & Versions:")
    required_packages = [
        "torch",
        "torchvision",
        "numpy",
        "scipy",
        "pandas",
        "PIL",
        "yaml",
        "tqdm",
        "pytest",
        "psutil",
        "matplotlib",
        "plotly",
        "edge_vision",
    ]

    imported_modules = {}
    for pkg in required_packages:
        try:
            mod = __import__(pkg)
            version = getattr(mod, "__version__", "installed (no __version__)")
            imported_modules[pkg] = version
            print(f"    - {pkg:<15}: {version}")
        except ImportError as e:
            print(f"    - {pkg:<15}: FAILED to import ({e})")
            sys.exit(1)

    import torch
    import torchvision
    import numpy as np
    import scipy

    print("    -> All required packages imported successfully.")

    # 3. CPU Tensor Operations
    print("\n[3] Testing CPU Tensor Operations:")
    try:
        x_cpu = torch.randn(100, 100, dtype=torch.float32, device="cpu")
        w_cpu = torch.randn(100, 100, dtype=torch.float32, device="cpu")
        y_cpu = torch.matmul(x_cpu, w_cpu) + 1.0
        assert y_cpu.shape == (100, 100), "Shape mismatch on CPU matmul"
        print(f"    - Matmul (100x100) on CPU: Success (Mean: {y_cpu.mean().item():.4f}, Std: {y_cpu.std().item():.4f})")
        print("    -> CPU tensor execution PASSED")
    except Exception as e:
        print(f"    -> CPU tensor execution FAILED: {e}")
        sys.exit(1)

    # 4. Apple Silicon MPS Availability & Operations
    print("\n[4] Testing Apple Silicon MPS Availability & Operations:")
    mps_available = torch.backends.mps.is_available()
    mps_built = torch.backends.mps.is_built()
    print(f"    - torch.backends.mps.is_built():     {mps_built}")
    print(f"    - torch.backends.mps.is_available(): {mps_available}")

    if mps_available:
        try:
            device = torch.device("mps")
            x_mps = torch.randn(100, 100, dtype=torch.float32, device=device)
            w_mps = torch.randn(100, 100, dtype=torch.float32, device=device)
            y_mps = torch.matmul(x_mps, w_mps) + 1.0
            # Sync / pull to CPU
            y_host = y_mps.cpu()
            assert y_host.shape == (100, 100), "Shape mismatch on MPS matmul"
            print(f"    - Matmul (100x100) on MPS: Success (Mean: {y_host.mean().item():.4f})")
            print("    -> Apple MPS tensor execution PASSED")
            print("    (Note: MPS availability does not guarantee all quantization/profiling kernels are supported)")
        except Exception as e:
            print(f"    -> Apple MPS tensor execution FAILED: {e}")
    else:
        print("    -> MPS is not available; falling back to CPU.")

    # 5. System Resources (RAM & Disk)
    print("\n[5] System Resource Inspection:")
    vm = psutil.virtual_memory()
    total_ram_gb = vm.total / (1024 ** 3)
    avail_ram_gb = vm.available / (1024 ** 3)
    print(f"    - Total Physical RAM: {total_ram_gb:.2f} GB")
    print(f"    - Available RAM:      {avail_ram_gb:.2f} GB ({vm.percent}% utilized)")

    disk = shutil.disk_usage(".")
    free_disk_gb = disk.free / (1024 ** 3)
    total_disk_gb = disk.total / (1024 ** 3)
    print(f"    - Free Disk Space:    {free_disk_gb:.2f} GB (Total: {total_disk_gb:.2f} GB)")

    print("\n" + "=" * 60)
    print("Environment Verification Complete: ALL CHECKS PASSED")
    print("=" * 60)

if __name__ == "__main__":
    main()

