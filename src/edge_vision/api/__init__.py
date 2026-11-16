"""
REST API Subsystem for Edge Vision Reliability Lab.

Provides local REST endpoints for health checks, model registry inspection,
model lifecycle management, and single/batch CIFAR-10 image inference.
"""

from edge_vision.api.app import create_app

__all__ = ["create_app"]
