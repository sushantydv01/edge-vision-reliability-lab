"""
Exception hierarchy for the inference and deployment subsystem.
"""


class InferenceError(Exception):
    """Base exception for all errors arising within the inference subsystem."""
    pass


class ModelNotFoundError(InferenceError):
    """Raised when a specified model checkpoint or artifact file is not found on disk."""
    pass


class UnsupportedModelError(InferenceError):
    """Raised when an unrecognised or unsupported model architecture name is requested."""
    pass


class UnsupportedPrecisionError(InferenceError):
    """Raised when an unrecognised or unsupported precision string is requested."""
    pass


class DeviceCompatibilityError(InferenceError):
    """Raised when an incompatible or unsupported compute device is selected for a model/precision."""
    pass


class InvalidInputError(InferenceError):
    """Raised when an inference input (tensor, image, or parameter) violates formatting contracts."""
    pass
