from importlib.metadata import version

# from qec_loss._fast_qec_loss import _FastQecLoss

__version__ = version("qec-loss")

__all__ = ["__version__", "_FastQecLoss"]
