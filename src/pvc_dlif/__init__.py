"""Pipeline for the master thesis on partial volume correction for
deep-learning-derived input function estimation in preclinical PET.

Stages, in the order they run::

    00_build_manifest   inventory the dataset
    01_calibrate_grid   recover the native -> DLIF-grid preprocessing
    02_run_pvc          RL / RVC deconvolution over every scan and frame
    03_make_dlif_inputs resample corrected series onto the DLIF input grid
    04_infer_baseline   predictions from the pretrained model
    05_retrain          retrain / fine-tune on corrected data
    06_evaluate         metrics, bias-variance, paired statistics, kinetics
    07_motion_subset    motion detection and correction on the affected subset
    08_report           figures and LaTeX tables
"""

from .config import Config, Condition, load_config

__version__ = "0.1.0"
__all__ = ["Config", "Condition", "load_config", "__version__"]
