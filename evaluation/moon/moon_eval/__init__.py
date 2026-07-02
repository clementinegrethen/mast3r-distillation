"""
moon_eval — Unified evaluation package for MASt3R Moon reconstruction.

Main entry points:
  MoonEvaluator : full evaluation pipeline (evaluator.py)
  reporter.*    : summary printing and results persistence

Quick start:
    from moon_eval import MoonEvaluator
    from moon_eval.reporter import print_all, save_results

    evaluator = MoonEvaluator(
        model=my_model,
        model_name="Teacher",
        device="cuda",
        K_GT=K_GT,
        output_root="eval_output",
        mode="global_aligner",
    )
    results = evaluator.evaluate_folder("Datas/TESTS/test_image_clean_landing",
                                         "landing", max_pairs=5)
    import pandas as pd
    df = pd.DataFrame(results)
    print_all(df)
    save_results(df, "eval_output")
"""

from .evaluator import MoonEvaluator
from .gt_loader import read_exr, load_gt_view, get_gt_relative_pose
from .alignment import (
    umeyama_sim3,
    align_sim3_ransac,
    apply_sim3,
    build_T4x4_from_srt,
    improved_gt_alignment,
)
from .reconstruction import run_global_aligner, run_sparse_ga, get_reconstruction
from . import metrics
from . import reporter
from . import visualizer
from .reporter import save_pair_report_txt

__version__ = "1.0.0"

__all__ = [
    "MoonEvaluator",
    # gt_loader
    "read_exr",
    "load_gt_view",
    "get_gt_relative_pose",
    # alignment
    "umeyama_sim3",
    "align_sim3_ransac",
    "apply_sim3",
    "build_T4x4_from_srt",
    "improved_gt_alignment",
    # reconstruction
    "run_global_aligner",
    "run_sparse_ga",
    "get_reconstruction",
    # reporter helpers
    "save_pair_report_txt",
    # sub-packages
    "metrics",
    "reporter",
    "visualizer",
]
