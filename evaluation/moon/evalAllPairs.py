
"""
evaluate_pair.py - Version finale qui reproduit exactement le notebook
"""
from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs
import mast3r.utils.path_to_dust3r
from dust3r.inference import inference
from dust3r.utils.image import load_images
from dust3r.utils.image import load_images

#imports for visualizing matches
import numpy as np
import torch
import torchvision.transforms.functional
from matplotlib import pyplot as pl
from mpl_toolkits.axes_grid1 import make_axes_locatable

import cv2 #for pnp

#supressing unnecessary warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


import time
import matplotlib.pyplot as plt
import csv
import os
from PIL import Image
from scipy.spatial.transform import Rotation as R
import base64
import pandas as pd
import io
from io import BytesIO

import numpy as np
import cv2
from pathlib import Path
# from dust3r.utils.device import to_numpy
# import matplotlib.pyplot as plt
# from scipy.stats import pearsonr, linregress
# from skimage.metrics import structural_similarity as ssim
# from scipy.ndimage import gaussian_filter, sobel
# from matplotlib.colors import LightSource
# import seaborn as sns
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# import OpenEXR, Imath
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Import utility classes
from MAST3RUtils import MAST3RUtils
from mast3r.model import AsymmetricMASt3R
#!/usr/bin/env python3
"""
evaluate_pair_ULTRA_COMPLETE.py

Script ULTRA COMPLET qui inclut TOUTES les métriques :
- Alignement Sim(3) 
- Métriques géométriques de base
- SSIM avec différentes variantes
- Profils de profondeur (simple + multiples)
- Régression linéaire
- Histogrammes d'erreurs
- ANALYSE DU RELIEF COMPLÈTE (pentes, courbures, rugosité)
- Classification du terrain
- Visualisations avancées avec hillshading
- Toutes les visualisations avec SAUVEGARDE
- Rapport final complet
"""
import numpy as np
import cv2
# import open3d as o3d
# from pathlib import Path
# from dust3r.utils.device import to_numpy
# import matplotlib.pyplot as plt
# from scipy.stats import pearsonr, linregress
# from skimage.metrics import structural_similarity as ssim
# from scipy.ndimage import gaussian_filter, sobel, uniform_filter
# from matplotlib.colors import LightSource
# import seaborn as sns
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# import OpenEXR, Imath
import warnings
import os
from datetime import datetime

warnings.simplefilter(action='ignore', category=FutureWarning)

# Import utility classes
from MAST3RUtils import MAST3RUtils
from mast3r.model import AsymmetricMASt3R
#!/usr/bin/env python3
"""
evaluate_pair_MAIN.py

Script principal pour l'évaluation complète de MASt3R.
Utilise la classe MASt3REvaluator pour une organisation propre du code.
"""

from pathlib import Path
from MASt3REvaluator import MASt3REvaluator
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#!/usr/bin/env python3
"""
evaluate_pair_MAIN.py

Script principal pour l'évaluation complète de MASt3R.
Utilise la classe MASt3REvaluator pour une organisation propre du code.
"""

from pathlib import Path
from MASt3REvaluator import MASt3REvaluator
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from pathlib import Path
from MASt3REvaluator import MASt3REvaluator
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
from pathlib import Path
from MASt3REvaluator import MASt3REvaluator
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
from pathlib import Path
from MASt3REvaluator import MASt3REvaluator
import warnings
import pandas as pd

warnings.simplefilter(action='ignore', category=FutureWarning)

def batch_evaluate(gt_folder, device, ckpt_path, output_root=None):
    gt_folder = Path(gt_folder)
    images   = sorted(gt_folder.glob("*.jpg"))
    pairs    = list(zip(images[0::2], images[1::2]))

    # Liste pour accumuler les résultats
    results = []

    for img1, img2 in pairs:
        name    = img1.stem
        out_dir = Path(output_root or "eval_batch") / f"{name}_vs_{img2.stem}"
        evaluator = MASt3REvaluator(
            gt_folder=gt_folder,
            name=name,
            img1=img1,
            img2=img2,
            device=device,
            ckpt_path=ckpt_path,
            output_dir=out_dir
        )
        print(f"\n▶ Évaluation de {img1.name} vs {img2.name}")
        try:
            metrics, terrain_metrics, profile_metrics, error_stats = evaluator.run_complete_evaluation()
            
            # On suppose que `metrics` est un dict plat, par ex. {'rmse': 0.5, 'mae': 0.3, ...}
            row = {'pair': f"{img1.name}_vs_{img2.name}"}
            row.update(metrics)
            results.append(row)

        except Exception as e:
            print(f"‼ Échec sur {img1.name} ↔ {img2.name}: {e}")

    # Création du DataFrame
    df = pd.DataFrame(results).set_index('pair')
    # Statistiques descriptives
    stats = df.describe().T  # transpose pour plus de lisibilité

    print("\n=== Tableau final des statistiques ===")
    print(stats)

    # Optionnel : sauvegarder en CSV
    stats.to_csv(Path(output_root or "eval_batch") / "summary_statistics.csv")

    return df, stats

if __name__ == "__main__":
    gt_folder   = "Datas/TESTS/test_image_clean_landing"
    device      = "cuda"
    ckpt_path = "output/Mast3r_dstillation_exp11_with_criterion_freeze_head/checkpoint-best.pth"
    df, stats = batch_evaluate(gt_folder, device, ckpt_path, output_root="eval_nadir_freeze")
