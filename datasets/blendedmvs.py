# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Dataloader for preprocessed BlendedMVS
# dataset at https://github.com/YoYo000/BlendedMVS
# See datasets_preprocess/preprocess_blendedmvs.py
# --------------------------------------------------------
import os.path as osp
import numpy as np

from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from dust3r.utils.image import imread_cv2


class BlendedMVS (BaseStereoViewDataset):
    """ Dataset of outdoor street scenes, 5 images each time
    """

    def __init__(self, *args, ROOT, split=None, max_pairs=0, **kwargs):
        self.ROOT = ROOT
        super().__init__(*args, **kwargs)
        self._load_data(split, max_pairs=max_pairs)

    def _load_data(self, split, max_pairs=0):
        pairs = np.load(osp.join(self.ROOT, 'blendedmvs_pairs.npy'))

        # filter out pairs whose scene directories don't exist on disk
        existing = np.array([osp.isdir(osp.join(self.ROOT, f"{h:08x}{l:016x}"))
                             for h, l in zip(pairs['seq_high'], pairs['seq_low'])])
        pairs = pairs[existing]

        if split is None:
            selection = slice(None)
        if split == 'train':
            # select 80% of all scenes
            selection = (pairs['seq_low'] % 10) > 1
        if split == 'val':
            # select 10% of all scenes
            selection = (pairs['seq_low'] % 10) == 0
        if split == 'test':
            # select 10% of all scenes (disjoint from train and val)
            selection = (pairs['seq_low'] % 10) == 1
        self.pairs = pairs[selection]

        if max_pairs > 0 and len(self.pairs) > max_pairs:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(self.pairs), max_pairs, replace=False)
            self.pairs = self.pairs[idx]

        # list of all scenes
        self.scenes = np.unique(self.pairs['seq_low'])  # low is unique enough

    def __len__(self):
        return len(self.pairs)

    def get_stats(self):
        return f'{len(self)} pairs from {len(self.scenes)} scenes'

    def _get_views(self, pair_idx, resolution, rng):
        seqh, seql, img1, img2, score = self.pairs[pair_idx]

        seq = f"{seqh:08x}{seql:016x}"
        seq_path = osp.join(self.ROOT, seq)

        views = []

        for view_index in [img1, img2]:
            impath = f"{view_index:08n}"
            image = imread_cv2(osp.join(seq_path, impath + ".jpg"))
            depthmap = imread_cv2(osp.join(seq_path, impath + ".exr"))
            camera_params = np.load(osp.join(seq_path, impath + ".npz"))

            intrinsics = np.float32(camera_params['intrinsics'])
            camera_pose = np.eye(4, dtype=np.float32)
            camera_pose[:3, :3] = camera_params['R_cam2world']
            camera_pose[:3, 3] = camera_params['t_cam2world']

            image, depthmap, intrinsics = self._crop_resize_if_necessary(
                image, depthmap, intrinsics, resolution, rng, info=(seq_path, impath))

            views.append(dict(
                img=image,
                depthmap=depthmap,
                camera_pose=camera_pose,  # cam2world
                camera_intrinsics=intrinsics,
                dataset='BlendedMVS',
                label=osp.relpath(seq_path, self.ROOT),
                instance=impath))

        return views


if __name__ == '__main__':
    from dust3r.datasets.base.base_stereo_view_dataset import view_name
    from dust3r.viz import SceneViz, auto_cam_size
    from dust3r.utils.image import rgb

    dataset = BlendedMVS(split='train', ROOT="data/blendedmvs_processed", resolution=224, aug_crop=16)

    for idx in np.random.permutation(len(dataset)):
        views = dataset[idx]
        assert len(views) == 2
        print(idx, view_name(views[0]), view_name(views[1]))
        viz = SceneViz()
        poses = [views[view_idx]['camera_pose'] for view_idx in [0, 1]]
        cam_size = max(auto_cam_size(poses), 0.001)
        for view_idx in [0, 1]:
            pts3d = views[view_idx]['pts3d']
            valid_mask = views[view_idx]['valid_mask']
            colors = rgb(views[view_idx]['img'])
            viz.add_pointcloud(pts3d, colors, valid_mask)
            viz.add_camera(pose_c2w=views[view_idx]['camera_pose'],
                           focal=views[view_idx]['camera_intrinsics'][0, 0],
                           color=(idx * 255, (1 - idx) * 255, 0),
                           image=colors,
                           cam_size=cam_size)
        viz.show()
