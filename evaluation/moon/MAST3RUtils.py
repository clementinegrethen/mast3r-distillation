import numpy as np
import cv2
import warnings
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
import torch
import torchvision.transforms.functional
from matplotlib import pyplot as pl
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Imports specific to MAST3R and OpenEXR
try:
    from mast3r.model import AsymmetricMASt3R
    from mast3r.fast_nn import fast_reciprocal_NNs
    import mast3r.utils.path_to_dust3r
    from dust3r.inference import inference
    from dust3r.utils.image import load_images
    import OpenEXR # Make sure OpenEXR is installed (e.g., pip install OpenEXR)
    import Imath  # Part of OpenEXR, for pixel types
except ImportError as e:
    warnings.warn(f"MAST3R or OpenEXR related imports failed: {e}. Some functionalities might be unavailable.")
    # Define dummy classes/functions if imports fail to prevent errors
    AsymmetricMASt3R = None
    fast_reciprocal_NNs = None
    inference = None
    load_images = None
    OpenEXR = None
    Imath = None

try:
    import open3d as o3d
except ImportError as e:
    warnings.warn(f"Open3D import failed: {e}. 3D visualization will be unavailable.")
    o3d = None

try:
    from pyproj import Proj # For UTM conversions
except ImportError as e:
    warnings.warn(f"pyproj import failed: {e}. UTM conversion functions will be unavailable.")
    Proj = None


# Suppressing unnecessary warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class MAST3RUtils:
    """
    A utility class to encapsulate various functions related to MAST3R processing,
    pose estimation, and visualization.
    """

    @staticmethod
    def computePoseError(est_pose, gt_pose):
        """
        Computes positional and rotational error between two 4x4 pose matrices.
        
        Args:
            est_pose (np.ndarray): Estimated 4x4 pose matrix.
            gt_pose (np.ndarray): Ground truth 4x4 pose matrix.
            
        Returns:
            tuple: (pos_error, rot_error) where pos_error is in units and rot_error in degrees.
        """
        # Compute positional error
        pos_error = np.linalg.norm(est_pose[:3, 3] - gt_pose[:3, 3])
        
        # Extract rotation quaternions
        est_quat = R.from_matrix(est_pose[:3, :3]).as_quat()
        gt_quat = R.from_matrix(gt_pose[:3, :3]).as_quat()
        
        # Convert to (w, x, y, z) for quaternion dot product
        est_quat = np.concatenate(([est_quat[3]], est_quat[:3]))
        gt_quat = np.concatenate(([gt_quat[3]], gt_quat[:3]))
        
        # Compute the quaternion dot product and account for double covering.
        dot = np.clip(np.abs(np.dot(est_quat, gt_quat)), -1.0, 1.0)
        theta = 2 * np.arccos(dot)
        rot_error = np.degrees(theta)
        
        print(f"Positional Error: {pos_error:.4f} units, Rotation Error: {rot_error:.4f} degrees")
        return pos_error, rot_error

    @staticmethod
    def softmax(x):
        """Applies the softmax function to an array."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    @staticmethod
    def min_max_normalize(x):
        """Performs min-max normalization on an array."""
        return (x - np.min(x)) / (np.max(x) - np.min(x))

    @staticmethod
    def makeHistogram(match_conf_im0, match_conf_im1, lowest_confidence_im0=None, lowest_confidence_im1=None):
        """
        Generates histograms for match confidence scores.
        
        Args:
            match_conf_im0 (np.ndarray): Confidence scores for image 0 matches.
            match_conf_im1 (np.ndarray): Confidence scores for image 1 matches.
            lowest_confidence_im0 (float, optional): Optional cutoff for image 0.
            lowest_confidence_im1 (float, optional): Optional cutoff for image 1.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        ax1.hist(match_conf_im0, bins=50, edgecolor='black', color='skyblue')
        if lowest_confidence_im0 is not None:
            ax1.axvline(lowest_confidence_im0, color='red', linestyle='--', label='Confidence Cutoff')
        ax1.set_title('Histogram of Confidence Scores (Anchor Matches)')
        ax1.set_xlabel('Confidence Value')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        ax2.hist(match_conf_im1, bins=50, edgecolor='black', color='lightgreen')
        if lowest_confidence_im1 is not None:
            ax2.axvline(lowest_confidence_im1, color='red', linestyle='--', label='Confidence Cutoff')
        ax2.set_title('Histogram of Confidence Scores (Query Matches)')
        ax2.set_xlabel('Confidence Value')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()
        plt.show()    

    @staticmethod
    def visualize2Dmatches(conf_im0, conf_im1, matches_im0, matches_im1, view1, view2, n_viz=100):
        """
        Visualizes 2D matches between two images, color-coded by confidence.
        
        Args:
            conf_im0 (np.ndarray): Confidence map for image 0.
            conf_im1 (np.ndarray): Confidence map for image 1.
            matches_im0 (np.ndarray): (N, 2) array of match coordinates in image 0.
            matches_im1 (np.ndarray): (N, 2) array of match coordinates in image 1.
            view1 (dict): Dictionary containing view data for image 0 from MAST3R output.
            view2 (dict): Dictionary containing view data for image 1 from MAST3R output.
            n_viz (int): Number of matches to visualize with connecting lines.
        """
        num_matches = matches_im0.shape[0]
        print(f"Number of matches before confidence mask: {num_matches}")
        
        if num_matches == 0:
            print("No matches to visualize.")
            return

        match_idx_to_viz = np.round(np.linspace(0, num_matches - 1, min(n_viz, num_matches))).astype(int)
        viz_matches_im0, viz_matches_im1 = matches_im0[match_idx_to_viz], matches_im1[match_idx_to_viz]

        image_mean = torch.as_tensor([0.5, 0.5, 0.5], device='cpu').reshape(1, 3, 1, 1)
        image_std = torch.as_tensor([0.5, 0.5, 0.5], device='cpu').reshape(1, 3, 1, 1)

        viz_imgs = []
        for i, view in enumerate([view1, view2]):
            rgb_tensor = view['img'] * image_std + image_mean
            viz_imgs.append(rgb_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy())

        H0, W0, H1, W1 = *viz_imgs[0].shape[:2], *viz_imgs[1].shape[:2]
        img0 = np.pad(viz_imgs[0], ((0, max(H1 - H0, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
        img1 = np.pad(viz_imgs[1], ((0, max(H0 - H1, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
        img = np.concatenate((img0, img1), axis=1)
        
        # Plot matches with lines
        pl.figure(figsize=(15, 8))
        pl.imshow(img)
        pl.title('Randomly Sampled 2D Matches')
        cmap = pl.get_cmap('jet')
        for i in range(len(viz_matches_im0)):
            (x0, y0), (x1, y1) = viz_matches_im0[i].T, viz_matches_im1[i].T
            pl.plot([x0, x1 + W0], [y0, y1], '-', color=cmap(i / (len(viz_matches_im0) - 1)), scalex=False, scaley=False, linewidth=1)
            pl.plot(x0, y0, '+', color=cmap(i / (len(viz_matches_im0) - 1)), markersize=5, scalex=False, scaley=False)
            pl.plot(x1 + W0, y1, '+', color=cmap(i / (len(viz_matches_im0) - 1)), markersize=5, scalex=False, scaley=False)
        pl.show(block=True)

        # Create the figure for scatter plots
        fig, ax = plt.subplots(figsize=(12, 8))
        im = ax.imshow(img)
        ax.set_title('Image Matches with Confidence (Anchor - left, Query - Right)')

        # Create scatter plots of matches with color-coded confidence
        # Ensure that `matches_im0` and `matches_im1` are within the bounds of `conf_im0`/`conf_im1`
        # Also ensure valid indices for confidence lookup
        valid_conf_indices_0 = (matches_im0[:, 1] < conf_im0.shape[0]) & (matches_im0[:, 0] < conf_im0.shape[1])
        valid_conf_indices_1 = (matches_im1[:, 1] < conf_im1.shape[0]) & (matches_im1[:, 0] < conf_im1.shape[1])

        if np.any(valid_conf_indices_0):
            scatter_im0 = ax.scatter(matches_im0[valid_conf_indices_0, 0], matches_im0[valid_conf_indices_0, 1], 
                                    c=conf_im0[matches_im0[valid_conf_indices_0, 1], matches_im0[valid_conf_indices_0, 0]], 
                                    cmap='viridis', s=10, alpha=0.7)
        else:
            print("No valid matches in image 0 for confidence visualization.")
            scatter_im0 = None

        if np.any(valid_conf_indices_1):
            scatter_im1 = ax.scatter(matches_im1[valid_conf_indices_1, 0] + W0, matches_im1[valid_conf_indices_1, 1], 
                                    c=conf_im1[matches_im1[valid_conf_indices_1, 1], matches_im1[valid_conf_indices_1, 0]], 
                                    cmap='viridis', s=10, alpha=0.7)
        else:
            print("No valid matches in image 1 for confidence visualization.")
            scatter_im1 = None

        # Create a divider for the existing axes instance
        if scatter_im0 is not None or scatter_im1 is not None:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            
            vmin = min(conf_im0.min() if conf_im0.size > 0 else np.inf, conf_im1.min() if conf_im1.size > 0 else np.inf)
            vmax = max(conf_im0.max() if conf_im0.size > 0 else -np.inf, conf_im1.max() if conf_im1.size > 0 else -np.inf)

            if vmin != np.inf and vmax != -np.inf: # Check if there's actual data range
                plt.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(vmin=vmin, vmax=vmax), 
                            cmap='viridis'), cax=cax, label='Confidence')
            else:
                print("Cannot plot colorbar: no valid confidence data range.")
        else:
            print("Skipping colorbar as no valid scatter plots were created.")

        plt.tight_layout()
        plt.show()

    def getMasterOutput(model, device, anchor_image, query_image, n_matches, visualizeMatches=False, verboseFlag=True): 
        """
        Inputs known image and unknown image paths to return mast3r output
        Uses both descriptor confidence (for matching quality) and 3D confidence (for 3D point quality)
        """
        
        # Load images
        images = load_images([anchor_image, query_image], size=512)
        
        # Run MASt3R inference
        mast3r_inference_start = time.time()
        output = inference([tuple(images)], model, device, batch_size=1, verbose=False)
        
        mast3r_inference_stop = time.time()
        mast3r_inference_time = mast3r_inference_stop - mast3r_inference_start
        
        # Extract outputs
        view1 = output['view1']
        view2 = output['view2'] 
        pred1 = output['pred1']
        pred2 = output['pred2']

        desc1 = pred1['desc'].squeeze(0).detach()
        desc2 = pred2['desc'].squeeze(0).detach()
        
        if verboseFlag:
            print("Keys in pred1:", pred1.keys())
            print("Keys in pred2:", pred2.keys())
        
        # Find 2D-2D matches between the two images
        point_matches_start = time.time()
        matches_im0, matches_im1 = fast_reciprocal_NNs(desc1, desc2, subsample_or_initxy1=8,
                                                        device=device, dist='dot', block_size=2**13)
        point_matches_stop = time.time()
        point_matches_time = point_matches_stop - point_matches_start

        if verboseFlag:
            print(f"Mast3r Inference Time: {mast3r_inference_time:.4f} seconds.")
            print(f"Point Matches Time: {point_matches_time:.4f} seconds.")
        
        # Filter matches based on image boundaries
        ignore = 0
        border = 3
        H0, W0 = view1['true_shape'][0]
        valid_matches_im0 = (matches_im0[:, 0] >= border) & (matches_im0[:, 0] < int(W0) - border) & \
                            (matches_im0[:, 1] >= border) & (matches_im0[:, 1] < int(H0) - border - ignore)

        H1, W1 = view2['true_shape'][0]
        valid_matches_im1 = (matches_im1[:, 0] >= border) & (matches_im1[:, 0] < int(W1) - border) & \
                            (matches_im1[:, 1] >= border) & (matches_im1[:, 1] < int(H1) - border - ignore)

        valid_matches = valid_matches_im0 & valid_matches_im1

        # Apply boundary filtering
        matches_im0 = matches_im0[valid_matches]
        matches_im1 = matches_im1[valid_matches]

        # Convert outputs to numpy arrays
        pts3d_im0 = pred1['pts3d'].squeeze(0).detach().cpu().numpy() 
        pts3d_im1 = pred2['pts3d_in_other_view'].squeeze(0).detach().cpu().numpy() 

        conf_im0 = pred1['conf'].squeeze(0).detach().cpu().numpy()
        conf_im1 = pred2['conf'].squeeze(0).detach().cpu().numpy()

        desc_conf_im0 = pred1['desc_conf'].squeeze(0).detach().cpu().numpy()
        desc_conf_im1 = pred2['desc_conf'].squeeze(0).detach().cpu().numpy()
        
        # Extract ALL confidence scores for the matches
        # 1. Descriptor confidence (for matching quality)
        match_desc_conf_im0 = desc_conf_im0[matches_im0[:, 1], matches_im0[:, 0]]
        match_desc_conf_im1 = desc_conf_im1[matches_im1[:, 1], matches_im1[:, 0]]
        
        # 2. 3D confidence (for 3D point quality)
        match_3d_conf_im0 = conf_im0[matches_im0[:, 1], matches_im0[:, 0]]
        match_3d_conf_im1 = conf_im1[matches_im1[:, 1], matches_im1[:, 0]]


        # Combine confidences: we want points that are good in BOTH matching AND 3D reconstruction
        # Option 1: Multiplication (both need to be high)
        combined_conf_im0 = match_desc_conf_im0 * match_3d_conf_im0
        combined_conf_im1 = match_desc_conf_im1 * match_3d_conf_im1
        
        # Take minimum between both views (match must be confident in both images)
        combined_total_conf = np.minimum(combined_conf_im0, combined_conf_im1)
        
        # Alternative Option 2: Weighted average (you can adjust weights)
        # weight_desc = 0.5  # weight for descriptor confidence
        # weight_3d = 0.5    # weight for 3D confidence
        # combined_conf_im0 = weight_desc * match_desc_conf_im0 + weight_3d * match_3d_conf_im0
        # combined_conf_im1 = weight_desc * match_desc_conf_im1 + weight_3d * match_3d_conf_im1
        # combined_total_conf = np.minimum(combined_conf_im0, combined_conf_im1)

        # Sort by combined total confidence
        sorted_indices = np.argsort(combined_total_conf)[::-1]

        # Select top n_matches based on combined confidence
        n_matches = min(n_matches, len(matches_im0))
        top_indices = sorted_indices[:n_matches]
        
        # Create mask for top matches
        conf_mask = np.zeros(len(matches_im0), dtype=bool)
        conf_mask[top_indices] = True

        # Apply the mask to filter matches
        filtered_matches_im0 = matches_im0[conf_mask]
        filtered_matches_im1 = matches_im1[conf_mask]

        if verboseFlag:
            print(f"Number of matches after confidence mask: {filtered_matches_im0.shape[0]}")
            print(f"Combined confidence range: [{combined_total_conf[top_indices].min():.4f}, {combined_total_conf[top_indices].max():.4f}]")
            print(f"Desc conf range: [{match_desc_conf_im0[top_indices].min():.4f}, {match_desc_conf_im0[top_indices].max():.4f}]")
            print(f"3D conf range: [{match_3d_conf_im0[top_indices].min():.4f}, {match_3d_conf_im0[top_indices].max():.4f}]")

        if visualizeMatches and len(match_desc_conf_im0) > 0:
            # Create histogram with combined confidence
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Descriptor confidence histograms
            axes[0, 0].hist(match_desc_conf_im0, bins=50, edgecolor='black', color='skyblue', alpha=0.7, label='All matches')
            axes[0, 0].hist(match_desc_conf_im0[top_indices], bins=30, edgecolor='black', color='darkblue', alpha=0.7, label='Selected matches')
            axes[0, 0].set_title('Descriptor Confidence (Anchor)')
            axes[0, 0].legend()
            
            axes[0, 1].hist(match_desc_conf_im1, bins=50, edgecolor='black', color='lightgreen', alpha=0.7, label='All matches')
            axes[0, 1].hist(match_desc_conf_im1[top_indices], bins=30, edgecolor='black', color='darkgreen', alpha=0.7, label='Selected matches')
            axes[0, 1].set_title('Descriptor Confidence (Query)')
            axes[0, 1].legend()
            
            # 3D confidence histograms
            axes[1, 0].hist(match_3d_conf_im0, bins=50, edgecolor='black', color='salmon', alpha=0.7, label='All matches')
            axes[1, 0].hist(match_3d_conf_im0[top_indices], bins=30, edgecolor='black', color='darkred', alpha=0.7, label='Selected matches')
            axes[1, 0].set_title('3D Confidence (Anchor)')
            axes[1, 0].legend()
            
            axes[1, 1].hist(match_3d_conf_im1, bins=50, edgecolor='black', color='gold', alpha=0.7, label='All matches')
            axes[1, 1].hist(match_3d_conf_im1[top_indices], bins=30, edgecolor='black', color='darkorange', alpha=0.7, label='Selected matches')
            axes[1, 1].set_title('3D Confidence (Query)')
            axes[1, 1].legend()
            
            plt.tight_layout()
            plt.show()

        return (filtered_matches_im0, filtered_matches_im1, matches_im0, matches_im1, 
                pts3d_im0, pts3d_im1, conf_im0, conf_im1, desc_conf_im0, desc_conf_im1)

    @staticmethod
    def scale_intrinsics(K: np.ndarray, prev_w: float, prev_h: float, master_w: float, master_h: float) -> np.ndarray:
        """
        Scales the intrinsics matrix based on image dimension changes.
        
        Args:
            K (np.ndarray): 3x3 intrinsics matrix.
            prev_w (float): Original image width.
            prev_h (float): Original image height.
            master_w (float): New image width.
            master_h (float): New image height.
            
        Returns:
            np.ndarray: Scaled 3x3 intrinsics matrix.
        """
        assert K.shape == (3, 3), f"Expected K to be (3, 3), but got {K.shape=}"

        scale_w = master_w / prev_w  
        scale_h = master_h / prev_h  

        K_scaled = K.copy()
        K_scaled[0, 0] *= scale_w # fx
        K_scaled[0, 2] *= scale_w # cx
        K_scaled[1, 1] *= scale_h # fy
        K_scaled[1, 2] *= scale_h # cy

        return K_scaled

    @staticmethod
    def CameraMatrix(fx, fy, cx, cy):
        """Constructs a 3x3 camera intrinsic matrix."""
        return np.array([[fx,  0, cx],
                         [ 0, fy, cy],
                         [ 0,  0, 1]])

    @staticmethod
    def cameraMatrixMapillary(focal, width, height): #converting open sfm intrinsics to standard
        """
        Converts Mapillary-style intrinsics (focal length as ratio of width) to standard K matrix.
        
        Args:
            focal (float): Focal length as a ratio of width.
            width (int): Image width.
            height (int): Image height.
            
        Returns:
            np.ndarray: 3x3 intrinsic matrix.
        """
        K = np.array([ [focal * width, 0, width / 2],
                       [0, focal * height, height / 2], # Assuming fy = fx * (height/width) for isotropic pixels
                       [0, 0, 1] ])
        return K

    @staticmethod
    def run_pnp(pts2D, pts3D, K, distortion=None): 
        # print("pts3D shape:", pts3D.shape)
        if pts3D.shape[0]<=3:
            print("Insufficient Points.")
            return False, None
        # print("pts2D shape:", pts2D.shape)
        try:
            success, r_pose, t_pose, _ = cv2.solvePnPRansac(pts3D, pts2D, K, distortion, flags=cv2.SOLVEPNP_SQPNP,
                                                            iterationsCount=10_000,
                                                            reprojectionError=2,
                                                            confidence=0.9999) #returns 3d to 2d transfromation #anchor to query
            if not success:
                print("Failed to find transform")
                return False, None
            r_pose = cv2.Rodrigues(r_pose)[0]  # world2cam == world2cam2
            RT = np.r_[np.c_[r_pose, t_pose], [(0,0,0,1)]] # world2cam2 #anchor to query

            return True, np.linalg.inv(RT)   #query to anchor
        except Exception as e:
            print(f"Error details: {str(e)}")
            return False, None

    @staticmethod
    def get_rotation_from_compass(compass_angle_deg):
        """ 
        Create a rotation matrix based on compass angle (in degrees).
        Assumes Z-up, Y-north, X-east for compass.
        """
        compass_angle_rad = np.deg2rad(compass_angle_deg)
        return np.array([
            [np.cos(compass_angle_rad), -np.sin(compass_angle_rad), 0],
            [np.sin(compass_angle_rad), np.cos(compass_angle_rad), 0],
            [0, 0, 1]
        ])

    @staticmethod
    def pnp_to_relative_global_coords(pnp_rotation, pnp_translation, ref_lat, ref_lon, compass_angle, ref_alt=0):
        """
        Transforms a PnP result (relative pose from Camera 0 to Camera 1)
        into relative global (Lat, Lon, Alt) coordinates, using a reference point
        and compass orientation for Camera 0.

        Args:
            pnp_rotation (np.ndarray): 3x3 rotation matrix (R_C0_C1).
            pnp_translation (np.ndarray): 3x1 translation vector (t_C0_C1).
            ref_lat (float): Latitude of the reference camera (Camera 0).
            ref_lon (float): Longitude of the reference camera (Camera 0).
            compass_angle (float): Compass angle (yaw) of the reference camera (Camera 0) in degrees.
            ref_alt (float): Altitude of the reference camera (Camera 0).

        Returns:
            tuple: (global_lat, global_lon, global_alt) of the query camera (Camera 1).
        """
        if Proj is None:
            print("pyproj is not installed. Cannot perform UTM conversions.")
            return None, None, None

        # Define the reference point in UTM coordinates
        utm_zone = MAST3RUtils.getUTMzone(ref_lon)
        utm_proj = Proj(proj='utm', zone=utm_zone, ellps='WGS84') 
        ref_x, ref_y = utm_proj(ref_lon, ref_lat)

        # Ensure pnp_rotation is a 3x3 matrix
        if pnp_rotation.shape == (3,):
            R_C0_C1, _ = cv2.Rodrigues(pnp_rotation) # Convert Rodrigues vector to matrix
        else:
            R_C0_C1 = pnp_rotation

        # The PnP result (pnp_rotation, pnp_translation) is T_C0_C1.
        # This transforms points from C1 frame to C0 frame.
        # We need the position of C1 in C0's frame, which is t_C0_C1.

        # The global orientation of Camera 0 in the world frame
        # Assuming World frame: X-East, Y-North, Z-Up
        # Compass angle is typically yaw from North.
        # This creates R_World_C0
        R_World_C0_yaw = MAST3RUtils.get_rotation_from_compass(compass_angle)

        # PnP translation is in Camera 0's frame. We need to rotate it to the world frame.
        # t_World_C1 = t_World_C0 + R_World_C0 @ t_C0_C1
        # Here, t_World_C0 is (ref_x, ref_y, ref_alt)
        
        # However, the example code implies a specific setup for axis alignment.
        # Let's align to the original example's logic:
        # R_cam_to_world in the original code seems to imply a rotation from camera's
        # Z-forward, X-right, Y-down convention to a typical world X-right, Y-forward, Z-up.
        # If your camera's local frame is X-right, Y-down, Z-forward, and world is X-East, Y-North, Z-Up:
        # Camera local to World: R_cam_to_world @ (X_cam, Y_cam, Z_cam)
        # This rotation matrix is applied to the PnP translation.
        
        # Let's reinterpret `query_camera_in_anchor_frame` from your original snippet:
        # `query_camera_in_anchor_frame = R_cam_to_world @ pnp_translation`
        # This suggests `R_cam_to_world` rotates the translation vector *from Camera 0's frame*
        # into some other 'world-aligned' frame, *before* applying the compass rotation.
        # This is a non-standard way to combine transformations.

        # Let's stick to the standard:
        # T_World_C0 = [R_World_C0_yaw | t_World_C0]
        # T_World_C1 = T_World_C0 @ T_C0_C1
        
        # Position of Camera 1 in the world frame
        t_World_C0 = np.array([ref_x, ref_y, ref_alt])
        
        # This is the most direct and standard way:
        # 1. Transform the C0-frame translation vector into world frame using C0's world orientation.
        # 2. Add C0's world position.
        t_World_C1_vector = R_World_C0_yaw @ pnp_translation.flatten() + t_World_C0
        
        new_x = t_World_C1_vector[0]
        new_y = t_World_C1_vector[1]
        global_alt = t_World_C1_vector[2]

        # Transform back to latitude and longitude
        global_lon, global_lat = utm_proj(new_x, new_y, inverse=True)

        return global_lat, global_lon, global_alt

    @staticmethod
    def getUTMzone(longitude):
        """Calculates the UTM zone from longitude."""
        return int((longitude + 180) / 6) + 1

    @staticmethod
    def getImageFromIndex(index, image_folder):
        """Retrieves image metadata and path from a metadata.csv based on ID."""
        filename = os.path.join(image_folder, 'metadata.csv')
        try:
            with open(filename, 'r', newline='') as csvfile:
                csvreader = csv.DictReader(csvfile)
                for row in csvreader:
                    if row['id'] == str(index):
                        image_path = os.path.join(image_folder, row['image_name'])
                        return row, image_path
            return None, None # ID not found
        except FileNotFoundError:
            print(f"Metadata file not found: {filename}")
            return None, None
        except Exception as e:
            print(f"Error reading metadata CSV: {e}")
            return None, None

    @staticmethod
    def getSequenceImageFromIndex(image_id, image_folder): # for mapillary data
        """Retrieves image metadata and path from a metadata.csv (Mapillary style) based on ID."""
        filename = os.path.join(image_folder, 'metadata.csv') # Assuming metadata.csv is directly in image_folder
        try:
            with open(filename, 'r', newline='') as csvfile:
                csvreader = csv.DictReader(csvfile)
                for row in csvreader:
                    if row['id'] == str(image_id):
                        image_path = os.path.join(image_folder, f"{row['id']}.jpg") # Assuming image_name is just ID.jpg
                        return row, image_path
            return None, None # ID not found
        except FileNotFoundError:
            print(f"Metadata file not found: {filename}")
            return None, None
        except Exception as e:
            print(f"Error reading metadata CSV (Mapillary): {e}")
            return None, None
            
    @staticmethod
    def plotImages(image_indices, image_folder, rotate=True, title=None):
        """
        Plots a series of images given their indices and folder.
        
        Args:
            image_indices (list): List of image IDs (e.g., [100, 101]).
            image_folder (str): Path to the folder containing images and metadata.csv.
            rotate (bool): Whether to rotate images by -90 degrees (common for portrait phone images).
            title (str, optional): Overall title for the plot.
        """
        image_paths = [MAST3RUtils.getImageFromIndex(index, image_folder)[1] for index in image_indices]
        image_paths = [p for p in image_paths if p is not None] # Filter out None paths if getImageFromIndex fails

        if not image_paths:
            print("No valid image paths to plot.")
            return

        images = [Image.open(path) for path in image_paths]
        
        fig, axes = plt.subplots(1, len(images), figsize=(20, 5))
        if len(images) == 1:
            axes = [axes] # Ensure axes is iterable even for single image
        
        for ax, img, index in zip(axes, images, image_indices):
            if rotate:
                img = img.rotate(-90, expand=True)
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(f'Image {index}')
        
        if title:
            fig.suptitle(title, fontsize=16)
        
        plt.tight_layout()
        plt.show()

    @staticmethod
    def shiftOrigin(points, x_offset, y_offset):
        """Shifts 2D points by a given offset."""
        return np.array([[p[0] + x_offset, p[1] + y_offset] for p in points])

    @staticmethod
    def get_image_html(img_path, width=50, rotate=True):
        """Generates HTML string for embedding an image, with optional rotation and resizing."""
        try:
            with Image.open(img_path) as img:
                if rotate:
                    img = img.rotate(-90, expand=True)
                img.thumbnail((width, width))
                buffered = BytesIO()
                img.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            return f'<img src="data:image/jpeg;base64,{img_str}" width="{width}" height="{width}">'
        except FileNotFoundError:
            return f'<div>Image not found: {os.path.basename(img_path)}</div>'
        except Exception as e:
            return f'<div>Error loading image {os.path.basename(img_path)}: {e}</div>'


    @staticmethod
    def plot_images_on_map(csv_path, image_folder, pin_locations=None, visualizePins=True, visualizeImages=True, output_map='map.html'):
        """
        Plots image locations and optional pins on an interactive Folium map.
        
        Args:
            csv_path (str): Path to the CSV file containing image metadata (lat, long, id, image_name).
            image_folder (str): Path to the folder containing image files.
            pin_locations (list, optional): List of [lat, lon] tuples for custom pins.
            visualizePins (bool): Whether to display the custom pins.
            visualizeImages (bool): Whether to display image markers with popups.
            output_map (str): Filename for the output HTML map.
        """
        try:
            import folium
            from folium.plugins import MarkerCluster
            from folium.features import DivIcon
        except ImportError:
            print("Folium not installed. Please install it with 'pip install folium' to use map plotting.")
            return

        try:
            data = pd.read_csv(csv_path)
        except FileNotFoundError:
            print(f"CSV file not found: {csv_path}")
            return
        except Exception as e:
            print(f"Error loading CSV: {e}")
            return

        avg_lat = data['lat'].mean() if not data.empty else 0
        avg_lon = data['long'].mean() if not data.empty else 0
        folium_map = folium.Map(location=[avg_lat, avg_lon], zoom_start=12)

        if visualizePins and pin_locations:
            for index, location in enumerate(pin_locations):
                folium.Marker(
                    location=location,
                    popup=f'Index: {index}<br>Location: {location}',
                    tooltip=f'Pin {index}',
                    icon=folium.Icon(color='red', icon='info-sign')
                ).add_to(folium_map)

        if visualizeImages and not data.empty:        
            for _, row in data.iterrows():
                image_path = os.path.join(image_folder, f"{row['image_name']}")
                lat, lon = row['lat'], row['long']
                image_id = row['id']
                
                image_html = MAST3RUtils.get_image_html(image_path)

                folium.Marker(
                    location = [lat,lon],
                    popup=folium.Popup(image_html, max_width=300),
                    icon=folium.Icon(color='blue'),
                    tooltip=image_id
                ).add_to(folium_map)

        legend_html = """
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; 
                    width: 180px; height: auto; 
                    z-index:9999; font-size:14px; 
                    background-color:white; 
                    border:2px solid grey; 
                    padding: 10px;">
            <b>Legend</b><br>
            <i class="fa fa-map-marker" style="color:blue"></i>&nbsp; Images<br>
            <i class="fa fa-map-marker" style="color:red"></i>&nbsp; Manually Sorted Locations<br>
        </div>
        """
        folium_map.get_root().html.add_child(folium.Element(legend_html))

        folium_map.save(output_map)
        print(f"Map saved as {output_map}")
        # display(folium_map) # This typically works in Jupyter, might not in a script

    @staticmethod
    def create_orientation_arrow(location, angle, color='blue'):
        """
        Creates a custom Folium DivIcon marker with an arrow indicating orientation.
        
        Args:
            location (list): [lat, lon] for the marker.
            angle (float): Rotation angle in degrees (e.g., compass heading).
            color (str): CSS color for the arrow.
            
        Returns:
            folium.Marker: A Folium marker object.
        """
        try:
            import folium
            from folium.features import DivIcon
        except ImportError:
            print("Folium not installed. Cannot create orientation arrows.")
            return None

        arrow_symbol = "↑" # Unicode arrow pointing north by default
        
        html = f"""
        <div style="
            transform: rotate({angle}deg);
            font-size: 30px;
            color: {color};
            width: 30px;
            height: 30px;
            text-align: center;
            line-height: 1.5;
        ">{arrow_symbol}</div>
        """
        return folium.Marker(
        location=location,
        icon=folium.DivIcon(html=html))

    @staticmethod
    def visualize_camera_and_points(K, master_size, T_C0_C1_gt, T_C0_C1_est, pts3d_anchor_frame):
        """
        Visualizes two camera frustums (ground truth and estimated) and a point cloud
        in Open3D. Assumes Camera 0 is at the origin (identity pose).
        
        Args:
            K (np.ndarray): 3x3 intrinsic matrix.
            master_size (tuple): (width, height) of the image used by MAST3R.
            T_C0_C1_gt (np.ndarray): 4x4 Ground Truth transformation matrix from C1 to C0.
            T_C0_C1_est (np.ndarray): 4x4 Estimated transformation matrix from C1 to C0.
            pts3d_anchor_frame (np.ndarray): (N, 3) array of 3D points in Camera 0's frame.
        """
        if o3d is None:
            print("Open3D is not installed. Cannot perform 3D visualization.")
            return

        def _create_camera_frame(K_mat, pose, scale=0.05, color=[1, 0, 0], image_W=None, image_H=None):
            """
            Helper to create a camera frustum visualization in Open3D.
            """
            if image_W is None or image_H is None:
                # Fallback to master_size if specific image dimensions not provided
                w, h = master_size 
            else:
                w, h = image_W, image_H

            fx, fy = K_mat[0, 0], K_mat[1, 1]
            cx, cy = K_mat[0, 2], K_mat[1, 2]

            # Frustum corners in camera frame (z-forward)
            # Standard camera coordinates: +X right, +Y down, +Z forward
            corners = np.array([
                [ (0 - cx) / fx,  (0 - cy) / fy, 1.0],  # Top-left
                [ (w - cx) / fx,  (0 - cy) / fy, 1.0],  # Top-right
                [ (w - cx) / fx,  (h - cy) / fy, 1.0],  # Bottom-right
                [ (0 - cx) / fx,  (h - cy) / fy, 1.0]   # Bottom-left
            ])
            corners *= scale

            # Add the origin (camera center)
            points = np.vstack(([0, 0, 0], corners))

            # Apply the pose (which transforms points from camera frame to world frame)
            # If `pose` is T_World_Camera, then `pose @ points_h.T` transforms points in camera frame to world.
            # Here, pose is T_C0_C1 (pose of C1 in C0 frame), so we use it to represent C1's position.
            # For C0, it's np.eye(4) (C0 at origin).
            points_h = np.hstack((points, np.ones((points.shape[0], 1))))
            transformed = (pose @ points_h.T).T[:, :3]

            lines = [[0,1],[0,2],[0,3],[0,4],[1,2],[2,3],[3,4],[4,1]]
            colors = [color for _ in lines]

            cam_frustum = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(transformed),
                lines=o3d.utility.Vector2iVector(lines)
            )
            cam_frustum.colors = o3d.utility.Vector3dVector(colors)
            return cam_frustum

        # --- Camera 0 (Reference - at origin) ---
        pose_cam0_world = np.eye(4)
        cam0_frustum = _create_camera_frame(K, pose_cam0_world, scale=0.05, color=[0,1,0], image_W=master_size[0], image_H=master_size[1])  # Green for Camera 0

        # --- Camera 1 (Ground Truth) ---
        # T_C0_C1_gt is the pose of Camera 1 relative to Camera 0. Use it directly for frustum of Camera 1.
        cam1_gt_frustum = _create_camera_frame(K, T_C0_C1_gt, scale=0.05, color=[0,0,1], image_W=master_size[0], image_H=master_size[1]) # Blue for GT Camera 1

        # --- Camera 1 (Estimated) ---
        # T_C0_C1_est is the estimated pose of Camera 1 relative to Camera 0.
        cam1_est_frustum = _create_camera_frame(K, T_C0_C1_est, scale=0.05, color=[1,0,0], image_W=master_size[0], image_H=master_size[1]) # Red for Estimated Camera 1

        # --- Point Cloud (from Camera 0's frame) ---
        # Filter out invalid points (NaNs, Infs)
        points3D_filtered = pts3d_anchor_frame[np.all(np.isfinite(pts3d_anchor_frame), axis=1)]
        
        pcd = o3d.geometry.PointCloud()
        if points3D_filtered.shape[0] > 0:
            pcd.points = o3d.utility.Vector3dVector(points3D_filtered)
            pcd.paint_uniform_color([0.7, 0.7, 0.7]) # Grey for point cloud
        else:
            print("Warning: No valid 3D points to visualize.")

        # --- Visualization ---
        print("\n--- Starting Open3D visualization ---")
        vis_objects = [pcd, cam0_frustum, cam1_gt_frustum, cam1_est_frustum]
        
        # Add coordinate frame for reference
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        vis_objects.append(mesh_frame)

        o3d.visualization.draw_geometries(vis_objects, 
                                          window_name="MAST3R Pose Visualization: Green=Cam0, Blue=GT Cam1, Red=Est Cam1")
        print("Open3D visualization finished.")