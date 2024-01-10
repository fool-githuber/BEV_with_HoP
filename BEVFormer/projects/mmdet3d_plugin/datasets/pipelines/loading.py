import numpy as np
import mmcv
import torch
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmdet.datasets.builder import PIPELINES
from pyquaternion import Quaternion

@PIPELINES.register_module()
class LoadMultiViewImageFromFiles_HoP(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(self, to_float32=False,
                 color_type='unchanged', 
                 add_adj_bbox=False, 
                 ego_cam="CAM_FRONT",
                 data_config=None,
                 is_train=False):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.add_adj_bbox = add_adj_bbox
        self.ego_cam = ego_cam
        self.data_config = data_config
        self.is_train = is_train

    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data. \
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        if self.add_adj_bbox:
            results['adjacent_bboxes'] = self.get_adjacent_bboxes(results)
            results['adjacent_bboxes'] = self.align_adj_bbox2keyego(results)

        filename = results['img_filename']
        # img is of shape (h, w, c, num_views)
        img = np.stack(
            [mmcv.imread(name, self.color_type) for name in filename], axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)
        results['filename'] = filename
        # unravel to list, see `DefaultFormatBundle` in formating.py
        # which will transpose each image separately and then stack into array
        results['img'] = [img[..., i] for i in range(img.shape[-1])]
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}, '
        repr_str += f"color_type='{self.color_type}')"
        return repr_str
    
    def get_adjacent_bboxes(self, results):
        adjacent_bboxes = list()
        for adj_info in results['adjacent']:
            adjacent_bboxes.append(adj_info['ann_info'])
        return adjacent_bboxes
    
    def align_adj_bbox2keyego(self, results):
        cam_name = self.choose_cams()[0]
        ret_list = []
        for idx, adj_info in enumerate(results['adjacent']):
            sweepego2keyego = self.get_sweep2key_transformation(adj_info,
                                results['curr'],
                                cam_name,
                                self.ego_cam)
            adj_bbox, adj_labels, adj_names = results['adjacent_bboxes'][idx]['gt_bboxes_3d'], results['adjacent_bboxes'][idx]['gt_labels_3d'], results['adjacent_bboxes'][idx]['gt_names']
            # adj_bbox = adj_bbox.tensor
            adj_labels = torch.tensor(adj_labels)
            gt_bbox = adj_bbox.tensor
            if len(adj_bbox.tensor) == 0:
                adj_bbox.tensor = torch.zeros(0, 9)
                ret_list.append((adj_bbox, adj_labels))
                continue
            # center
            homo_sweep_center = torch.cat([gt_bbox[:,:3], torch.ones_like(gt_bbox[:,0:1])], dim=-1)
            homo_key_center = (sweepego2keyego @ homo_sweep_center.t()).t() # [4, N]
            # velo
            rot = sweepego2keyego[:3, :3]
            homo_sweep_velo =  torch.cat([gt_bbox[:, 7:], torch.zeros_like(gt_bbox[:,0:1])], dim=-1)
            homo_key_velo = (rot @ homo_sweep_velo.t()).t()
            # yaw
            def get_new_yaw(box_cam, extrinsic):
                corners = box_cam.corners
                cam2lidar_rt = torch.tensor(extrinsic)
                N = corners.shape[0]
                corners = corners.reshape(N*8, 3)
                extended_xyz = torch.cat(
                    [corners, corners.new_ones(corners.size(0), 1)], dim=-1)
                corners = extended_xyz @ cam2lidar_rt.T
                corners = corners.reshape(N, 8, 4)[:, :, :3]
                yaw = np.arctan2(corners[:,1,1]-corners[:,2,1], corners[:,1,0]-corners[:,2,0])
                def limit_period(val, offset=0.5, period=np.pi):
                    """Limit the value into a period for periodic function.

                    Args:
                        val (np.ndarray): The value to be converted.
                        offset (float, optional): Offset to set the value range. \
                            Defaults to 0.5.
                        period (float, optional): Period of the value. Defaults to np.pi.

                    Returns:
                        torch.Tensor: Value in the range of \
                            [-offset * period, (1-offset) * period]
                    """
                    return val - np.floor(val / period + offset) * period
                return limit_period(yaw + (np.pi/2), period=np.pi * 2)

            new_yaw_sweep = get_new_yaw(LiDARInstance3DBoxes(adj_bbox.tensor, box_dim=adj_bbox.tensor.shape[-1],
                                        origin=(0.5, 0.5, 0.5)), sweepego2keyego).reshape(-1,1)
            adj_bbox.tensor = torch.cat([homo_key_center[:, :3], gt_bbox[:,3:6], new_yaw_sweep, homo_key_velo[:, :2]], dim=-1)
            ret_list.append((adj_bbox, adj_labels))

        return ret_list
    
    def choose_cams(self):
        if self.is_train and self.data_config['Ncams'] < len(
                self.data_config['cams']):
            cam_names = np.random.choice(
                self.data_config['cams'],
                self.data_config['Ncams'],
                replace=False)
        else:
            cam_names = self.data_config['cams']
        return cam_names
    
    def get_sweep2key_transformation(self,
                                    cam_info,
                                    key_info,
                                    cam_name,
                                    ego_cam=None):
        if ego_cam is None:
            ego_cam = cam_name
        # sweep ego to global
        w, x, y, z = cam_info['cams'][cam_name]['ego2global_rotation']
        sweepego2global_rot = torch.Tensor(
            Quaternion(w, x, y, z).rotation_matrix)
        sweepego2global_tran = torch.Tensor(
            cam_info['cams'][cam_name]['ego2global_translation'])
        sweepego2global = sweepego2global_rot.new_zeros((4, 4))
        sweepego2global[3, 3] = 1
        sweepego2global[:3, :3] = sweepego2global_rot
        sweepego2global[:3, -1] = sweepego2global_tran

        # global sensor to cur ego
        w, x, y, z = key_info['cams'][ego_cam]['ego2global_rotation']
        keyego2global_rot = torch.Tensor(
            Quaternion(w, x, y, z).rotation_matrix)
        keyego2global_tran = torch.Tensor(
            key_info['cams'][ego_cam]['ego2global_translation'])
        keyego2global = keyego2global_rot.new_zeros((4, 4))
        keyego2global[3, 3] = 1
        keyego2global[:3, :3] = keyego2global_rot
        keyego2global[:3, -1] = keyego2global_tran
        global2keyego = keyego2global.inverse()

        sweepego2keyego = global2keyego @ sweepego2global

        return sweepego2keyego