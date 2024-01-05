import numpy as np
import mmcv
import torch
from PIL import Image
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmdet.datasets.builder import PIPELINES
from pyquaternion import Quaternion

def mmlabNormalize(img, img_norm_cfg=None):
    from mmcv.image.photometric import imnormalize
    if img_norm_cfg is None:
        mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
        to_rgb = True
    else:
        mean = np.array(img_norm_cfg['mean'], dtype=np.float32)
        std = np.array(img_norm_cfg['std'], dtype=np.float32)
        to_rgb = img_norm_cfg['to_rgb']

    img = imnormalize(np.array(img), mean, std, to_rgb)
    img = torch.tensor(img).float().permute(2, 0, 1).contiguous()
    return img

@PIPELINES.register_module()
class PrepareImageInputs(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(
        self,
        data_config,
        is_train=False,
        sequential=False,
        align_after_view_trans=False,
        ego_cam='CAM_FRONT',
        file_client_args=None,
        add_adj_bbox=False,
        with_future_pred=False,
        img_norm_cfg=None,
    ):
        self.is_train = is_train
        self.data_config = data_config
        self.normalize_img = mmlabNormalize
        self.sequential = sequential
        self.align_after_view_trans = align_after_view_trans
        self.ego_cam = ego_cam
        self.file_client_args = file_client_args.copy()
        if self.file_client_args['backend'] == 'petrel':
            self.file_client = mmcv.FileClient(**self.file_client_args)
        self.with_future_pred = with_future_pred
        self.add_adj_bbox = add_adj_bbox
        self.img_norm_cfg = img_norm_cfg

    def get_rot(self, h):
        return torch.Tensor([
            [np.cos(h), np.sin(h)],
            [-np.sin(h), np.cos(h)],
        ])

    def img_transform(self, img, post_rot, post_tran, resize, resize_dims,
                      crop, flip, rotate):
        # adjust image
        img = self.img_transform_core(img, resize_dims, crop, flip, rotate)

        # post-homography transformation
        post_rot *= resize
        post_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            post_rot = A.matmul(post_rot)
            post_tran = A.matmul(post_tran) + b
        A = self.get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b

        return img, post_rot, post_tran

    def img_transform_core(self, img, resize_dims, crop, flip, rotate):
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)
        return img

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

    def sample_augmentation(self, H, W, flip=None, scale=None):
        fH, fW = self.data_config['input_size']
        if self.is_train:
            resize = float(fW) / float(W)
            resize += np.random.uniform(*self.data_config['resize'])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_config['crop_h'])) *
                         newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = self.data_config['flip'] and np.random.choice([0, 1])
            rotate = np.random.uniform(*self.data_config['rot'])
        else:
            resize = float(fW) / float(W)
            resize += self.data_config.get('resize_test', 0.0)
            if scale is not None:
                resize = scale
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_config['crop_h'])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False if flip is None else flip
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def get_sensor2ego_transformation(self,
                                      cam_info,
                                      key_info,
                                      cam_name,
                                      ego_cam=None):
        if ego_cam is None:
            ego_cam = cam_name
        w, x, y, z = cam_info['cams'][cam_name]['sensor2ego_rotation']
        # sweep sensor to sweep ego
        sweepsensor2sweepego_rot = torch.Tensor(
            Quaternion(w, x, y, z).rotation_matrix)
        sweepsensor2sweepego_tran = torch.Tensor(
            cam_info['cams'][cam_name]['sensor2ego_translation'])
        sweepsensor2sweepego = sweepsensor2sweepego_rot.new_zeros((4, 4))
        sweepsensor2sweepego[3, 3] = 1
        sweepsensor2sweepego[:3, :3] = sweepsensor2sweepego_rot
        sweepsensor2sweepego[:3, -1] = sweepsensor2sweepego_tran
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

        sweepsensor2keyego = \
            global2keyego @ sweepego2global @ sweepsensor2sweepego

        # global sensor to cur ego
        w, x, y, z = key_info['cams'][cam_name]['ego2global_rotation']
        keyego2global_rot = torch.Tensor(
            Quaternion(w, x, y, z).rotation_matrix)
        keyego2global_tran = torch.Tensor(
            key_info['cams'][cam_name]['ego2global_translation'])
        keyego2global = keyego2global_rot.new_zeros((4, 4))
        keyego2global[3, 3] = 1
        keyego2global[:3, :3] = keyego2global_rot
        keyego2global[:3, -1] = keyego2global_tran
        global2keyego = keyego2global.inverse()

        # cur ego to sensor
        w, x, y, z = key_info['cams'][cam_name]['sensor2ego_rotation']
        keysensor2keyego_rot = torch.Tensor(
            Quaternion(w, x, y, z).rotation_matrix)
        keysensor2keyego_tran = torch.Tensor(
            key_info['cams'][cam_name]['sensor2ego_translation'])
        keysensor2keyego = keysensor2keyego_rot.new_zeros((4, 4))
        keysensor2keyego[3, 3] = 1
        keysensor2keyego[:3, :3] = keysensor2keyego_rot
        keysensor2keyego[:3, -1] = keysensor2keyego_tran
        keyego2keysensor = keysensor2keyego.inverse()
        keysensor2sweepsensor = (
            keyego2keysensor @ global2keyego @ sweepego2global
            @ sweepsensor2sweepego).inverse()
        return sweepsensor2keyego, keysensor2sweepsensor

    def get_inputs(self, results, flip=None, scale=None):
        imgs = []
        rots = []
        trans = []
        intrins = []
        post_rots = []
        post_trans = []
        ego2img = []
        cam_names = self.choose_cams()

        results['cam_names'] = cam_names
        canvas = []
        sensor2sensors = []
        for cam_name in cam_names:
            cam_data = results['curr']['cams'][cam_name]
            filename = cam_data['data_path']
            # Potential BUG: Load img type
            img = self.load_image(filename)
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)

            intrin = torch.Tensor(cam_data['cam_intrinsic'])

            sensor2keyego, sensor2sensor = \
                self.get_sensor2ego_transformation(results['curr'],
                                                   results['curr'],
                                                   cam_name,
                                                   self.ego_cam)
            rot = sensor2keyego[:3, :3]
            tran = sensor2keyego[:3, 3]
            # image view augmentation (resize, crop, horizontal flip, rotate)
            img_augs = self.sample_augmentation(
                H=img.height, W=img.width, flip=flip, scale=scale)
            resize, resize_dims, crop, flip, rotate = img_augs
            img, post_rot2, post_tran2 = \
                self.img_transform(img, post_rot,
                                   post_tran,
                                   resize=resize,
                                   resize_dims=resize_dims,
                                   crop=crop,
                                   flip=flip,
                                   rotate=rotate)

            # for convenience, make augmentation matrices 3x3
            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)
            post_tran[:2] = post_tran2
            post_rot[:2, :2] = post_rot2

            canvas.append(np.array(img))
            imgs.append(self.normalize_img(img, self.img_norm_cfg))

            if self.sequential:
                assert 'adjacent' in results
                if self.with_future_pred:
                    adjacent_results = results['adjacent'][:-1]
                else:
                    adjacent_results = results['adjacent']
                for adj_info in adjacent_results:
                    filename_adj = adj_info['cams'][cam_name]['data_path']
                    # img_adjacent = Image.open(filename_adj)
                    img_adjacent = self.load_image(filename_adj)
                    img_adjacent = self.img_transform_core(
                        img_adjacent,
                        resize_dims=resize_dims,
                        crop=crop,
                        flip=flip,
                        rotate=rotate)
                    imgs.append(self.normalize_img(img_adjacent, self.img_norm_cfg))
            intrins.append(intrin)
            rots.append(rot)
            trans.append(tran)
            post_rots.append(post_rot)
            post_trans.append(post_tran)
            sensor2sensors.append(sensor2sensor)

        if self.sequential:
            if self.with_future_pred:
                adjacent_results = results['adjacent'][:-1]
            else:
                adjacent_results = results['adjacent']
            for adj_info in adjacent_results:
                post_trans.extend(post_trans[:len(cam_names)])
                post_rots.extend(post_rots[:len(cam_names)])
                intrins.extend(intrins[:len(cam_names)])

                # align
                trans_adj = []
                rots_adj = []
                sensor2sensors_adj = []
                for cam_name in cam_names:
                    if self.align_after_view_trans:
                        adjsensor2keyego, sensor2sensor = \
                            self.get_sensor2ego_transformation(adj_info,
                                                            adj_info,
                                                            cam_name,
                                                            self.ego_cam)
                    else:
                        adjsensor2keyego, sensor2sensor = \
                            self.get_sensor2ego_transformation(adj_info,
                                                            results['curr'],
                                                            cam_name,
                                                            self.ego_cam)
                    rot = adjsensor2keyego[:3, :3]
                    tran = adjsensor2keyego[:3, 3]
                    rots_adj.append(rot)
                    trans_adj.append(tran)
                    sensor2sensors_adj.append(sensor2sensor)
                rots.extend(rots_adj)
                trans.extend(trans_adj)
                sensor2sensors.extend(sensor2sensors_adj)
            if self.add_adj_bbox:
                results['adjacent_bboxes'] = self.align_adj_bbox2keyego(results)
        imgs = torch.stack(imgs)
        rots = torch.stack(rots)
        trans = torch.stack(trans)
        intrins = torch.stack(intrins)
        post_rots = torch.stack(post_rots)
        post_trans = torch.stack(post_trans)
        sensor2sensors = torch.stack(sensor2sensors)
        results['canvas'] = canvas
        results['sensor2sensors'] = sensor2sensors
        results['img_shape'] = [(self.data_config['input_size'][0], self.data_config['input_size'][1]) for _ in range(6)]
        return (imgs, rots, trans, intrins, post_rots, post_trans)

    def __call__(self, results):
        if self.add_adj_bbox:
            results['adjacent_bboxes'] = self.get_adjacent_bboxes(results)
        results['img_inputs'] = self.get_inputs(results)
        return results

    def load_image(self, filename, color_type='color'):
        ''' Adapt for petrel. Origin Implementation: img = Image.open(filename)
        copy from LoadMultiViewImageFromFiles
        Image.open() default is RGB files
        Validated 
        '''
        if self.file_client_args is None or self.file_client_args['backend'] == 'disk':
            load_fun = lambda x: Image.open(x)

        elif self.file_client_args['backend'] == 'petrel':
            def petrel_load_image(name, color_type):
                img_bytes = self.file_client.get(name)
                img_array = mmcv.imfrombytes(img_bytes, flag=color_type, channel_order='rgb', backend='pillow')
                return Image.fromarray(img_array.astype(np.uint8), mode='RGB')

            load_fun = lambda x: petrel_load_image(x, color_type)
        else:
            raise NotImplementedError(f'File client args is {self.file_client_args}')

        img_pil = load_fun(filename)
        return img_pil
    
    def get_adjacent_bboxes(self, results):
        adjacent_bboxes = list()
        for adj_info in results['adjacent']:
            adjacent_bboxes.append(adj_info['ann_infos'])
        return adjacent_bboxes
    
    
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
    
    def align_adj_bbox2keyego(self, results):
        cam_name = self.choose_cams()[0]
        ret_list = []
        for idx, adj_info in enumerate(results['adjacent']):
            sweepego2keyego = self.get_sweep2key_transformation(adj_info,
                                results['curr'],
                                cam_name,
                                self.ego_cam)
            adj_bbox, adj_labels = results['adjacent_bboxes'][idx]
            adj_bbox = torch.Tensor(adj_bbox)
            adj_labels = torch.tensor(adj_labels)
            gt_bbox = adj_bbox
            if len(adj_bbox) == 0:
                adj_bbox = torch.zeros(0, 9)
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

            new_yaw_sweep = get_new_yaw(LiDARInstance3DBoxes(adj_bbox, box_dim=adj_bbox.shape[-1],
                                        origin=(0.5, 0.5, 0.5)), sweepego2keyego).reshape(-1,1)
            adj_bbox = torch.cat([homo_key_center[:, :3], gt_bbox[:,3:6], new_yaw_sweep, homo_key_velo[:, :2]], dim=-1)
            ret_list.append((adj_bbox, adj_labels))

        return ret_list

@PIPELINES.register_module()
class LoadAnnotationsBEVDepth(object):

    def __init__(self, bda_aug_conf, classes, is_train=True, sequential=False, align_adj_bbox=False):
        self.bda_aug_conf = bda_aug_conf
        self.is_train = is_train
        self.classes = classes
        self.sequential = sequential
        self.align_adj_bbox = align_adj_bbox

    def sample_bda_augmentation(self):
        """Generate bda augmentation values based on bda_config."""
        if self.is_train:
            rotate_bda = np.random.uniform(*self.bda_aug_conf['rot_lim'])
            scale_bda = np.random.uniform(*self.bda_aug_conf['scale_lim'])
            flip_dx = np.random.uniform() < self.bda_aug_conf['flip_dx_ratio']
            flip_dy = np.random.uniform() < self.bda_aug_conf['flip_dy_ratio']
        else:
            rotate_bda = 0
            scale_bda = 1.0
            flip_dx = False
            flip_dy = False
        return rotate_bda, scale_bda, flip_dx, flip_dy

    def bev_transform(self, gt_boxes, rotate_angle, scale_ratio, flip_dx,
                      flip_dy):
        rotate_angle = torch.tensor(rotate_angle / 180 * np.pi)
        rot_sin = torch.sin(rotate_angle)
        rot_cos = torch.cos(rotate_angle)
        rot_mat = torch.Tensor([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0],
                                [0, 0, 1]])
        scale_mat = torch.Tensor([[scale_ratio, 0, 0], [0, scale_ratio, 0],
                                  [0, 0, scale_ratio]])
        flip_mat = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        if flip_dx:
            flip_mat = flip_mat @ torch.Tensor([[-1, 0, 0], [0, 1, 0],
                                                [0, 0, 1]])
        if flip_dy:
            flip_mat = flip_mat @ torch.Tensor([[1, 0, 0], [0, -1, 0],
                                                [0, 0, 1]])
        rot_mat = flip_mat @ (scale_mat @ rot_mat)
        if gt_boxes.shape[0] > 0:
            gt_boxes[:, :3] = (
                rot_mat @ gt_boxes[:, :3].unsqueeze(-1)).squeeze(-1)
            gt_boxes[:, 3:6] *= scale_ratio
            gt_boxes[:, 6] += rotate_angle
            if flip_dx:
                gt_boxes[:,
                         6] = 2 * torch.asin(torch.tensor(1.0)) - gt_boxes[:,
                                                                           6]
            if flip_dy:
                gt_boxes[:, 6] = -gt_boxes[:, 6]
            gt_boxes[:, 7:] = (
                rot_mat[:2, :2] @ gt_boxes[:, 7:].unsqueeze(-1)).squeeze(-1)
        return gt_boxes, rot_mat

    def __call__(self, results):
        gt_boxes, gt_labels = results['ann_infos']
        gt_boxes, gt_labels = torch.Tensor(gt_boxes), torch.tensor(gt_labels)
        rotate_bda, scale_bda, flip_dx, flip_dy = self.sample_bda_augmentation(
        )
        bda_mat = torch.zeros(4, 4)
        bda_mat[3, 3] = 1
        gt_boxes, bda_rot = self.bev_transform(gt_boxes, rotate_bda, scale_bda,
                                               flip_dx, flip_dy)
        bda_mat[:3, :3] = bda_rot
        if len(gt_boxes) == 0:
            gt_boxes = torch.zeros(0, 9)
        results['gt_bboxes_3d'] = \
            LiDARInstance3DBoxes(gt_boxes, box_dim=gt_boxes.shape[-1],
                                 origin=(0.5, 0.5, 0.5))
        results['gt_labels_3d'] = gt_labels
        imgs, rots, trans, intrins = results['img_inputs'][:4]
        post_rots, post_trans = results['img_inputs'][4:]
        results['img_inputs'] = (imgs, rots, trans, intrins, post_rots,
                                 post_trans, bda_rot)
        ego2img_rts = []
        if not self.sequential:
            for rot, tran, intrin, post_rot, post_tran in zip(
                    rots, trans, intrins, post_rots, post_trans):
                viewpad = torch.eye(3).to(imgs.device)
                viewpad[:post_rot.shape[0], :post_rot.shape[1]] = \
                    post_rot @ intrin[:post_rot.shape[0], :post_rot.shape[1]]
                viewpad[:post_tran.shape[0], 2] += post_tran
                intrinsic = viewpad

                ego2img_r = intrinsic @ torch.linalg.inv(rot) @ torch.linalg.inv(bda_rot)
                ego2img_t = -intrinsic @ torch.linalg.inv(rot) @ tran
                ego2img_rt = torch.eye(4).to(imgs.device)
                ego2img_rt[:3, :3] = ego2img_r
                ego2img_rt[:3, 3] = ego2img_t
                '''
                X_{3d} = bda * (rots * (intrinsic)^(-1) * X_{img} + trans)
                bda^(-1) * X_{3d} = rots * (intrinsic)^(-1) * X_{img} + trans
                bda^(-1) * X_{3d} - trans = rots * (intrinsic)^(-1) * X_{img}
                intrinsic * rots^(-1) * (bda^(-1) * X_{3d} - trans) = X_{img}
                intrinsic * rots^(-1) * bda^(-1) * X_{3d} - intrinsic * rots^(-1) * trans = X_{img}
                rotate = intrinsic * rots^(-1) * bda^(-1)
                translation = - intrinsic * rots^(-1) * trans
                '''
                ego2img_rts.append(ego2img_rt)
            ego2img_rts = torch.stack(ego2img_rts, dim=0)
        if self.align_adj_bbox:
            results = self.align_adj_bbox_bda(results, rotate_bda, scale_bda,
                                               flip_dx, flip_dy)
        results['lidar2img'] = np.asarray(ego2img_rts)
        return results

    def align_adj_bbox_bda(self, results, rotate_bda, scale_bda, flip_dx, flip_dy):
        for adjacent_bboxes in results['adjacent_bboxes']:
            adj_bbox, adj_label = adjacent_bboxes
            gt_boxes = adj_bbox
            if len(gt_boxes) == 0:
                gt_boxes = torch.zeros(0, 9)
            gt_boxes, _ = self.bev_transform(gt_boxes, rotate_bda, scale_bda,
                                        flip_dx, flip_dy)
            if not 'adj_gt_3d' in results.keys():
                adj_bboxes_3d = \
                    LiDARInstance3DBoxes(gt_boxes, box_dim=gt_boxes.shape[-1],
                                        origin=(0.5, 0.5, 0.5))
                adj_labels_3d = adj_label
                results['adj_gt_3d'] = [[adj_bboxes_3d, adj_labels_3d]]
            else:
                adj_bboxes_3d = \
                    LiDARInstance3DBoxes(gt_boxes, box_dim=gt_boxes.shape[-1],
                                        origin=(0.5, 0.5, 0.5))
                results['adj_gt_3d'].append([
                    adj_bboxes_3d, adj_label
                ])
        return results 