import argparse
import cv2
import numpy as np
import os
import seaborn as sns
from PIL import Image
import logging
import math

import torch
import torch.utils.data
import torch.optim as optim
import torch.nn.functional as F
from pytorch3d.io import IO as io3d
from configs import make_cfg

from models.model import Model

from utils.train_util import cpu_data_to_gpu
from utils.image_util import to_8b_image
from utils.tb_util import TBLogger

# from utils.lpips import LPIPS
from skimage.metrics import structural_similarity
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import trimesh
from matplotlib import cm

EXCLUDE_KEYS_TO_GPU = ['frame_name', 'img_width', 'img_height']
POSE_COLORS = (np.array(sns.color_palette("hls", 36)) * 255.).astype(int).tolist()

os.environ['TORCH_HOME'] = '/ubc/cs/home/c/chunjins/chunjin_shield/project/torch'

def parse_args():
	parser = argparse.ArgumentParser()

	parser.add_argument(
		"--type",
		default='mesh_novel_pose',
		choices=['view', 'pose', 'train', 'freeview', 'pose_mdm', 'video', 'mesh_training', 'mesh_novel_pose', 'mesh_novel_view'],
		type=str
	)
	parser.add_argument(
		"--cfg",
		default='exps/mvhuman/mvhuman_200173.yaml',
		type=str
	)
	parser.add_argument(
		"--iter",
		default=None,
		type=int
	)

	parser.add_argument(
		"--frame_idx",
		default=0,
		type=int,
		help="freeview only"
	)
	parser.add_argument(
		"--n_frames",
		default=100,
		type=int,
		help="freeview only"
	)

	parser.add_argument(
		"--bgcolor",
		default=None,
		type=float,
		help="background color that overrides the config file, range [0, 255]"
	)

	parser.add_argument(
		"--pose_path",
		default='data/mdm_poses/sample.npy',
		type=str
	)

	return parser.parse_args()


def unpack(rgbs, masks, bgcolors):
	rgbs = rgbs * masks.unsqueeze(-1) + bgcolors[:, None, None, :] * (1 - masks).unsqueeze(-1)
	rgbs = torch.clamp(rgbs, min=0, max=1)
	return rgbs

def apply_depth_map(
    depth_data,
    mask,
):
	mask = mask > 0.9
	depth_fg = depth_data[mask]  ## value in range [0, 1]
	depth_map = torch.zeros_like(depth_data)
	if len(depth_fg) > 0:
		min_val, max_val = torch.min(depth_fg), torch.max(depth_fg)
		depth_normalized_foreground = 1 - (
                (depth_fg - min_val) / (max_val - min_val)
        )  ## for visualization, foreground is 1 (white), background is 0 (black)

		depth_map[mask] = depth_normalized_foreground

	depth_map = torch.cat((depth_map, depth_map, depth_map), axis=-1)
	return depth_map

def depths_to_points(K, H, W, depthmap):
    intrins = torch.tensor(K).float().cuda()
    grid_x, grid_y = torch.meshgrid(torch.arange(W, device='cuda').float() + 0.5, torch.arange(H, device='cuda').float() + 0.5, indexing='xy')
    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)
    rays_d = points @ intrins.inverse().T
    points = depthmap.reshape(-1, 1) * rays_d
    return points

def depth_to_normal(K, H, W, depth):
    """
        view: view camera
        depth: depthmap
    """
    points = depths_to_points(K, H, W, depth).reshape(H, W, 3)
    points[..., 1] = points[..., 1] * -1
    points[..., 2] = points[..., 2] * -1
    output = torch.zeros_like(points)
    dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
    dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    output[1:-1, 1:-1, :] = normal_map
    return output, points

def apply_colormap(image, cmap="viridis"):
    colormap = cm.get_cmap(cmap)
    colormap = torch.tensor(colormap.colors).to(image.device)  # type: ignore
    image_long = (image * 255).long()
    image_long_min = torch.min(image_long)
    image_long_max = torch.max(image_long)
    assert image_long_min >= 0, f"the min value is {image_long_min}"
    assert image_long_max <= 255, f"the max value is {image_long_max}"
    return colormap[image_long[..., 0]]


def apply_depth_colormap(
    depth,
    accumulation,
    near_plane = 2.0,
    far_plane = 6.0,
    cmap="turbo",
):
    near_plane = near_plane or float(torch.min(depth))
    far_plane = far_plane or float(torch.max(depth))

    depth = (depth - near_plane) / (far_plane - near_plane + 1e-10)
    depth = torch.clip(depth, 0, 1)
    # depth = torch.nan_to_num(depth, nan=0.0) # TODO(ethan): remove this

    colored_image = apply_colormap(depth, cmap=cmap)

    if accumulation is not None:
        colored_image = colored_image * accumulation + (1 - accumulation)

    return colored_image



class Evaluator:
	"""
	copied from https://github.com/zju3dv/neuralbody/blob/6bf1905822f71d1e568ef831110728fd1d06c94d/lib/evaluators/neural_volume.py
	adapted from https://github.com/escapefreeg/humannerf-eval/blob/master/eval.py
	"""

	def __init__(self):
		self.lpips_model = LPIPS(net='vgg').cuda()
		for param in self.lpips_model.parameters():
			param.requires_grad = False
		self.mse = []
		self.psnr = []
		self.ssim = []
		self.lpips = []

	def psnr_metric(self, img_pred, img_gt):
		mse = np.mean((img_pred - img_gt) ** 2)
		psnr = -10 * np.log(mse) / np.log(10)
		return psnr

	def ssim_metric(self, img_pred, img_gt):
		ssim = structural_similarity(img_pred, img_gt, multichannel=True)
		return ssim

	def lpips_metric(self, img_pred, img_gt):
		# convert range from 0-1 to -1-1
		processed_pred = torch.from_numpy(img_pred).float().unsqueeze(0).cuda() * 2. - 1.
		processed_gt = torch.from_numpy(img_gt).float().unsqueeze(0).cuda() * 2. - 1.

		lpips_loss = self.lpips_model(processed_pred.permute(0, 3, 1, 2), processed_gt.permute(0, 3, 1, 2))
		return torch.mean(lpips_loss).cpu().detach().item() * 1000

	def evaluate(self, rgb_pred, rgb_gt):
		mse = np.mean((rgb_pred - rgb_gt) ** 2)
		self.mse.append(mse)

		psnr = self.psnr_metric(rgb_pred, rgb_gt)
		self.psnr.append(psnr)

		ssim = self.ssim_metric(rgb_pred, rgb_gt)
		self.ssim.append(ssim)

		lpips = self.lpips_metric(rgb_pred, rgb_gt)
		self.lpips.append(lpips)

	def summarize(self, path):
		result_path = os.path.join(path)
		os.system('mkdir -p {}'.format(os.path.dirname(result_path)))
		metrics = {'mse': self.mse, 'psnr': self.psnr, 'ssim': self.ssim, 'lpips': self.lpips}
		np.save(result_path, metrics)
		print('mse: {}'.format(np.mean(self.mse)))
		print('psnr: {}'.format(np.mean(self.psnr)))
		print('ssim: {}'.format(np.mean(self.ssim)))
		print('lpips: {}'.format(np.mean(self.lpips)))
		self.mse = []
		self.psnr = []
		self.ssim = []
		self.lpips = []


class Evaluator_snapshot:
	"""
	adapted from https://github.com/JanaldoChen/Anim-NeRF/blob/main/models/evaluator.py
	"""
	def __init__(self):
		self.psnr = []
		self.ssim = []
		self.lpips = []

		self.lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type="alex")
		self.psnr_metric = PeakSignalNoiseRatio(data_range=1)
		self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1)

	# custom_fwd: turn off mixed precision to avoid numerical instability during evaluation
	def evaluate(self, rgb_pred, rgb_gt):
		# torchmetrics assumes NCHW format
		processed_pred = torch.from_numpy(rgb_pred).float().unsqueeze(0).permute(0, 3, 1, 2)
		processed_gt = torch.from_numpy(rgb_gt).float().unsqueeze(0).permute(0, 3, 1, 2)

		self.psnr.append(self.psnr_metric(processed_pred, processed_gt).detach().cpu().numpy())
		self.ssim.append(self.ssim_metric(processed_pred, processed_gt).detach().cpu().numpy())
		self.lpips.append(self.lpips_metric(processed_pred, processed_gt).detach().cpu().numpy())

	def summarize(self, path):
		result_path = os.path.join(path)
		os.system('mkdir -p {}'.format(os.path.dirname(result_path)))
		metrics = {'psnr': self.psnr, 'ssim': self.ssim, 'lpips': self.lpips}
		np.save(result_path, metrics)
		# print('mse: {}'.format(np.mean(self.mse)))
		print('psnr: {}'.format(np.mean(self.psnr)))
		print('ssim: {}'.format(np.mean(self.ssim)))
		print('lpips: {}'.format(np.mean(self.lpips)))
		self.psnr = []
		self.ssim = []
		self.lpips = []


def main(args):
	# configs
	cfg = make_cfg(args.cfg)
	cfg.model.eval_mode = True

	if args.type == 'pose_mdm':
		cfg.img_size = [512, 512]
		cfg.model.img_size = [512, 512]

	if args.bgcolor is not None:
		cfg.bgcolor = [args.bgcolor, args.bgcolor, args.bgcolor]

	if args.pose_path is not None:
		cfg.dataset.test_pose_mdm.pose_path = args.pose_path

	if args.type == 'mesh_training' or args.type == 'mesh_novel_view' or args.type == 'mesh_novel_pose':
		save_dir = os.path.join(cfg.save_dir, 'eval', args.type)
		os.makedirs(save_dir, exist_ok=True)
	else:
		save_dir = os.path.join(cfg.save_dir, 'eval', f'novel_{args.type}_img_geo')
		# save_dir_normal = os.path.join(cfg.save_dir, 'eval', f'novel_{args.type}' + '_normal')
		os.makedirs(save_dir, exist_ok=True)
		# os.makedirs(save_dir_normal, exist_ok=True)

	# setup logger
	logging_path = os.path.join(cfg.save_dir, 'eval', f'log_novel_{args.type}.txt')
	logging.basicConfig(
		handlers=[
			logging.FileHandler(logging_path),
			logging.StreamHandler()
		],
		format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
		level=logging.INFO
	)

	# test dataset
	if args.type == 'view':
		# evaluate novel view synthesis following monohuman's split
		if cfg.dataset.test_view.name == 'zju-mocap':
			from dataset.test import Dataset as NovelViewDataset
			test_dataset = NovelViewDataset(
				cfg.dataset.test_view.raw_dataset_path,
				cfg.dataset.test_view.dataset_path,
				test_type='view',
				skip=cfg.dataset.test_view.skip,  # to match monohuman
				exclude_view=cfg.dataset.test_view.exclude_view,
				bgcolor=cfg.bgcolor,
			)
		else:
			from dataset.train_h5py import Dataset as NovelViewDataset
			test_dataset = NovelViewDataset(
				cfg.dataset.test_view.dataset_path,
				bgcolor=cfg.bgcolor,
				skip=cfg.dataset.test_view.skip,
				target_size=cfg.model.img_size,
			)
		test_dataloader = torch.utils.data.DataLoader(
			batch_size=cfg.dataset.test_view.batch_size,
			dataset=test_dataset,
			shuffle=False,
			drop_last=False,
			num_workers=cfg.dataset.test_view.num_workers)
	elif args.type == 'mesh_training':
		from dataset.train_h5py import Dataset as TrainingDataset
		test_dataset = TrainingDataset(
			cfg.dataset.test_mesh_training.dataset_path,
			bgcolor=cfg.bgcolor,
			skip=cfg.dataset.test_mesh.skip,
			target_size=cfg.model.img_size,
			type=args.type
		)
		test_dataloader = torch.utils.data.DataLoader(
			batch_size=cfg.dataset.test_view.batch_size,
			dataset=test_dataset,
			shuffle=False,
			drop_last=False,
			num_workers=cfg.dataset.test_view.num_workers)

	elif args.type == 'mesh_novel_view':
		from dataset.train_h5py import Dataset as TrainingDataset
		test_dataset = TrainingDataset(
			cfg.dataset.test_mesh_novel_view.dataset_path,
			bgcolor=cfg.bgcolor,
			skip=cfg.dataset.test_mesh.skip,
			target_size=cfg.model.img_size,
			type=args.type
		)
		test_dataloader = torch.utils.data.DataLoader(
			batch_size=cfg.dataset.test_view.batch_size,
			dataset=test_dataset,
			shuffle=False,
			drop_last=False,
			num_workers=cfg.dataset.test_view.num_workers)

	elif args.type == 'mesh_novel_pose':
		from dataset.train_h5py import Dataset as TrainingDataset
		test_dataset = TrainingDataset(
			cfg.dataset.test_mesh_novel_pose.dataset_path,
			bgcolor=cfg.bgcolor,
			skip=cfg.dataset.test_mesh.skip,
			target_size=cfg.model.img_size,
			type=args.type
		)
		test_dataloader = torch.utils.data.DataLoader(
			batch_size=cfg.dataset.test_view.batch_size,
			dataset=test_dataset,
			shuffle=False,
			drop_last=False,
			num_workers=cfg.dataset.test_view.num_workers)
	elif args.type == 'pose':
		# evaluate novel pose synthesis following monohuman's split
		if cfg.dataset.test_pose.name == 'zju-mocap':
			from dataset.test import Dataset as NovelPoseDataset
			test_dataset = NovelPoseDataset(
				cfg.dataset.test_pose.raw_dataset_path,
				cfg.dataset.test_pose.dataset_path,
				test_type='pose',
				skip=cfg.dataset.test_pose.skip,  # to match monohuman
				exclude_training_view=True,   # to match monohuman
				bgcolor=cfg.bgcolor,
			)
		else:
			from dataset.train_h5py import Dataset as NovelPoseDataset
			test_dataset = NovelPoseDataset(
				cfg.dataset.test_pose.dataset_path,
				bgcolor=cfg.bgcolor,
				skip=cfg.dataset.test_pose.skip,
				target_size=cfg.model.img_size,
			)
		test_dataloader = torch.utils.data.DataLoader(
			batch_size=cfg.dataset.test_pose.batch_size,
			dataset=test_dataset,
			shuffle=False,
			drop_last=False,
			num_workers=cfg.dataset.test_pose.num_workers)
	elif args.type == 'pose_mdm':
		# render novel poses, poses are from mdm
		from dataset.newpose import Dataset as NovelPoseDataset
		test_dataset = NovelPoseDataset(
			cfg.dataset.test_pose_mdm.dataset_path,
			cfg.dataset.test_pose_mdm.pose_path,
			format=cfg.dataset.test_pose_mdm.format)
		test_dataloader = torch.utils.data.DataLoader(
			batch_size=cfg.dataset.test_pose_mdm.batch_size,
			dataset=test_dataset,
			shuffle=False,
			drop_last=False,
			num_workers=cfg.dataset.test_pose_mdm.num_workers)
	elif args.type == 'video':
		# render training views for debugging
		from dataset.test import Dataset as VideoDataset
		test_dataset = VideoDataset(
			cfg.dataset.test_video.raw_dataset_path,
			cfg.dataset.test_video.dataset_path,
			test_type='video',
			skip=cfg.dataset.test_video.skip,  # to match monohuman
			exclude_training_view=False,
			bgcolor=cfg.bgcolor,
		)
		test_dataloader = torch.utils.data.DataLoader(
			batch_size=cfg.dataset.test_view.batch_size,
			dataset=test_dataset,
			shuffle=False,
			drop_last=False,
			num_workers=cfg.dataset.test_view.num_workers)
	elif args.type == 'train':
		# render training views for debugging
		from dataset.train import Dataset as TrainDataset
		test_dataset = TrainDataset(
			cfg.dataset.test_on_train.dataset_path,
			bgcolor=cfg.bgcolor,
			skip=1,
			target_size=cfg.model.img_size)
		test_dataloader = torch.utils.data.DataLoader(
			batch_size=cfg.dataset.test_on_train.batch_size,
			dataset=test_dataset,
			shuffle=False,
			drop_last=False,
			num_workers=cfg.dataset.test_on_train.num_workers)
	elif args.type == 'freeview':
		# render in 360 degree freeview
		from dataset.freeview import Dataset as FreeviewDataset
		test_dataset = FreeviewDataset(
			cfg.dataset.test_freeview.dataset_path,
			args.frame_idx,
			src_type=cfg.dataset.test_freeview.src_type,
			target_size=cfg.model.img_size,
			total_frames=args.n_frames,
		)
		test_dataloader = torch.utils.data.DataLoader(
			batch_size=cfg.dataset.test_freeview.batch_size,
			dataset=test_dataset,
			shuffle=False,
			drop_last=False,
			num_workers=cfg.dataset.test_freeview.num_workers)

	# load the model
	model = Model(cfg.model, test_dataset.get_canonical_info())
	if len(cfg.model.subdivide_iters) > 0:
		for _ in range(len(cfg.model.subdivide_iters)):
			model.subdivide(need_face_connectivity=False)

	# load checkpoints
	ckpt_dir = os.path.join(cfg.save_dir, 'checkpoints')
	if args.iter is None:
		max_iter = max([int(filename.split('_')[-1][:-3]) for filename in os.listdir(ckpt_dir) if 'pose' not in filename])
		ckpt_path = os.path.join(ckpt_dir, f'iter_{max_iter}.pt')
	else:
		ckpt_path = os.path.join(ckpt_dir, f'iter_{args.iter}.pt')
	logging.info(f'loading model from {ckpt_path}')
	ckpt = torch.load(ckpt_path)

	# # Load the checkpoint
	# checkpoint = torch.load(ckpt_path)
	# # Print the parameter names and their sizes
	# for name, param in checkpoint['network'].items():
	# 	print(f"Parameter: {name}, Size: {param.size()}")
	#
	# # Get the number of learnable parameters
	# total_params = sum(param.numel() for param in checkpoint['network'].values())
	# print(f"Total number of learnable parameters: {total_params}")


	model.load_state_dict(ckpt['network'], strict=False)

	model.cuda()
	model.eval()

	param_size = 0
	for param in model.parameters():
		param_size += param.nelement() * param.element_size()
	size_all_mb = param_size / 1024 ** 2
	logging.info('model size: {:.3f}MB'.format(size_all_mb))

	# for name, param in model.named_parameters():
	# 	print(f"Parameter: {name}, Size: {param.size()}")
	# 
	# 
	# total_params = sum(param.numel() for param in model.parameters())
	# print(f"Total number of learnable parameters: {total_params}")

	if args.type == 'pose' or args.type == 'video' or args.type == 'tpose' or args.type == 'pose_mdm':
		# disable pose refinement when poses are not in training
		model.pose_refinement_module = None

	# if args.type == 'view' and cfg.dataset.test_view.name == 'snapshot':
	# 	# follow anim-nerf and instantavatar's evaluation
	# 	evaluator = Evaluator_snapshot()
	# else:
	# 	evaluator = Evaluator()

	geometry_dict = {}
	for batch_idx, batch in enumerate(test_dataloader):
		data = cpu_data_to_gpu(
			batch, exclude_keys=EXCLUDE_KEYS_TO_GPU)

		with torch.no_grad():
			pred, mask, outputs = model(
				data['K'], data['E'],
				data['cnl_gtfms'], data['dst_Rs'], data['dst_Ts'], data['dst_posevec'])

		if cfg.random_bgcolor:
			bgcolor_tensor = torch.tensor(cfg.bgcolor).float()[None].to(pred.device) / 255.
			pred = unpack(pred, mask, bgcolor_tensor)

		if 'mesh' in args.type:
			# mesh = outputs['mesh']
			# frame_name = batch['frame_name'][0]
			# io3d().save_mesh(mesh, f'{save_dir}/{frame_name}.ply')

			verts = outputs['verts'].detach().cpu().numpy()
			faces = outputs['faces'].detach().cpu().numpy()
			face_colors = outputs['colors'].detach().cpu().numpy()

			# Convert to [0, 255] and add alpha channel for trimesh
			face_colors_rgba = (face_colors * 255).astype(np.uint8)
			face_colors_rgba = np.hstack([face_colors_rgba, np.full((faces.shape[0], 1), 255, dtype=np.uint8)])
	
			# Create trimesh mesh
			mesh = trimesh.Trimesh(vertices=verts, faces=faces, face_colors=face_colors_rgba)

			frame_name = batch['frame_name'][0]
			mesh.export(f'{save_dir}/{frame_name}.ply')
			print(f'{save_dir}/{frame_name}.ply')


			# pred_imgs = pred.detach().cpu().numpy()
			# mask_imgs = mask.detach().cpu().numpy()
			# normal_pred = F.normalize(outputs['normal'], dim=-1)
			# normal_mask = 1. - outputs['normal_mask']
			#
			# normal_map = normal_pred.detach().cpu().numpy()
			# normal_mask = normal_mask[..., None].detach().cpu().numpy()
			# normal_imgs = 255. - (normal_map - normal_mask + 1) * 0.5 * 255.
			# normal_imgs = (normal_imgs).astype(np.uint8)
			#
			# if args.type == 'view' or args.type == 'pose' or args.type == 'train':
			# 	truth_imgs = data['target_rgbs'].detach().cpu().numpy()
			#
			# for i, (frame_name, pred_img, mask_img, normal_img) in enumerate(
			# 		zip(batch['frame_name'], pred_imgs, mask_imgs, normal_imgs)):
			# 	pred_img = to_8b_image(pred_img)
			# 	print(os.path.join(save_dir, frame_name + '.png'))
			#
			# 	pred_imgs = []
			# 	normal_imgs = []
			# 	if args.type == 'view' or args.type == 'pose' or args.type == 'train':
			# 		truth_img = to_8b_image(truth_imgs[i])
			# 		evaluator.evaluate(pred_img / 255., truth_img / 255.)
			# 	pred_imgs.append(pred_img)
			# 	pred_imgs = np.concatenate(pred_imgs, axis=1)
			# 	Image.fromarray(pred_imgs).save(os.path.join(save_dir, frame_name + '.png'))

		else:
			pred_imgs = pred.detach().cpu().numpy()
			mask_imgs = mask.detach().cpu().numpy()
			normal_pred = F.normalize(outputs['normal'], dim=-1)
			normal_mask = 1. - outputs['normal_mask']

			normal_map = normal_pred.detach().cpu().numpy()
			normal_mask = normal_mask[..., None].detach().cpu().numpy()
			normal_imgs = 255. - (normal_map - normal_mask + 1) * 0.5 * 255. # for white bg
			# normal_imgs = 255. - (normal_map + 1) * 0.5 * 255. # for grey bg
			normal_imgs = (normal_imgs).astype(np.uint8)

			height, width = pred_imgs.shape[1:3]
			depth_normal, _ = depth_to_normal(data['K'][0], height, width, outputs['depth'][0])
			
			geometry_map = torch.cat([mask.permute(1, 2, 0), outputs['depth'][0], depth_normal, outputs['normal_mask'].permute(1, 2, 0), normal_pred[0]], dim=-1)
			geometry_dict[batch['frame_name'][0]] = geometry_map.detach().cpu().numpy()

			depth_normal_image = (depth_normal + 1.) / 2.
			depth_normal_image = depth_normal_image * mask.permute(1, 2, 0) + (1 - mask.permute(1, 2, 0))
			depth_normal_image = depth_normal_image.detach().cpu().numpy()
			depth_normal_image = (depth_normal_image * 255).astype(np.uint8)

			depth_map_image = apply_depth_colormap(outputs['depth'][0], mask[0][..., None], near_plane=None, far_plane=None)
			# depth_map = apply_depth_map(outputs['depth'][0], mask[0][..., None])
			depth_map_image = depth_map_image.detach().cpu().numpy()
			depth_map_image =  (depth_map_image * 255).astype(np.uint8)

			# if args.type == 'view' or args.type == 'pose' or args.type == 'train':
			# 	truth_imgs = data['target_rgbs'].detach().cpu().numpy()

			for i, (frame_name, pred_img, mask_img, normal_img) in enumerate(zip(batch['frame_name'], pred_imgs, mask_imgs, normal_imgs)):
				pred_img = to_8b_image(pred_img)
				print(os.path.join(save_dir, frame_name + '.png'))

				# if args.type == 'view' or args.type == 'pose' or args.type == 'train':
				# 	truth_img = to_8b_image(truth_imgs[i])
					# evaluator.evaluate(pred_img / 255., truth_img / 255.)

				save_imgs = np.concatenate([pred_img, depth_map_image, normal_img, depth_normal_image], axis=1)
				Image.fromarray(save_imgs).save(os.path.join(save_dir, frame_name + '.png'))

	if args.type == 'view' or args.type == 'pose':
		np.savez(os.path.join(cfg.save_dir, 'eval', f'geometry_novel_{args.type}.npz'), **geometry_dict)

	# evaluator.summarize(os.path.join(cfg.save_dir, 'eval', f'metric_{args.type}.npy'))


if __name__ == "__main__":
	args = parse_args()
	main(args)
