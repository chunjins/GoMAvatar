import os
import sys
sys.path.append('../../')
from shutil import copyfile
import yaml
import numpy as np
from tqdm import tqdm
from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string('cfg',
                    '394.yaml',
                    'the path of config file')

MODEL_DIR = '../../utils/smpl/models'


def parse_config():
    config = None
    with open(FLAGS.cfg, 'r') as file:
        config = yaml.full_load(file)

    return config


def prepare_dir(out_dir, name=None):
    if name is not None:
        out_dir = os.path.join(out_dir, name)
    os.makedirs(out_dir, exist_ok=True)

    return out_dir


def main(argv):
    del argv  # Unused.

    cfg = parse_config()
    subject = cfg['dataset']['subject']
    max_frames = cfg['max_frames']

    dataset_dir = cfg['dataset']['zju_mocap_path']
    subject_dir = os.path.join(dataset_dir, f"CoreView_{subject}")

    anno_path = os.path.join(subject_dir, 'annots.npy')
    annots = np.load(anno_path, allow_pickle=True).item()

    # load image paths
    img_path_frames_views = annots['ims']
    img_paths = np.array([
        np.array(multi_view_paths['ims'])[:] \
        for multi_view_paths in img_path_frames_views
    ])
    if max_frames > 0:
        img_paths = img_paths[:max_frames]

    skip = 30
    img_paths_nv = img_paths[:-(len(img_paths) // 5)][::skip]
    img_paths_np = img_paths[-(len(img_paths) // 5):][::skip]
    img_paths = np.concatenate([img_paths_nv, img_paths_np], axis=0)

    output_path = os.path.join(cfg['output']['dir'], f"CoreView_{subject}")

    os.makedirs(output_path, exist_ok=True)
    out_img_dir = prepare_dir(output_path)
    out_msk1_dir = prepare_dir(output_path, 'mask')
    out_msk2_dir = prepare_dir(output_path, 'mask_cihp')

    ipaths = img_paths[0]
    for ipath in ipaths:
        cam_dir = ipath.split('/')[0]
        prepare_dir(out_img_dir, cam_dir)
        prepare_dir(out_msk1_dir, cam_dir)
        prepare_dir(out_msk2_dir, cam_dir)


    copyfile(anno_path, os.path.join(output_path, 'annots.npy'))

    for idx, ipaths in enumerate(tqdm(img_paths)):
        for ipath in ipaths:
            img_path = os.path.join(subject_dir, ipath)
            out_img_path = os.path.join(out_img_dir, ipath)

            ipath = ipath.replace('jpg', 'png')
            msk1_path = os.path.join(subject_dir, 'mask', ipath)
            msk2_path = os.path.join(subject_dir, 'mask_cihp', ipath)
            out_msk1_path = os.path.join(out_msk1_dir, ipath)
            out_msk2_path = os.path.join(out_msk2_dir, ipath)

            copyfile(img_path, out_img_path)
            copyfile(msk1_path, out_msk1_path)
            copyfile(msk2_path, out_msk2_path)


if __name__ == '__main__':
    app.run(main)
