import os
import shutil
import glob
data_seqs = {'mvhuman': ['100846', '100990', '102107', '102145', '103708', '204112', '204129', '200173'],
            'actorhq': ['actor0101', 'actor0301', 'actor0601', 'actor0701', 'actor0801'],
             'mpi': ['0056', 'FranziRed', 'Antonia', 'Magdalena']
             }

dir_source = '/ubc/cs/home/c/chunjins/chunjin_scratch1/project/project/humannerf/gomavatar/log/'
dir_target = '/ubc/cs/home/c/chunjins/chunjin_shield/project/gomavatar/log/'
for key in data_seqs.keys():
    seqs = data_seqs[key]
    for seq in seqs:
        if key == 'mvhuman':
            seq = 'mvhuman_'+seq
        path_source = os.path.join(dir_source, key, seq, 'checkpoints')
        path_file = glob.glob(path_source + '/*.pt')[0]

        os.makedirs(os.path.join(dir_target, key, seq, 'checkpoints'), exist_ok=True)
        path_target = path_file.replace(dir_source, dir_target)
        shutil.copy(path_file, path_target)