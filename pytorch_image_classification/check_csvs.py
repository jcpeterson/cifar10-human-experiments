import argparse
from os import listdir
from os.path import join, isdir
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str,
                    default='../../../model_results/mixup_250_run1')
args = parser.parse_args()

out_path = getattr(args, 'path')

def list_folders_only(path):
    return [x for x in listdir(path) if isdir(join(path, x))]

# assume true and try to prove false
epochs_consistent = True

model_folders = sorted(list_folders_only(out_path))
for model_folder in model_folders:
    param_folders = list_folders_only(join(out_path, model_folder))
    param_folders = sorted(param_folders)

    print(model_folder)
    print('(contains', len(param_folders), 'folders)')
    
    for param_folder in param_folders:
        df = pd.read_csv(join(out_path, 
                              model_folder, 
                              param_folder, 
                              'scores.csv'))
        try:
            if df.epoch.max() != last_epoch:
                epochs_consistent = False
            else:
                last_epoch = df.epoch.max() 
        except:
            last_epoch = df.epoch.max()
        print('   ', param_folder, 'EPOCHS DONE:', df.epoch.max())

if not epochs_consistent: 
    print("WARNING: NOT ALL MODELS HAVE SAME # OF EPOCHS!!!")
else:
    print("SUCCESS: ALL MODELS HAVE SAME # OF EPOCHS!")