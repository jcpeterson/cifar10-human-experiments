import os
import shutil

dir = os.getcwd()
print(dir)

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

subdirs = get_immediate_subdirectories(dir)
print(subdirs)

for model in subdirs:
    if model != 'miniconda3':    
        subsubdirs = get_immediate_subdirectories(model)
        for ssd in subsubdirs:
            print(ssd)
            shutil.copy(os.path.join(dir, model, ssd, 'scores.csv'),
                      os.path.join(dir, 'csvs', model + '_' + ssd + '_scores.csv'))
