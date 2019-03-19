import os, sys, shutil
import numpy as np
def copy_optimal_dir(params, in_dir = '/tigress/ruairidh/model_results/9k_tuning_run_1', 
out_dir = '/tigress/ruairidh/model_results/optimal_9k', num_folds = 10):
    """Takes the params of the optimal model and copys the source 
    dir to a nice format in the optimal folder.
    Params: model, control, lr, seed"""

    model, control, lr, seed = params
    for fold in np.arange(num_folds):
        identifier = 'con_{0}_lr_{1}_seed_{2}_fold_{3}'.format(control, lr, seed, fold)
        in_path = os.path.join(in_dir, model, identifier)
        print(in_path)
        out_path = os.path.join(out_dir, control, model, 'fold_{0}'.format(fold))
        print(out_path)
        params_path = os.path.join(out_dir, control, model, '{0}_{1}'.format(model, identifier))
        print(params_path)
        shutil.copytree(in_path, out_path)
        open('{0}.txt'.format(params_path), 'a').close() 
    
copy_optimal_dir(['resnet_basic_110', 'False', '0.1', '0'])
