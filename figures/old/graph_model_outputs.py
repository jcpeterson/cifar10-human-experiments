import os
import shutil
import os, sys, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv
import matplotlib.gridspec as gridspec
import itertools


dir = os.getcwd()
print(dir)

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if (os.path.isdir(os.path.join(a_dir, name))) and name != 'miniconda3']


def lr_dict(model, lr, seed):
    for val in [True, False]:
        master[str(val)] = {}
        total = []
        identifier = 'con_'+ str(val) + '_lr_' + str(lr) +'_seed_' + str(seed)
        print('identifier: ', identifier)
#        with open(model + '/' + identifier + '/' + identifier + '_scores.csv', newline='') as csvfile:
        with open(model + '/' + identifier + '/scores.csv', newline='') as csvfile:
            reader = csv.reader(csvfile)
            keys = next(reader)
    
            for row in reader:
                total.append(np.float_(row))
    
        print(keys)
        total = np.array(total)
        mins = np.array([np.min(x) for x in total.T])
    
        print(keys, mins)
    
        acc_dict = {}
        loss_dict = {}
        for key in keys:
    
            if 'acc' in key:
    #            print(key)
                acc_dict[key] = total[:, keys.index(key)]  
            elif 'loss' in key:
    #            print(key)
                loss_dict[key] = total[:, keys.index(key)]
    
        master[str(val)]['loss'] = loss_dict
        master[str(val)]['acc'] = acc_dict
        
    return master
    


subdirs = get_immediate_subdirectories(dir)
print('models: ', subdirs)

subdirs.remove('csvs')

print('subdirs / models: ', subdirs)

for model in subdirs:
    raw_filenames = os.listdir(os.path.join(dir, model))
    new_filenames = [x for x in raw_filenames if 'csv' in x]
    print('filenames: \n', new_filenames)

    master = {}
        
    for lr in [0.1, 0.01, 0.001]:
        for seed in [0, 1, 2]:
            master = lr_dict(model, lr, seed)
             
            ### do ordered keys!!!


            acc_keys = ['c10_50k_acc', 'c10h_train_acc',  'c10h_train_c10_acc', 'c10h_val_acc', 'c10h_val_c10_acc', 'v4_acc', 'v6_acc']
            loss_keys = ['c10_50k_loss', 'c10h_train_loss', 'c10h_train_c10_loss', 'c10h_val_loss', 'c10h_val_c10_loss', 'v4_loss', 'v6_loss']
            my_labels = ['CIFAR10 training 50k', 'CIFAR10H training 9k', 'CIFAR10 tuning 9k', 'CIFAR10H validation 1k', 'CIFAR10 tuning 1k', 'CIFAR10.1 v4', 'CIFAR10.1 v6']
            acc_keys.reverse()
            loss_keys.reverse()
            my_labels.reverse()
            print(acc_keys)    
            print(loss_keys)    
            print(my_labels)    

            fig = plt.figure(figsize=(15, 10))
            suptitle = fig.suptitle(model + ': comparision at ' + str(lr) + '_seed_' + str(seed), fontsize=16)
            real_dict = master['False']
            control_dict = master['True']
    
            gs = gridspec.GridSpec(2, 2)
    
            ax_ul = plt.subplot(gs[0, 0])
            ax_ul.set_title('Actual')
            for i, k in enumerate(acc_keys):
                v = real_dict['acc'][k]
                ax_ul.plot(v,  label = my_labels[i])
            ax_ul.set_ylabel('Accuracy')
            #ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    
            ax_ll = plt.subplot(gs[1, 0])
            for i, k in enumerate(loss_keys):
                v = real_dict['loss'][k]
                ax_ll.plot(v, label = my_labels[i])
            ax_ll.set_ylabel('Loss')
    
            #ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    
            ax_ur = plt.subplot(gs[0, 1], sharex=ax_ul)
            ax_ur.set_title('Control')

            
            for i, k in enumerate(acc_keys):
                v = control_dict['acc'][k]
                ax_ur.plot(v,  label = my_labels[i])
            ax_ur.set_yticks([])
            ax_ur.set_xticks([])
            lgd = ax_ur.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    
            ax_lr = plt.subplot(gs[1, 1], sharex=ax_ll)

            for i, k in enumerate(loss_keys):
                v = control_dict['loss'][k]
                ax_lr.plot(v,  label = my_labels[i])
            ax_lr.set_yticks([])
            #ax_lr.set_ylim([0, 1])
            #ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., labels=my_labels)

            save_path = os.path.join(dir, model, ('{0}._comparison_fig_lr_' + str(lr) + '_seed_' + str(seed) + '.png').format(model))

            plt.savefig(save_path,
                        bbox_extra_artists=(lgd, suptitle,), bbox_inches='tight')
            
            copypath = os.path.join('/home/ruairidh/superman/cifar10-human-experiments/figures', model)          

            if not os.path.exists(copypath):
                os.makedirs(copypath)

            shutil.copy(save_path,
            os.path.join(copypath, model + '_comparison_fig_lr_' + str(lr) + '_seed_' + str(seed) + '.png'))
            
