import glob, os
import shutil
import pandas as pd
import numpy as np

def parse_file(file_str):
    with open(file_str) as file:
        file_contents = file.read()
    good_stuff = file_contents.split('Test accuracy of pretrained model')[-1]
    file_contents = None
    loss = good_stuff.split('Loss')[1].split('Accuracy')[0]
    accuracy = good_stuff.split('Accuracy')[1].split('\n')[0]
    run = good_stuff.split('run:')[1].split('\n')[0]
    good_stuff = good_stuff.split('Loss')[1]
    print('Loss:', good_stuff.split('Accuracy')[0])
    print('training_run:', good_stuff.split('training_run')[1])
    return loss, accuracy, run

# just add rows to dataframe!
df_list = []
models = []
print('lists created')

for file in glob.glob("*.out"):
    print('file: ', file)
    model = file.split('.')[0]
    models.append(model)

models = list(set(models))
print('final models: ', models)

for model in models:
    for run in np.arange(2,5):
        file = ('{0}.pretrained_accuracy.run_{1}.out'.format(model, run))
        print(file)
        l, a, r = parse_file(file)

        temp_df = pd.DataFrame({'Model':[model], 'Loss': [np.float(l)], 
'Accuracy': [np.float(a)], 'Error': [1-np.float(a)], 'Run': [np.int(r)]})
        df_list.append(temp_df)
   
master_df = pd.concat(df_list, ignore_index=True)
print(master_df)

master_df.to_csv('readout.csv')

mmodels = []
means = []
for model in models:
    print(model)
    model_mean = master_df[['Accuracy']].loc[master_df['Model'] == model].mean(axis=0)
    mmodels.append([model])
    means.append([model_mean])

mean_df = pd.DataFrame({'Model': mmodels, 'Mean': means})
print(mean_df)
mean_df.to_csv('means_readout.csv')

errors=[]
for model in models:
    print(model)
    model_errors = master_df[['Error']].loc[master_df['Model'] == model].mean(axis=0)
    errors.append([model_errors])

errors_df = pd.DataFrame({'Model': mmodels, 'Errors': errors})
print(errors_df)
errors_df.to_csv('errors_readout.csv')

runs=[]
for model in models:
    print(model)
    max_acc = master_df[['Accuracy']].loc[master_df['Model'] == model].max(axis=0)
    print('max_acc: ', max_acc)
    model_runs = master_df[['Run']].loc[(master_df['Model'] == model) & (master_df['Accuracy'] == max_acc)]
    runs.append([model_runs])

runs_df = pd.DataFrame({'Model': mmodels, 'Opt run': runs})
print(runs_df)
runs_df.to_csv('opt_runs_readout.csv')

# copy folders
for model, run in list(zip(mmodels, runs)):
    print(model, run)
    
