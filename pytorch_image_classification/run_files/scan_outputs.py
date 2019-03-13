import os
import sys
import numpy as np
#os.chdir(os.path.dirname(__file__))
print(os.getcwd())

script_name, model_form = sys.argv
print('Model form: ', model_form)

files = [f for f in os.listdir('.') if os.path.isfile(f)]

for file in files:
    if model_form in file:
        print(file)
        
        with open(file) as f:
            data = f.readlines()

        for ri in np.arange(10)[::-1]:
            print(data[-ri])
#            print('\n')
