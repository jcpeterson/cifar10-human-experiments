import os

for fn in os.listdir(os.getcwd()):
    if fn[-4:] == '.out':# and 'mixup' in fn:
        print(fn)
        with open(fn, 'r') as f:
            txt = f.readlines()
        print(txt[-5:])