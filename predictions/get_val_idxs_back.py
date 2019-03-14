import os, sys
import numpy as np

script, seed = sys.argv
seed = int(seed)
perc = 0.1
data_len = 10000
c10h_rnd_idxs = np.arange(data_len)

# this seed is the same for self.set in ['train','test']
np.random.seed(seed)
np.random.shuffle(c10h_rnd_idxs)
split_idx = int((1 - perc)*data_len)
final = c10h_rnd_idxs[split_idx:]
print(final[:5])


