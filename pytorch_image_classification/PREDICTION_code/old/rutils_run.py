import os
import torch

def save_checkpoint_epoch(state, epoch, outdir):
    model_path = os.path.join(outdir, 'model_state_' + str(epoch) + '.pth')
    torch.save(state, model_path)

