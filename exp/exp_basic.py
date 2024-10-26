import os
import torch
import numpy as np
import wandb

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _set_wandb(self, setting):
        if self.args.run_name != None:
            run_name = self.args.run_name
        elif self.args.run_name == None:
            run_name = setting
            
        wandb.init(
            project='pv-forecasting',
            entity='pv-forecasting',
            name=f'{run_name}',
            config=self.args)
        

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
