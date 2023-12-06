import os

import torch
import numpy as np
from torch.nn import DataParallel
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
sys.path.append("./")
# from dataset.name2dataset import name2dataset

from network.loss import name2loss
from network.renderer import name2network
from network.metrics import name2metrics
from train.train_tools import to_cuda, Logger, reset_learning_rate, MultiGPUWrapper, DummyLoss
from train.train_valid import ValidationEvaluator
from utils.dataset_utils import simple_collate_fn, dummy_collate_fn
from seed import seed_everything
# import numpy as np
# import random

# def setup_seed(seed):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.backends.cudnn.deterministic = True
#     # cudnn setting
#     torch.backends.cudnn.enabled = True
#     torch.backends.cudnn.benchmark = True

class Trainer:
    
    worker_num=0
    
    default_cfg={
        "optimizer_type": 'adam',
        "multi_gpus": False,
        "lr_type": "exp_decay",
        "lr_cfg":{
            "lr_init": 1.0e-4,
            "decay_step": 100000,
            "decay_rate": 0.5,
        },
        "total_step": 300000,
        "train_log_step": 20,
        "val_interval": 10000,
        "save_interval": 500,
        "worker_num": worker_num, 

        "MAGNET_mvs_weighting": "CW5",
        "wo_hdh": False,
        "change_input": False,
        "revise_range": False,
        "handle_distort": False,
        "handle_distort_all": False,
        "handle_distort_input_all": False,
        "use_polar_weighted_loss": False,
        "eval_only": False,
        "render_uncert": False,
        "uncert_tune": False,
        "use_disp": True,
        "with_sin": False,
        "wo_mono_feat": False,
        "mono_uncert_tune": False,
        "fix_all": False,
        "fix_coarse": False,
        
    }

    # def _init_dataset(self):
    #     if self.cfg["train_dataset_type"] == "gen":
    #         mode="train"
    #         if self.cfg["debug"]:
    #             mode="test"
    #         from data_readers.habitat_data_neuray import HabitatImageGenerator
    #         train_set = HabitatImageGenerator(
    #             args=self.cfg,
    #             split=mode,
    #             seq_len=self.cfg["seq_len"],
    #             reference_idx=self.cfg["reference_idx"],
    #             full_width=self.cfg["width"],
    #             full_height=self.cfg["height"],
    #             m3d_dist=self.cfg["m3d_dist"]
    #         )

    #         self.train_set =DataLoader(train_set,1,True,num_workers=self.cfg['worker_num'],collate_fn=dummy_collate_fn)                        
    #         mode="val"
    #         if self.cfg["debug"]:
    #             mode="test"
    #         val_set = HabitatImageGenerator(
    #             args=self.cfg,
    #             split=mode,
    #             seq_len=self.cfg["seq_len"],
    #             reference_idx=self.cfg["reference_idx"],
    #             full_width=self.cfg["width"],
    #             full_height=self.cfg["height"],
    #             m3d_dist=self.cfg["m3d_dist"]
    #         )
    #         self.val_set = DataLoader(val_set,1,False,num_workers=self.cfg['worker_num'],collate_fn=dummy_collate_fn)
    #     else:
    #         from data_readers.habitat_data_neuray_ft import FinetuningRendererDataset
    #         train_set = FinetuningRendererDataset(self.cfg, is_train=True)
    #         self.train_set =DataLoader(train_set,1,True,num_workers=self.cfg['worker_num'],collate_fn=dummy_collate_fn)
    #         val_set = FinetuningRendererDataset(self.cfg, is_train=False)            
    #         self.val_set = DataLoader(val_set,1,False,num_workers=self.cfg['worker_num'],collate_fn=dummy_collate_fn)


    #     # self.train_set=name2dataset[self.cfg['train_dataset_type']](self.cfg['train_dataset_cfg'], True)       
    #     # print("self.cfg['worker_num']:", self.cfg['worker_num'])
    #     # self.train_set=DataLoader(self.train_set,1,True,num_workers=self.cfg['worker_num'],collate_fn=dummy_collate_fn)
    #     # print(f'train set len {len(self.train_set)}')
    #     # self.val_set_list, self.val_set_names = [], []
    #     # for val_set_cfg in self.cfg['val_set_list']:
    #     #     name, val_type, val_cfg = val_set_cfg['name'], val_set_cfg['type'], val_set_cfg['cfg']
    #     #     val_set = name2dataset[val_type](val_cfg, False)
    #     #     val_set = DataLoader(val_set,1, False, num_workers=self.cfg['worker_num'],collate_fn=dummy_collate_fn)
    #     #     self.val_set_list.append(val_set)
    #     #     self.val_set_names.append(name)
    #     #     print(f'{name} val set len {len(val_set)}')

    def _init_dataset(self):
        if self.cfg["train_dataset_type"] == "gen":
            mode="train"
            if self.cfg["debug"]:
                mode="test"
            if self.cfg["use_lmdb"]:
                from data_readers.habitat_data_neuray_lmdb import HabitatImageGenerator_LMDB
                train_set = HabitatImageGenerator_LMDB(
                    args=self.cfg,
                    split=mode,
                    seq_len=self.cfg["seq_len"],
                    reference_idx=self.cfg["reference_idx"],
                    full_width=self.cfg["width"],
                    full_height=self.cfg["height"],
                    m3d_dist=self.cfg["m3d_dist"]
                )
            else:
                from data_readers.habitat_data_neuray import HabitatImageGenerator            
                train_set = HabitatImageGenerator(
                    args=self.cfg,
                    split=mode,
                    seq_len=self.cfg["seq_len"],
                    reference_idx=self.cfg["reference_idx"],
                    full_width=self.cfg["width"],
                    full_height=self.cfg["height"],
                    m3d_dist=self.cfg["m3d_dist"]
                )
            self.train_data = train_set
            self.train_set =DataLoader(train_set,1,shuffle=True,num_workers=self.cfg['worker_num'],collate_fn=dummy_collate_fn)
        else:
            from data_readers.habitat_data_neuray_ft import FinetuningRendererDataset
            train_set = FinetuningRendererDataset(self.cfg, is_train=True)
            self.train_set =DataLoader(train_set,1, shuffle=True,num_workers=self.cfg['worker_num'],collate_fn=dummy_collate_fn, pin_memory=True)
            # val_set = FinetuningRendererDataset(self.cfg, is_train=False)            
            # self.val_set = DataLoader(val_set,1,False,num_workers=self.cfg['worker_num'],collate_fn=dummy_collate_fn, pin_memory=True, prefetch_factor=1)


        # self.train_set=name2dataset[self.cfg['train_dataset_type']](self.cfg['train_dataset_cfg'], True)       
        # print("self.cfg['worker_num']:", self.cfg['worker_num'])
        # self.train_set=DataLoader(self.train_set,1,True,num_workers=self.cfg['worker_num'],collate_fn=dummy_collate_fn)
        # print(f'train set len {len(self.train_set)}')
        # self.val_set_list, self.val_set_names = [], []
        # for val_set_cfg in self.cfg['val_set_list']:
        #     name, val_type, val_cfg = val_set_cfg['name'], val_set_cfg['type'], val_set_cfg['cfg']
        #     val_set = name2dataset[val_type](val_cfg, False)
        #     val_set = DataLoader(val_set,1, False, num_workers=self.cfg['worker_num'],collate_fn=dummy_collate_fn)
        #     self.val_set_list.append(val_set)
        #     self.val_set_names.append(name)
        #     print(f'{name} val set len {len(val_set)}')


    def _init_val_dataset(self):
        if self.cfg["train_dataset_type"] == "gen":
            if self.cfg["use_lmdb"]:
                from data_readers.habitat_data_neuray_lmdb import HabitatImageGenerator_LMDB                
                mode="val" #"val"
                if self.cfg["debug"]:
                    mode="test"

                val_set = HabitatImageGenerator_LMDB(
                    args=self.cfg,
                    split=mode,
                    seq_len=self.cfg["seq_len"],
                    reference_idx=self.cfg["reference_idx"],
                    full_width=self.cfg["width"],
                    full_height=self.cfg["height"],
                    m3d_dist=self.cfg["m3d_dist"]
                )
            else:
                from data_readers.habitat_data_neuray import HabitatImageGenerator
                mode="val"
                if self.cfg["debug"]:
                    mode="test"
                val_set = HabitatImageGenerator(
                    args=self.cfg,
                    split=mode,
                    seq_len=self.cfg["seq_len"],
                    reference_idx=self.cfg["reference_idx"],
                    full_width=self.cfg["width"],
                    full_height=self.cfg["height"],
                    m3d_dist=self.cfg["m3d_dist"]
                )
            self.val_data = val_set
            self.val_set = DataLoader(val_set, 1, False, num_workers=0, collate_fn=dummy_collate_fn, pin_memory=True)
        else:
            from data_readers.habitat_data_neuray_ft import FinetuningRendererDataset
            val_set = FinetuningRendererDataset(self.cfg, is_train=False)            
            self.val_set = DataLoader(val_set, 1, False, num_workers=0, collate_fn=dummy_collate_fn, pin_memory=True)


    def _init_network(self):
        # only work when training-fine-tune renderer and depth_guided_sampling=true, and training time.
        # test time: normal ?

        self.network=name2network[self.cfg['network']](self.cfg).cuda()

        # loss
        self.val_losses = []
        for loss_name in self.cfg['loss']:
            self.val_losses.append(name2loss[loss_name](self.cfg))
        self.val_metrics = []

        # metrics
        for metric_name in self.cfg['val_metric']:
            if metric_name in name2metrics:
                self.val_metrics.append(name2metrics[metric_name](self.cfg))
            else:
                self.val_metrics.append(name2loss[metric_name](self.cfg))

        # we do not support multi gpu training for NeuRay
        if self.cfg['multi_gpus']:
            raise NotImplementedError
            # make multi gpu network
            # self.train_network=DataParallel(MultiGPUWrapper(self.network,self.val_losses))
            # self.train_losses=[DummyLoss(self.val_losses)]
        else:
            self.train_network=self.network
            self.train_losses=self.val_losses

        if self.cfg['optimizer_type']=='adam':
            self.optimizer = Adam
        elif self.cfg['optimizer_type']=='sgd':
            self.optimizer = SGD
        else:
            raise NotImplementedError

        self.val_evaluator=ValidationEvaluator(self.cfg)
        
        if "lr_diff" in self.cfg and self.cfg["lr_diff"]:
            # pass
            from train.ft_lr_common_manager import name2lr_manager
            self.lr_manager=name2lr_manager[self.cfg['lr_type']](self.cfg['lr_cfg'])
            self.optimizer=self.lr_manager.construct_optimizer(self.optimizer,self.network)
        else:
            from train.lr_common_manager import name2lr_manager
            self.lr_manager=name2lr_manager[self.cfg['lr_type']](self.cfg['lr_cfg'])        
            self.optimizer=self.lr_manager.construct_optimizer(self.optimizer,self.network)


    def __init__(self,cfg, data_idx):
        if cfg["train_dataset_type"] == "gen":
            pass
        else:
            cfg["data_idx"] = data_idx
            cfg["name"] = cfg["name"]+"_id_"+str(data_idx)

        self.cfg={**self.default_cfg,**cfg}

        # import ipdb;ipdb.set_trace()


        # self.cfg["name"]
        seed=2022
        # setup_seed(seed)
        seed_everything(seed)
        self.model_name=cfg['name']
        self.model_dir=os.path.join('data/model', cfg['name'])
        if not os.path.exists(self.model_dir): os.makedirs(self.model_dir, exist_ok=True)
        self.pth_fn=os.path.join(self.model_dir,'model.pth')
        self.best_pth_fn=os.path.join(self.model_dir,'model_best.pth')

    def run(self):
        print("init...")

        self._init_dataset()
        self._init_network()
        self._init_logger()
        print("init end...")

        best_para,start_step=self._load_model()
        train_iter=iter(self.train_set)

        pbar=tqdm(total=self.cfg['total_step'],bar_format='{r_bar}')
        pbar.update(start_step)
        print("training ...")

        if self.cfg["eval_only"]:
            step = start_step
            # if (step+1)%self.cfg['val_interval']==0 or (step+1)==self.cfg['total_step']:
            torch.cuda.empty_cache()
            self._init_val_dataset()#re-init validation data

            val_results={}
            val_para = 0
            # for vi, val_set in enumerate(self.val_set_list):
            val_results_cur, val_para_cur = self.val_evaluator(
                self.network, self.val_metrics, self.val_set, step,
                self.model_name, val_set_name="m3d")
            if self.cfg["use_lmdb"] and self.cfg["train_dataset_type"] == "gen":
                self.val_data.env.close()

            for k,v in val_results_cur.items():
                val_results[f'm3d-{k}'] = v
                # always use the final val set to select model!
            val_para = val_para_cur

            # if val_para>best_para:
            #     print(f'New best model {self.cfg["key_metric_name"]}: {val_para:.5f} previous {best_para:.5f}')
            #     best_para=val_para
            #     self._save_model(step+1,best_para,self.best_pth_fn)


            self._log_data(val_results,step+1,'val')
            del val_results, val_para, val_para_cur, val_results_cur
            exit()


        for step in range(start_step,self.cfg['total_step']):
            if self.cfg["debug"]:
                torch.cuda.empty_cache()
                val_results={}
                val_para = 0
                # for vi, val_set in enumerate(self.val_set_list):
                val_results_cur, val_para_cur = self.val_evaluator(
                    self.network, self.val_losses + self.val_metrics, self.val_set, step,
                    self.model_name, val_set_name="m3d")
                for k,v in val_results_cur.items():
                    val_results[f'm3d-{k}'] = v
                    # always use the final val set to select model!
                val_para = val_para_cur

                if val_para>best_para:
                    print(f'New best model {self.cfg["key_metric_name"]}: {val_para:.5f} previous {best_para:.5f}')
                    best_para=val_para
                    self._save_model(step+1,best_para,self.best_pth_fn)

                self._log_data(val_results,step+1,'val')
                del val_results, val_para, val_para_cur, val_results_cur

            print("step:", step)
            # import pdb;pdb.set_trace()
            try:

                train_data = next(train_iter)
            except StopIteration:
                # self.train_set.dataset.reset()
                train_iter = iter(self.train_set)
                train_data = next(train_iter)
            if not self.cfg['multi_gpus']:
                train_data = to_cuda(train_data)
            train_data['step']=step

            self.train_network.train()
            self.network.train()
            lr = self.lr_manager(self.optimizer, step)



            self.optimizer.zero_grad()
            self.train_network.zero_grad()

            log_info={}
            
            outputs=self.train_network(train_data, is_perspec="render_cubes" in self.cfg and self.cfg["render_cubes"])
            for loss in self.train_losses:
                loss_results = loss(outputs,train_data,step)
                for k,v in loss_results.items():
                    log_info[k]=v

            loss=0
            # import ipdb;ipdb.set_trace()
            for k,v in log_info.items():
                # if k.startswith('loss'):
                #     loss=loss+torch.mean(v)
                if 'loss' in k:
                    loss=loss+torch.mean(v)

        #    import ipdb;ipdb.set_trace()
            if self.cfg["fix_all"]:
                pass
            else:
                loss.backward()
            self.optimizer.step()
            if ((step+1) % self.cfg['train_log_step']) == 0:
                self._log_data(log_info,step+1,'train')

            if (step+1)%self.cfg['save_interval']==0:
                self._save_model(step+1,best_para)


            if (step+1)%self.cfg['val_interval']==0 or (step+1)==self.cfg['total_step']:
                torch.cuda.empty_cache()
                self._init_val_dataset()#re-init validation data

                val_results={}
                val_para = 0
                # for vi, val_set in enumerate(self.val_set_list):
                val_results_cur, val_para_cur = self.val_evaluator(
                    self.network, self.val_metrics, self.val_set, step,
                    self.model_name, val_set_name="m3d")
                if self.cfg["use_lmdb"] and self.cfg["train_dataset_type"] == "gen":
                    self.val_data.env.close()

                for k,v in val_results_cur.items():
                    val_results[f'm3d-{k}'] = v
                    # always use the final val set to select model!
                val_para = val_para_cur

                # if val_para>best_para:
                #     print(f'New best model {self.cfg["key_metric_name"]}: {val_para:.5f} previous {best_para:.5f}')
                #     best_para=val_para
                #     self._save_model(step+1,best_para,self.best_pth_fn)


                self._log_data(val_results,step+1,'val')
                del val_results, val_para, val_para_cur, val_results_cur
            
            pbar.set_postfix(loss=float(loss.detach().cpu().numpy()),lr=lr)
            pbar.update(1)
            del loss, log_info

        pbar.close()
        if self.cfg["use_lmdb"] and self.cfg["train_dataset_type"] == "gen":
            self.train_data.env.close()

    def _load_model(self):
        best_para,start_step=0,0
        # import ipdb;ipdb.set_trace()
        # print("self.pth_fn")
        if os.path.exists(self.pth_fn):
            checkpoint=torch.load(self.pth_fn)
            best_para = checkpoint['best_para']
            start_step = checkpoint['step']
            self.network.load_state_dict(checkpoint['network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f'==> resuming from step {start_step} best para {best_para}')

        return best_para, start_step

    def _save_model(self, step, best_para, save_fn=None):
        save_fn = self.pth_fn if save_fn is None else save_fn
        torch.save({
            'step':step,
            'best_para':best_para,
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        },save_fn)

    def _init_logger(self):
        self.logger = Logger(self.model_dir)

    def _log_data(self,results,step,prefix='train',verbose=False):
        log_results={}
        for k, v in results.items():
            if isinstance(v,float) or np.isscalar(v):
                log_results[k] = v
            elif type(v)==np.ndarray:
                log_results[k]=np.mean(v)
            else:
                log_results[k]=np.mean(v.detach().cpu().numpy())
        self.logger.log(log_results,prefix,step,verbose)




