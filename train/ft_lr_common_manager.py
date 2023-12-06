# import abc

# class LearningRateManager(abc.ABC):
    # @staticmethod
    # def set_lr_for_all(optimizer, lr):
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = lr



    # @abc.abstractmethod
    # def __call__(self, optimizer, step, *args, **kwargs):
    #     pass
# LearningRateManager
class ExpDecayLR():
    def __init__(self,cfg):
        self.lr_init=cfg['lr_init']
        self.density_lr_init = cfg["density_lr_init"]
        self.decay_step=cfg['decay_step']
        self.decay_rate=cfg['decay_rate']
        self.lr_min=1e-5

    def construct_optimizer(self, optimizer, network):
        # may specify different lr for different parts
        # use group to set learning rate
        # import ipdb;ipdb.set_trace()
        paras = network.parameters()
        # network.agg_net.agg_impl.rgb_fc: lr_init
        # network.agg_net.agg_impl.: density_lr_init
        # base_fc, vis_fc, vis_fc2, ray_attention, out_geometry_fc, neuray_fc, geometry_fc
        # rest: lr_init 
        # 

        density_params = []
        # density_params += list(map(id, network.agg_net.agg_impl.base_fc.parameters()))        
        # density_params += list(map(id, network.agg_net.agg_impl.vis_fc.parameters()))        
        # density_params += list(map(id, network.agg_net.agg_impl.vis_fc2.parameters()))        
        # density_params += list(map(id, network.agg_net.agg_impl.ray_attention.parameters()))        
        # density_params += list(map(id, network.agg_net.agg_impl.out_geometry_fc.parameters()))        
        # density_params += list(map(id, network.agg_net.agg_impl.neuray_fc.parameters()))        
        density_params += list(map(id, network.agg_net.agg_impl.geometry_fc.parameters()))        
        # density_params += list(map(id, network.agg_net.prob_embed.parameters()))        

        # fine:
        # density_params += list(map(id, network.fine_agg_net.agg_impl.base_fc.parameters()))        
        # density_params += list(map(id, network.fine_agg_net.agg_impl.vis_fc.parameters()))        
        # density_params += list(map(id, network.fine_agg_net.agg_impl.vis_fc2.parameters()))        
        # density_params += list(map(id, network.fine_agg_net.agg_impl.ray_attention.parameters()))        
        # density_params += list(map(id, network.fine_agg_net.agg_impl.out_geometry_fc.parameters()))        
        # density_params += list(map(id, network.fine_agg_net.agg_impl.neuray_fc.parameters()))        
        density_params += list(map(id, network.fine_agg_net.agg_impl.geometry_fc.parameters()))        
        # density_params += list(map(id, network.fine_agg_net.prob_embed.parameters()))        
        
        #vis_encoder
        # density_params += list(map(id, network.vis_encoder.parameters()))                
        # import ipdb;ipdb.set_trace()


        self.rest_params = filter(lambda x:id(x) not in density_params, network.parameters())  #提出剩下的参数        
        self.density_params = filter(lambda x:id(x) in density_params, network.parameters())         


        # import ipdb;ipdb.set_trace()
        self.paras =[{'params':self.density_params, 'lr': self.density_lr_init},
                {'params':self.rest_params, 'lr': self.lr_init},
            ]
        new_optim = optimizer(self.paras)
        return new_optim

    def __call__(self, optimizer, step, *args, **kwargs):
        lr=max(self.lr_init*(self.decay_rate**(step//self.decay_step)),self.lr_min)
        density_lr = self.density_lr_init*(self.decay_rate**(step//self.decay_step))

        # self.set_lr_for_density(optimizer, density_lr)
        optimizer.param_groups[0]['lr'] = density_lr
        optimizer.param_groups[1]['lr'] = lr        
        return lr

        

    # def set_lr_for_all(self, optimizer, lr):
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = lr

# class ExpDecayLRRayFeats(ExpDecayLR):
#     def construct_optimizer(self, optimizer, network):
#         paras = network.parameters()
#         return optimizer([para for para in paras] + network.ray_feats, lr=1e-3)

# class WarmUpExpDecayLR(LearningRateManager):
#     def __init__(self, cfg):
#         self.lr_warm=cfg['lr_warm']
#         self.warm_step=cfg['warm_step']
#         self.lr_init=cfg['lr_init']
#         self.decay_step=cfg['decay_step']
#         self.decay_rate=cfg['decay_rate']
#         self.lr_min=1e-5

#     def __call__(self, optimizer, step, *args, **kwargs):
#         if step<self.warm_step:
#             lr=self.lr_warm
#         else:
#             lr=max(self.lr_init*(self.decay_rate**((step-self.warm_step)//self.decay_step)),self.lr_min)
#         self.set_lr_for_all(optimizer,lr)
#         return lr

name2lr_manager={
    'exp_decay': ExpDecayLR,
    # 'exp_decay_ray_feats': ExpDecayLRRayFeats,
    # 'warm_up_exp_decay': WarmUpExpDecayLR,
}