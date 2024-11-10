class Config:
    def __init__(self):
        # Data
        self.output_dir = './outputs'
        self.pretrain_dataset_name = 'shapenet'
        self.validate_dataset_name = 'shapenet'
        self.use_height = False
        self.npoints = 8192

        # Model
        self.model = 'PN_SSG'

        # Training
        self.epochs = 200
        self.warmup_epochs = 1
        self.start_epoch = 0
        self.batch_size = 35
        self.lr = 3e-3
        self.lr_start = 1e-6
        self.lr_end = 1e-5
        self.update_freq = 1
        self.wd = 0.1
        self.betas = (0.9, 0.98)
        self.eps = 1e-8
        self.eval_freq = 1
        self.disable_amp = False
        self.resume = ''

        # System (General Settings)
        self.print_freq = 10
        self.workers = 4
        self.seed = 0
        self.gpu = [1]
        self.wandb = False
        self.test_ckpt_addr = ''


    def update_from_args(self, args):
        for key, value in vars(args).items():
            if hasattr(self, key):
                setattr(self, key, value)

    def __repr__(self):
        return '\n'.join([f"{k}: {v}" for k, v in vars(self).items()])

config = Config()