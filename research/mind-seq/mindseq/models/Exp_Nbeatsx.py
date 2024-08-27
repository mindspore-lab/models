from .nbeatsx import Nbeats
from .Exp_basic import Exp_Basic
from ..utils.nbeatsx_utils import EPF, EPFInfo
from ..utils.nbeatsx_utils import TimeSeriesDataset
from ..utils.nbeatsx_utils import TimeSeriesLoader
import numpy as np
class Exp_Nbeatsx(Exp_Basic):
    def __init__(self, args):
        super(Exp_Nbeatsx, self).__init__(args)
        self.train_loader = TimeSeriesLoader(model='nbeats',
                            ts_dataset=self.ts_dataset,
                            window_sampling_limit=365*4*24,
                            offset=0,
                            input_size=7*24,
                            output_size=24,
                            idx_to_sample_freq=24,
                            batch_size=self.args.batch_size,
                            is_train_loader=True,
                            shuffle=True,
                            rank_id=self.args.rank_id,
                            dist_flag=self.args.device_num)
        print("Finish train_loader")

        self.val_loader = TimeSeriesLoader(model='nbeats',
                                ts_dataset=self.ts_dataset,
                                window_sampling_limit=365*4*24,
                                offset=0,
                                input_size=7*24,
                                output_size=24,
                                idx_to_sample_freq=24,
                                batch_size=self.args.batch_size,
                                is_train_loader=False,
                                shuffle=False,
                                rank_id=self.args.rank_id,
                                dist_flag=None)
        print("Finish val_loader")
        self.do_train = args.do_train

    def _build_model(self):
        Y_df, X_df, _ = EPF.load_groups(directory=self.args.root_path, groups=['NP'])
        # train_mask: 1 to keep, 0 to mask
        train_mask = np.ones(len(Y_df))
        train_mask[-168:] = 0 # Last week of data (168 hours)
        self.ts_dataset = TimeSeriesDataset(Y_df=Y_df, X_df=X_df, ts_train_mask=train_mask)
        include_var_dict = {'y': [-8,-4,-3,-2],
                    'Exogenous1': [-8,-2,-1],
                    'Exogenous2': [-8,-2,-1],
                    'week_day': [-1]}
        model = Nbeats(input_size_multiplier=7,
            output_size=24,
            shared_weights=False,
            initialization='glorot_normal',
            activation='selu',
            stack_types=['identity']+['exogenous_tcn'],
            n_blocks=[1, 1],
            n_layers=[2, 2],
            n_hidden=[[512,512], [512,512]],
            n_harmonics=0, # not used with exogenous_tcn
            n_polynomials=0, # not used with exogenous_tcn
            x_s_n_hidden = 0,
            exogenous_n_channels=9,
            include_var_dict=include_var_dict,
            t_cols=self.ts_dataset.t_cols,
            batch_normalization = True,
            dropout_prob_theta=0.1,
            dropout_prob_exogenous=0,
            learning_rate=0.001,
            lr_decay=0.5,
            n_lr_decay_steps=3,
            early_stopping=5000,
            weight_decay=0,
            l1_theta=0,
            n_iterations=self.args.train_epochs,
            loss='MAE',
            loss_hypar=0.5,
            val_loss='MAE',
            seasonality=24, # not used: only used with MASE loss
            random_seed=self.args.seed,
            distribute=self.args.distribute,
            rank_id=self.args.rank_id if self.args.rank_id is not None else 0)
        return model
    
    def train(self):
        self.model.fit(train_ts_loader=self.train_loader, val_ts_loader=self.val_loader, eval_steps=1)

    def test(self):
        self.model.test(train_ts_loader=self.train_loader, val_ts_loader=self.val_loader,
                ckpt_path=self.args.ckpt_path if not self.do_train else \
                            f"./checkpoints/train_ckpt/Nbeatsx_best_{self.args.rank_id if self.args.rank_id is not None else 0}.ckpt")