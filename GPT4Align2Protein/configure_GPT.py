import torch

class Config:
    """
    Class to handle the configuration settings for the project.
    """
    def __init__(self, att_bias=False, gpt_bias=True, att_drop_rate=0.1,
                 gpt_drop_rate=0.1, n_layer=8, n_head=8, n_embed=256,
                 ff_mult=4, doGELU=True, attention_times=[], do_flash=True,
                 wandb_project='Pipeline', slice_data=False, batch_size=512,
                 betas=(0.965, 0.99), rho=0.04, weight_decay=0.1,
                 num_workers=0, smiles_key='smiles', mode='Active Learning'):
        """
        Initialize the configuration with the default and user-provided settings.
        """
        DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        
        self.config_dict = {
            "device": DEVICE,
            "smiles_key": smiles_key,
            "att_bias": att_bias,
            "gpt_bias": gpt_bias,
            "att_drop_rate": att_drop_rate,
            "gpt_drop_rate": gpt_drop_rate,
            "n_layer": n_layer,
            "n_head": n_head,
            "n_embed": n_embed,
            "ff_mult": ff_mult,  # multiplier for Feed Forward number of hidden units inside multihead,
            "doGELU": doGELU,    # else ReLU
            "attention_times": attention_times,
            "do_flash": do_flash,
            "wandb_project": wandb_project,
            "slice_data": slice_data,  # False for all data
            "batch_size": batch_size,  # 512,
            "betas": betas,  # (0.9, 0.95)
            "rho": rho,  # For SophiaG
            "weight_decay": weight_decay,
            "num_workers": num_workers  # number of worker processes to use for loading data
        }
        
        self.set_mode(mode)

    def set_mode(self, mode):
        """
        Sets the mode and adjusts the configuration settings accordingly.
        """
        job_config = {}

        if mode == 'Pretraining':
            job_config.update({'lr_warmup': True, 'lr_decay': True, 'epochs': 30, 'learning_rate': 3e-4,})
        elif mode == 'Active Learning':
            job_config.update({'lr_warmup': False, 'lr_decay': True, 'epochs': 10, 'learning_rate': 3e-5,})

        self.config_dict.update(job_config)

    def get_config(self):
        """
        Retrieve the configuration dictionary.
        """
        return self.config_dict

"""
# Example usage:
config = Config(mode='Active Learning')
config_dict = config.get_config()
print(config_dict)
"""
