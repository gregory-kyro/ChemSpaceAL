# Necessary import statements
import os
import torch

class Config:
    """
    Class to handle the configuration settings for the GPT pretraining and alignment
    """
    
    BASE_PATH = '/content/drive/MyDrive/Generative_ML/current_data/'
    REGEX_PATTERN = r"(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|@@|\?|>|!|~|\*|\$|\%[0-9]{2}|[0-9])"

    """
    Initializes the configuration with the default and user-provided settings
    """
    def __init__(self, att_bias=False, gpt_bias=True, att_drop_rate=0.1,
                 gpt_drop_rate=0.1, n_layer=8, n_head=8, n_embed=256,
                 ff_mult=4, doGELU=True, attention_times=[], do_flash=True,
                 wandb_project='Pipeline', slice_data=False, batch_size=512,
                 betas=(0.965, 0.99), rho=0.04, weight_decay=0.1,
                 num_workers=0, smiles_key='smiles', mode='Active Learning',
                 protein_name='HNH.pdb', random_seed=42,
                 train_fname="moses_and_binding_no_rare_tokens_train.csv.gz",
                 val_fname="moses_and_binding_no_rare_tokens_test.csv.gz",
                 train_ckpt_name='model1_softdiv_al2',
                 current_cycle_prefix="model1_softdiv_al2"):

        DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        
        # Build dictionary
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
            "ff_mult": ff_mult,
            "doGELU": doGELU,
            "attention_times": attention_times,
            "do_flash": do_flash,
            "wandb_project": wandb_project,
            "slice_data": slice_data,
            "batch_size": batch_size,
            "betas": betas,
            "rho": rho,
            "weight_decay": weight_decay,
            "num_workers": num_workers
        }

        # Paths and additional settings
        self.protein_name = protein_name
        self.random_seed = random_seed
        self.mode = mode
        self.train_fname = train_fname
        self.val_fname = val_fname
        self.train_ckpt_name = train_ckpt_name
        self.current_cycle_prefix = current_cycle_prefix

        self.pretraining_path = self.BASE_PATH + '1. Pretraining/'
        self.generation_path = self.BASE_PATH + '2. Generation/'
        self.sampling_path = self.BASE_PATH + '3. Sampling/'
        self.diffdock_path = self.BASE_PATH + '4. DiffDock/'
        self.scoring_path = self.BASE_PATH + '5. Scoring/'
        self.al_path = self.BASE_PATH + '6. ActiveLearning/'

        self.protein_path = self.diffdock_path + 'proteins/' + self.protein_name

        self.set_mode(mode)

    def set_mode(self, mode):
        """
        Sets the mode and adjusts the configuration settings accordingly
        """
        job_config = {
            "train_path": f"{self.pretraining_path}datasets/{self.train_fname}",
            "val_path": f"{self.pretraining_path}datasets/{self.val_fname}",
            "wandb_runname": self.train_ckpt_name,
        }

        if mode == 'Pretraining':
            job_config.update({
                "lr_warmup": True,
                "lr_decay": True,
                "epochs": 30,
                "learning_rate": 3e-4,
                "save_ckpt_path": f"{self.pretraining_path}model_weights/{self.train_ckpt_name}.pt",
                "desc_path": f"{self.pretraining_path}dataset_descriptors/{self.train_fname.split('.')[0][:-6]}.yaml",
            })
        elif mode == 'Active Learning':
            dataset_desc_fname = "model1_softsub_al1_threshold11_softmax_sub"
            job_config.update({
                "lr_warmup": False,
                "lr_decay": True,
                "epochs": 10,
                "learning_rate": 3e-5,
                "save_ckpt_path": f"{self.al_path}model_weights/{self.train_ckpt_name}.pt",
                "desc_path": f"{self.al_path}dataset_descriptors/{dataset_desc_fname}.yaml"
            })

        self.config_dict.update(job_config)

    def get_config(self):
        """
        Retrieve the configuration dictionary
        """
        return self.config_dict

"""
# Example usage:
config = Config(mode='Active Learning')
config_dict = config.get_config()
print(config_dict)
"""