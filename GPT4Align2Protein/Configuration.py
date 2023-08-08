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
                 current_cycle_prefix="model1_softdiv_al2", context='!', temperature=1.0,
                 gen_batch_size=8, load_ckpt_name='model1_softdiv_al2.pt',
                 gen_val_fname="moses_and_binding_no_rare_tokens_test.csv.gz",
                 pca_fname='scaler_pca_moses+bindingdb.pkl', n_clusters=100, 
                 samples_per_cluster=10):

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
            "num_workers": num_workers,
            "protein_name": protein_name,
            "random_seed": random_seed,
            "mode": mode,
            "train_fname": train_fname,
            "val_fname": val_fname,
            "train_ckpt_name": train_ckpt_name,
            "current_cycle_prefix": current_cycle_prefix,
            "pretraining_path": self.BASE_PATH + '1. Pretraining/',
            "generation_path": self.BASE_PATH + '2. Generation/',
            "sampling_path": self.BASE_PATH + '3. Sampling/',
            "diffdock_path": self.BASE_PATH + '4. DiffDock/',
            "scoring_path": self.BASE_PATH + '5. Scoring/',
            "al_path": self.BASE_PATH + '6. ActiveLearning/',
            "protein_path": self.BASE_PATH + '4. DiffDock/' + 'proteins/' + protein_name,
            "generation_context": context,
            "gen_size": 100_000,
            "inference_temp": temperature,
            "gen_batch_size": gen_batch_size,
            "load_ckpt_name": load_ckpt_name,
            "gen_val_fname": gen_val_fname,
            "diffdock_scored_path_list": [f"{self.BASE_PATH + '5. Scoring/'}scored_dataframes/{i}" for i in ["model1_baseline.csv", "model1_softdiv_al1.csv"]],
            "al_trainsets_path_list": [f"{self.BASE_PATH + '6. ActiveLearning/'}training_sets/{i}" for i in ["model1_baseline_threshold11_softmax_divf0.25.csv", 'model1_softdiv_al1_threshold11_softmax_divf0.25.csv']],
            "pca_fname": pca_fname,
            "n_clusters": n_clusters,
            "samples_per_cluster": samples_per_cluster,
            "path_to_completions": self.BASE_PATH + '2. Generation/'+f"_temp{temperature}_completions.csv",
            "path_to_predicted": self.BASE_PATH + '2. Generation/'+f"_temp{temperature}_processed.csv",
            "path_to_descriptors": self.BASE_PATH + '3. Sampling/' + "descriptors/" + self.BASE_PATH + '2. Generation/'.split('/')[-1] +f"_temp{temperature}.pkl",
            "path_to_pca": f"{self.BASE_PATH + '3. Sampling/'}pca_weights/{pca_fname}",
            "kmeans_save_path": f"{self.BASE_PATH + '3. Sampling/'}kmeans_objects/{current_cycle_prefix}_k{n_clusters}means.pkl",
            "clusters_save_path": f"{self.BASE_PATH + '3. Sampling/'}clusterings/{current_cycle_prefix}_cluster_to_mols.pkl",
            "samples_save_path": f"{self.BASE_PATH + '3. Sampling/'}clusterings/{current_cycle_prefix}_cluster_to_samples.pkl",
            "diffdock_save_path": f"{self.BASE_PATH + '4. DiffDock/'}sampled_mols/{current_cycle_prefix}_samples.csv",
            "diffdock_results_path": f"{self.BASE_PATH + '4. DiffDock/'}poses/{current_cycle_prefix}_T4/"
                     }

        self.set_mode(mode)

    def set_mode(self, mode):
        """
        Sets the mode and adjusts the configuration settings accordingly
        """
        job_config = {
            "train_path": f"{self.BASE_PATH + '1. Pretraining/'}datasets/{self.config_dict['train_fname']}",
            "val_path": f"{self.BASE_PATH + '1. Pretraining/'}datasets/{self.config_dict['val_fname']}",
            "wandb_runname": self.config_dict['train_ckpt_name'],
        }

        if mode == 'Pretraining':
            job_config.update({
                "lr_warmup": True,
                "lr_decay": True,
                "epochs": 30,
                "learning_rate": 3e-4,
                "save_ckpt_path": f"{self.BASE_PATH + '1. Pretraining/'}model_weights/{self.config_dict['train_ckpt_name']}.pt",
                "desc_path": f"{self.BASE_PATH + '1. Pretraining/'}dataset_descriptors/{self.config_dict['train_fname'].split('.')[0][:-6]}.yaml",
                "load_ckpt_path": f"{self.BASE_PATH + '1. Pretraining/'}model_weights/{self.config_dict['load_ckpt_name']}"
            })
        elif mode == 'Active Learning':
            dataset_desc_fname = "model1_softsub_al1_threshold11_softmax_sub"
            job_config.update({
                "lr_warmup": False,
                "lr_decay": True,
                "epochs": 10,
                "learning_rate": 3e-5,
                "save_ckpt_path": f"{self.BASE_PATH + '6. ActiveLearning/'}model_weights/{self.config_dict['train_ckpt_name']}.pt",
                "desc_path": f"{self.BASE_PATH + '6. ActiveLearning/'}dataset_descriptors/{self.config_dict['dataset_desc_fname']}.yaml",
                "load_ckpt_path": f"{self.BASE_PATH + '6. ActiveLearning/'}model_weights/{self.config_dict['load_ckpt_name']}"

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
