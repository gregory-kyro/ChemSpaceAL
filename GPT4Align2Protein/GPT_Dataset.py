# Necessary import statements
import re
import torch
import yaml
from torch.utils.data import Dataset

class SMILESDataset(Dataset):
    """
    Dataset class for handling SMILES data
    """

    def __init__(self, data=None, chars=None, block_size=None, len_data=None):
        """
        Initialize the dataset with the given SMILES data, vocabulary characters, block size, and data length
        """
        if chars is None:
            self.desc_only = True
            return
        self.desc_only = False
        self.vocab = set(chars)
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: s for s, i in self.stoi.items()}
        self.block_size = block_size
        self.data = data
        self.len_data = len_data

    def export_desc_attributes(self, export_path):
        """
        Export description attributes of the dataset to a given path
        """
        attr_dict = {
            "desc_only": self.desc_only,
            "vocab_size": self.vocab_size,
            "block_size": self.block_size,
            "stoi": self.stoi,
            "itos": self.itos,
            "len_data": self.len_data
        }
        with open(export_path, 'w') as f:
            yaml.dump(attr_dict, f)

    def load_desc_attributes(self, load_path):
        """
        Load description attributes of the dataset from a given path
        """
        with open(load_path, 'r') as f:
            attr_dict = yaml.load(f, Loader=yaml.SafeLoader)
        self.__dict__.update(attr_dict)

    def __len__(self):
        """
        Get the length of the dataset
        """
        assert not self.desc_only, "Dataset is not initialized"
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get the item at the given index
        """
        assert not self.desc_only, "Dataset is not initialized"
        smiles = self.data[idx].strip()
        # Define regular expression pattern used to identify characters in the SMILES strings
        regex = re.compile(REGEX_PATTERN)
        smiles_matches = regex.findall(smiles)

        if len(smiles_matches) > self.block_size + 1:
            smiles = smiles[:self.block_size + 1]

        embedded_smile = [self.stoi[s] for s in smiles_matches]
        x = torch.tensor(embedded_smile[:-1], dtype=torch.long)
        y = torch.tensor(embedded_smile[1:], dtype=torch.long)
      
        return x, y
