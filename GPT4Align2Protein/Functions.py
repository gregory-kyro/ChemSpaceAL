# Necessary import statements
import torch
import torch.nn.functional as F
from rdkit import Chem
from .Configuration import *

@torch.no_grad()
def sample(model, x, steps, temperature=1.0):
    """
    Sample a sequence of tokens from a given model
    """
    block_size = model.get_block_size()
    model.eval()
    for k in range(steps):
        x_cond = x if x.size(1) <= block_size else x[:, -block_size:] # Get last block_size elements
        logits, _ = model(x_cond) # Get logits from model
        logits = logits[:, -1, :] / temperature # Extract logits for next token
        probs = F.softmax(logits, dim=-1)
        ix = torch.multinomial(probs, 1)
        x = torch.cat((x, ix), dim=1) # Concatenate token with sequence
    return x

def check_novelty(gen_smiles, train_smiles):
    """
    Check the novelty of generated SMILES by comparing with training set
    """
    if len(gen_smiles) == 0:
        novel_ratio = 0
    else:
        duplicates = [1 for mol in gen_smiles if mol in train_smiles] # Find duplicates
        novel_ratio = (len(gen_smiles) - sum(duplicates)) * 100 / len(gen_smiles) # Compute novelty ratio
    return novel_ratio

def canonic_smiles(smiles_or_mol):
    """
    Convert SMILES or RDKit mol object to canonical SMILES string
    """
    mol = get_mol(smiles_or_mol) # Convert to RDKit mol object
    if mol is None:
        return None
    return Chem.MolToSmiles(mol) # Convert to canonical SMILES




def load_data(config_dict, mode='Active Learning', forced_block_size=None):
    """
    Load data to be used for either pretraining or active learning
    """

    # Pretraining
    if mode == 'Pretraining':
        # Get compression
        if 'gz' in config_dict["train_path"]:
            compression = 'gzip'
        else:
            compression = None

        # Slice data if set in configuration dictionary
        if (cut:=config_dict["slice_data"]):
            train_data = pd.read_csv(config_dict["train_path"], compression=compression)[:cut]
            val_data = pd.read_csv(config_dict["val_path"], compression=compression)[:cut]

        else:
            train_data = pd.read_csv(config_dict["train_path"], compression=compression)
            val_data = pd.read_csv(config_dict["val_path"], compression=compression)
        iterators = (train_data[config_dict['smiles_key']].values, val_data[config_dict['smiles_key']].values)
        assert len(train_data) == len(train_data[config_dict['smiles_key']].values), "There's no reason why this shouldn't be true"
    
    elif mode == 'Active Learning':
        print(f"Loading Active Learning dataset from", '/'.join(config_dict["al_path"].split('/')[6:]))
        al_data = pd.read_csv(config_dict["al_path"])
        iterators = (al_data[config_dict['smiles_key']].values, )
    else:
        raise KeyError(f"Only pretraining and active learning are currently supported")

    # Compile pattern into a regular expression object that can be used for matching operations
    regex = re.compile(Config.REGEX_PATTERN)
    char_set = {'<', '!', '~'}

    max_len = 0
    for iterator in iterators:
        for i in iterator:
            chars = regex.findall(i.strip())
            max_len = max(max_len, len(chars))
            for char in chars:
                char_set.add(char)

    chars = sorted(list(char_set))
    max_len += 1    # Accounting for the start token, which hasn't been added yet
    if forced_block_size is not None:
        assert mode == 'Active Learning', "Cannot force a block size in pretraining"
        max_len = forced_block_size

    datasets = []
    for iterator in iterators:
        padded = ['!' + i + '~' + '<'*(max_len - 1 - len(regex.findall(i.strip()))) for i in iterator]
        dataset = SMILESDataset(data=padded, chars=chars, block_size=max_len, len_data=len(iterator))
        datasets.append(dataset)
        
    dataset.export_desc_attributes(config_dict["desc_path"])
    
    return datasets
