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
