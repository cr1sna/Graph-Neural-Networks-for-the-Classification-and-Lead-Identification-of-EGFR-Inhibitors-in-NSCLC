import numpy as np
import torch
from torch_geometric.data import Data
from rdkit import Chem

# Allowed atom types
ATOM_TYPES = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P']

def one_hot_encoding(x, permitted_list):
    """
    Maps input elements x which are not in the permitted list to the last element
    of the permitted list.
    """
    if x not in permitted_list:
        x = permitted_list[-1]
    binary_encoding = [int(boolean_value) for boolean_value in list(map(lambda s: x == s, permitted_list))]
    return binary_encoding

def get_atom_features(atom):
    """
    Takes an RDKit atom object and returns a list of features.
    Total features: 10 + 6 + 1 + 5 + 5 + 1 + 1 = 29
    """
    # 1. Atom type (One-hot, 10 dims)
    atom_type_enc = one_hot_encoding(str(atom.GetSymbol()), ATOM_TYPES + ['Unknown'])
    
    # 2. Degree (One-hot, 6 dims)
    degree_enc = one_hot_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 'MoreThanFour'])
    
    # 3. Formal charge (Integer, 1 dim)
    formal_charge = [atom.GetFormalCharge()]
    
    # 4. Number of Hydrogens (One-hot, 5 dims)
    num_h_enc = one_hot_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 'MoreThanThree'])
    
    # 5. Hybridization (One-hot, 5 dims)
    hybridization_type = atom.GetHybridization()
    hybridization_enc = one_hot_encoding(str(hybridization_type), ['S', 'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'OTHER'])[:5] # Keep to 5 dims for simplicity as per plan
    
    # Adjust hybridization to match plan exactly if possible, or just use what we have.
    hybridization_enc = one_hot_encoding(str(hybridization_type), ['SP', 'SP2', 'SP3', 'SP3D', 'SP3D2'])
    
    # 6. Aromaticity (Binary, 1 dim)
    aromaticity = [1 if atom.GetIsAromatic() else 0]
    
    # 7. In ring (Binary, 1 dim)
    in_ring = [1 if atom.IsInRing() else 0]
    
    atom_features = atom_type_enc + degree_enc + formal_charge + num_h_enc + hybridization_enc + aromaticity + in_ring
    return np.array(atom_features, dtype=np.float32)

def get_bond_features(bond):
    """
    Takes an RDKit bond object and returns a list of features.
    Total features: 4 + 1 + 1 + 6 = 12
    """
    bond_type = bond.GetBondType()
    
    # 1. Bond type (One-hot, 4 dims)
    bond_type_enc = one_hot_encoding(str(bond_type), ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC'])
    
    # 2. Conjugated (Binary, 1 dim)
    conjugated = [1 if bond.GetIsConjugated() else 0]
    
    # 3. In ring (Binary, 1 dim)
    in_ring = [1 if bond.IsInRing() else 0]
    
    # 4. Stereo (One-hot, 6 dims)
    stereo_type = bond.GetStereo()
    stereo_enc = one_hot_encoding(str(stereo_type), ['STEREONONE', 'STEREOANY', 'STEREOZ', 'STEREOE', 'STEREOCIS', 'STEREOTRANS'])
    
    bond_features = bond_type_enc + conjugated + in_ring + stereo_enc
    return np.array(bond_features, dtype=np.float32)

def mol_to_graph(smiles, label=None):
    """
    Converts a SMILES string to a PyTorch Geometric Data object.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
        
    num_atoms = mol.GetNumAtoms()
    num_bonds = mol.GetNumBonds()
    
    # Node features
    node_features = []
    for atom in mol.GetAtoms():
        node_features.append(get_atom_features(atom))
    x = torch.tensor(np.array(node_features), dtype=torch.float)
    
    # Edge index and edge features
    edge_indices = []
    edge_features = []
    
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        
        # Graph is undirected, add edges in both directions
        edge_indices += [[i, j], [j, i]]
        
        b_feature = get_bond_features(bond)
        edge_features += [b_feature, b_feature]
        
    if len(edge_indices) > 0:
        edge_index = torch.tensor(np.array(edge_indices), dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(np.array(edge_features), dtype=torch.float)
    else:
        # Handle molecules with no bonds (e.g., single ions)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 12), dtype=torch.float)
        
    # Create Data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    if label is not None:
        data.y = torch.tensor([label], dtype=torch.float)
        
    data.smiles = smiles
    
    return data

if __name__ == "__main__":
    # Test
    smiles = "CC(=O)OC1=CC=CC=C1C(=O)O" # Aspirin
    data = mol_to_graph(smiles, label=1)
    print(f"Nodes: {data.x.shape}")
    print(f"Edges: {data.edge_index.shape}")
    print(f"Edge Attr: {data.edge_attr.shape}")
