"""
Dataset PyTorch pour les séquences MIDI tokenizées

Ce module définit le dataset custom qui charge les séquences tokenizées
depuis les fichiers pickle créés par le preprocessing.

Architecture:
- MIDIDataset: Dataset PyTorch pour charger les séquences
- Gère le padding si nécessaire
- Supporte le masking pour l'attention du Transformer
"""

import pickle
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Optional
import numpy as np


class MIDIDataset(Dataset):
    """
    Dataset PyTorch pour les séquences MIDI tokenizées.
    
    Args:
        sequences_path (Path): Chemin vers le fichier .pkl contenant les séquences
        max_seq_len (int): Longueur maximale des séquences (pour padding)
        pad_token_id (int): ID du token de padding (par défaut 0)
        
    Attributes:
        sequences (List[List[int]]): Liste des séquences tokenizées
        max_seq_len (int): Longueur maximale des séquences
        pad_token_id (int): Token utilisé pour le padding
    """
    
    def __init__(
        self, 
        sequences_path: Path, 
        max_seq_len: int = 1024,
        pad_token_id: int = 0
    ):
        """Initialise le dataset."""
        self.sequences_path = sequences_path
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        
        # Charger les séquences
        print(f"📂 Chargement des séquences depuis: {sequences_path}")
        with open(sequences_path, 'rb') as f:
            self.sequences = pickle.load(f)
        
        print(f"✓ {len(self.sequences)} séquences chargées")
        
        # Statistiques
        lengths = [len(seq) for seq in self.sequences]
        print(f"  Longueur moyenne: {np.mean(lengths):.1f} tokens")
        print(f"  Longueur min/max: {min(lengths)}/{max(lengths)} tokens")
    
    def __len__(self) -> int:
        """Retourne le nombre de séquences dans le dataset."""
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> dict:
        """
        Retourne une séquence et son masque d'attention.
        
        Args:
            idx (int): Index de la séquence
            
        Returns:
            dict: Dictionnaire contenant:
                - 'input_ids': Tensor [seq_len] des tokens d'entrée
                - 'attention_mask': Tensor [seq_len] masque (1 = token réel, 0 = padding)
                - 'labels': Tensor [seq_len] des tokens de sortie (décalés de 1)
                
        Note:
            Pour l'entraînement causal, input_ids = séquence[:-1] et labels = séquence[1:]
            Cela permet au modèle d'apprendre à prédire le token suivant.
        """
        # Récupérer la séquence
        sequence = self.sequences[idx].copy()
        seq_len = len(sequence)
        
        # Si la séquence est trop longue, la tronquer
        if seq_len > self.max_seq_len:
            sequence = sequence[:self.max_seq_len]
            seq_len = self.max_seq_len
        
        # Créer le masque d'attention (1 pour les tokens réels, 0 pour le padding)
        attention_mask = [1] * seq_len
        
        # Padding si nécessaire
        if seq_len < self.max_seq_len:
            padding_length = self.max_seq_len - seq_len
            sequence = sequence + [self.pad_token_id] * padding_length
            attention_mask = attention_mask + [0] * padding_length
        
        # Convertir en tensors
        input_ids = torch.tensor(sequence, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        
        # Pour l'entraînement causal: input = tokens[:-1], target = tokens[1:]
        # Le modèle apprend à prédire le token suivant
        input_ids_shifted = input_ids[:-1]
        labels = input_ids[1:]
        attention_mask_shifted = attention_mask[:-1]
        
        return {
            'input_ids': input_ids_shifted,        # [seq_len-1]
            'attention_mask': attention_mask_shifted,  # [seq_len-1]
            'labels': labels                        # [seq_len-1]
        }
    
    def get_vocab_size(self) -> int:
        """
        Estime la taille du vocabulaire à partir des séquences.
        
        Returns:
            int: Taille du vocabulaire (max token ID + 1)
        """
        max_token = max(max(seq) for seq in self.sequences)
        return max_token + 1


def create_dataloaders(
    train_sequences_path: Path,
    val_sequences_path: Path,
    batch_size: int = 8,
    max_seq_len: int = 1024,
    num_workers: int = 4,
    pin_memory: bool = True
) -> tuple:
    """
    Crée les DataLoaders pour l'entraînement et la validation.
    
    Args:
        train_sequences_path: Chemin vers les séquences d'entraînement
        val_sequences_path: Chemin vers les séquences de validation
        batch_size: Taille des batchs
        max_seq_len: Longueur maximale des séquences
        num_workers: Nombre de workers pour le chargement
        pin_memory: Utiliser pin_memory pour GPU
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    from torch.utils.data import DataLoader
    
    # Créer les datasets
    train_dataset = MIDIDataset(
        sequences_path=train_sequences_path,
        max_seq_len=max_seq_len
    )
    
    val_dataset = MIDIDataset(
        sequences_path=val_sequences_path,
        max_seq_len=max_seq_len
    )
    
    # Créer les dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Mélanger les données d'entraînement
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # Éviter les derniers batchs incomplets
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Pas de shuffle pour la validation
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    print(f"\n✓ DataLoaders créés:")
    print(f"  Train: {len(train_dataset)} séquences, {len(train_loader)} batchs")
    print(f"  Val:   {len(val_dataset)} séquences, {len(val_loader)} batchs")
    
    return train_loader, val_loader


# Test du module
if __name__ == "__main__":
    """Test rapide du dataset."""
    from pathlib import Path
    
    # Chemins (adapter selon ton setup)
    # Depuis src/data/dataset.py, aller vers src/data/processed/
    PROCESSED_DIR = Path(__file__).parent / "processed"
    train_path = PROCESSED_DIR / "train_sequences.pkl"
    
    if train_path.exists():
        # Créer le dataset
        dataset = MIDIDataset(
            sequences_path=train_path,
            max_seq_len=1024
        )
        
        # Tester un item
        print(f"\n🧪 Test d'un item:")
        item = dataset[0]
        print(f"  input_ids shape: {item['input_ids'].shape}")
        print(f"  attention_mask shape: {item['attention_mask'].shape}")
        print(f"  labels shape: {item['labels'].shape}")
        print(f"  Premiers tokens: {item['input_ids'][:10].tolist()}")
        
        # Vocabulaire
        vocab_size = dataset.get_vocab_size()
        print(f"\n  Taille du vocabulaire: {vocab_size}")
    else:
        print(f"❌ Fichier non trouvé: {train_path}")
        print("   Exécute d'abord le notebook de preprocessing !")