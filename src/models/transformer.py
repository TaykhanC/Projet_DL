"""
Architecture Transformer pour la génération de musique

Ce module définit l'architecture du modèle Transformer GPT-style
pour la génération de musique à partir de séquences MIDI tokenizées.

Architecture:
- TransformerBlock: Un bloc Transformer (Multi-Head Attention + FFN)
- MusicTransformer: Modèle complet pour la génération de musique
- Basé sur l'architecture GPT (decoder-only, causal masking)

Inspiré de: 
- "Attention Is All You Need" (Vaswani et al., 2017)
- "Music Transformer" (Huang et al., 2018)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class PositionalEncoding(nn.Module):
    """
    Encodage positionnel sinusoïdal pour le Transformer.
    
    Ajoute des informations de position aux embeddings pour que le modèle
    puisse distinguer la position des tokens dans la séquence.
    
    Args:
        d_model (int): Dimension des embeddings
        max_seq_len (int): Longueur maximale des séquences
        dropout (float): Taux de dropout
    """
    
    def __init__(self, d_model: int, max_seq_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Créer la matrice d'encodage positionnel [max_seq_len, d_model]
        position = torch.arange(max_seq_len).unsqueeze(1)  # [max_seq_len, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        
        pe = torch.zeros(max_seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)  # Dimensions paires
        pe[:, 1::2] = torch.cos(position * div_term)  # Dimensions impaires
        
        # Ajouter une dimension batch: [1, max_seq_len, d_model]
        pe = pe.unsqueeze(0)
        
        # Enregistrer comme buffer (ne sera pas entraîné)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor [batch_size, seq_len, d_model]
        Returns:
            Tensor [batch_size, seq_len, d_model] avec encodage positionnel
        """
        # Ajouter l'encodage positionnel
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention avec masque causal.
    
    Args:
        d_model (int): Dimension des embeddings
        num_heads (int): Nombre de têtes d'attention
        dropout (float): Taux de dropout
    """
    
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model doit être divisible par num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Projections Q, K, V
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        
        # Projection de sortie
        self.out_linear = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        causal_mask: bool = True
    ) -> torch.Tensor:
        """
        Args:
            x: Tensor [batch_size, seq_len, d_model]
            mask: Mask d'attention padding [batch_size, seq_len]
            causal_mask: Si True, applique un masque causal (pour autoregression)
            
        Returns:
            Tensor [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape
        
        # Projections linéaires et reshape pour multi-head
        # [batch_size, seq_len, d_model] -> [batch_size, num_heads, seq_len, head_dim]
        Q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Calcul des scores d'attention
        # [batch_size, num_heads, seq_len, seq_len]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Appliquer le masque causal (pour la génération autoregressive)
        if causal_mask:
            causal_mask_matrix = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device), 
                diagonal=1
            ).bool()
            scores = scores.masked_fill(causal_mask_matrix, float('-inf'))
        
        # Appliquer le masque de padding
        if mask is not None:
            # mask shape: [batch_size, seq_len]
            # Reshape pour broadcast: [batch_size, 1, 1, seq_len]
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax et dropout
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Appliquer l'attention aux valeurs
        # [batch_size, num_heads, seq_len, head_dim]
        attention_output = torch.matmul(attention_weights, V)
        
        # Concatener les têtes
        # [batch_size, seq_len, d_model]
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        # Projection finale
        output = self.out_linear(attention_output)
        
        return output


class FeedForward(nn.Module):
    """
    Feed-Forward Network (FFN) utilisé dans chaque bloc Transformer.
    
    Architecture: Linear -> ReLU -> Dropout -> Linear
    
    Args:
        d_model (int): Dimension des embeddings
        d_ff (int): Dimension interne du FFN (généralement 4 * d_model)
        dropout (float): Taux de dropout
    """
    
    def __init__(self, d_model: int, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor [batch_size, seq_len, d_model]
        Returns:
            Tensor [batch_size, seq_len, d_model]
        """
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerBlock(nn.Module):
    """
    Un bloc Transformer complet.
    
    Architecture:
        1. Multi-Head Self-Attention + Residual + LayerNorm
        2. Feed-Forward Network + Residual + LayerNorm
    
    Args:
        d_model (int): Dimension des embeddings
        num_heads (int): Nombre de têtes d'attention
        d_ff (int): Dimension du FFN
        dropout (float): Taux de dropout
    """
    
    def __init__(
        self, 
        d_model: int, 
        num_heads: int = 8, 
        d_ff: int = 2048, 
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Multi-Head Attention
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        # Feed-Forward
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Tensor [batch_size, seq_len, d_model]
            mask: Mask de padding [batch_size, seq_len]
            
        Returns:
            Tensor [batch_size, seq_len, d_model]
        """
        # Self-Attention + Residual + Norm
        attn_output = self.attention(x, mask=mask, causal_mask=True)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # FFN + Residual + Norm
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_output))
        
        return x


class MusicTransformer(nn.Module):
    """
    Modèle Transformer complet pour la génération de musique.
    
    Architecture GPT-style (decoder-only) avec masque causal pour
    l'auto-régression.
    
    Args:
        vocab_size (int): Taille du vocabulaire de tokens
        d_model (int): Dimension des embeddings
        num_heads (int): Nombre de têtes d'attention
        num_layers (int): Nombre de blocs Transformer
        d_ff (int): Dimension du FFN
        max_seq_len (int): Longueur maximale des séquences
        dropout (float): Taux de dropout
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        max_seq_len: int = 1024,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        
        # Token Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional Encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Blocs Transformer
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Layer finale pour prédire les tokens
        self.output_layer = nn.Linear(d_model, vocab_size)
        
        # Initialisation des poids
        self._init_weights()
    
    def _init_weights(self):
        """Initialisation des poids du modèle."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass du modèle.
        
        Args:
            input_ids: Tensor [batch_size, seq_len] des token IDs
            attention_mask: Tensor [batch_size, seq_len] masque de padding
            
        Returns:
            logits: Tensor [batch_size, seq_len, vocab_size] des scores pour chaque token
        """
        # Token embeddings
        x = self.token_embedding(input_ids)  # [batch_size, seq_len, d_model]
        
        # Positional encoding
        x = self.pos_encoding(x)
        
        # Passer à travers les blocs Transformer
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, mask=attention_mask)
        
        # Projection finale vers le vocabulaire
        logits = self.output_layer(x)  # [batch_size, seq_len, vocab_size]
        
        return logits
    
    def count_parameters(self) -> int:
        """Compte le nombre de paramètres entraînables."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Test du module
if __name__ == "__main__":
    """Test rapide du modèle."""
    
    print("🧪 Test du MusicTransformer\n")
    
    # Hyperparamètres
    vocab_size = 500
    batch_size = 4
    seq_len = 128
    d_model = 256
    num_heads = 8
    num_layers = 4
    
    # Créer le modèle
    model = MusicTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=1024,
        max_seq_len=1024,
        dropout=0.1
    )
    
    print(f"✓ Modèle créé")
    print(f"  Vocabulaire: {vocab_size}")
    print(f"  d_model: {d_model}")
    print(f"  Têtes d'attention: {num_heads}")
    print(f"  Nombre de couches: {num_layers}")
    print(f"  Paramètres entraînables: {model.count_parameters():,}")
    
    # Créer des données de test
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    # Forward pass
    print(f"\n🔄 Forward pass:")
    print(f"  Input shape: {input_ids.shape}")
    
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
    
    print(f"  Output shape: {logits.shape}")
    print(f"  ✓ Forward pass réussi!")
    
    # Vérifier la sortie
    assert logits.shape == (batch_size, seq_len, vocab_size)
    print(f"\n✓ Toutes les assertions passées!")