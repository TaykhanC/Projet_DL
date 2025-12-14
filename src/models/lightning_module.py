"""
Module PyTorch Lightning pour l'entraînement du Music Transformer

Ce module encapsule le modèle Transformer dans un LightningModule
pour gérer automatiquement l'entraînement, la validation et l'optimisation.

PyTorch Lightning gère:
- Le training loop (forward, backward, optimizer step)
- La validation
- Les checkpoints
- Les logs (TensorBoard, WandB, etc.)
- Le multi-GPU automatique
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, Optional, Tuple

# Import qui fonctionne à la fois en module et en standalone
try:
    from .transformer import MusicTransformer
except ImportError:
    from transformer import MusicTransformer


class MusicTransformerLightning(pl.LightningModule):
    """
    Module Lightning pour l'entraînement du Music Transformer.
    
    Args:
        vocab_size (int): Taille du vocabulaire
        d_model (int): Dimension des embeddings
        num_heads (int): Nombre de têtes d'attention
        num_layers (int): Nombre de blocs Transformer
        d_ff (int): Dimension du FFN
        max_seq_len (int): Longueur maximale des séquences
        dropout (float): Taux de dropout
        learning_rate (float): Learning rate pour l'optimizer
        warmup_steps (int): Nombre de steps de warmup pour le scheduler
        weight_decay (float): Weight decay pour l'optimizer
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        max_seq_len: int = 1024,
        dropout: float = 0.1,
        learning_rate: float = 1e-4,
        warmup_steps: int = 4000,
        weight_decay: float = 0.01,
    ):
        super().__init__()
        
        # Sauvegarder les hyperparamètres (accessible via self.hparams)
        self.save_hyperparameters()
        
        # Créer le modèle
        self.model = MusicTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            max_seq_len=max_seq_len,
            dropout=dropout
        )
        
        # Loss function (Cross Entropy avec ignore_index pour le padding)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)  # -100 = padding token
        
        # Métriques pour le logging
        self.train_loss_history = []
        self.val_loss_history = []
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            
        Returns:
            logits: [batch_size, seq_len, vocab_size]
        """
        return self.model(input_ids, attention_mask)
    
    def _compute_loss(
        self, 
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calcule la loss pour un batch.
        
        Args:
            batch: Dict contenant 'input_ids', 'attention_mask', 'labels'
            
        Returns:
            loss: Scalar loss
            accuracy: Accuracy des prédictions
            perplexity: Perplexité du modèle
        """
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        
        # Forward pass
        logits = self(input_ids, attention_mask)  # [batch_size, seq_len, vocab_size]
        
        # Reshape pour la loss
        # logits: [batch_size * seq_len, vocab_size]
        # labels: [batch_size * seq_len]
        logits_flat = logits.view(-1, logits.size(-1))
        labels_flat = labels.view(-1)
        
        # Calculer la loss
        loss = self.criterion(logits_flat, labels_flat)
        
        # Calculer l'accuracy (en ignorant les tokens de padding)
        with torch.no_grad():
            predictions = torch.argmax(logits_flat, dim=-1)
            mask = labels_flat != 0  # Masque pour ignorer le padding
            if mask.sum() > 0:
                accuracy = (predictions[mask] == labels_flat[mask]).float().mean()
            else:
                accuracy = torch.tensor(0.0, device=loss.device)
            
            # Perplexité = exp(loss)
            perplexity = torch.exp(loss)
        
        return loss, accuracy, perplexity
    
    def training_step(
        self, 
        batch: Dict[str, torch.Tensor], 
        batch_idx: int
    ) -> torch.Tensor:
        """
        Step d'entraînement (appelé automatiquement par Lightning).
        
        Args:
            batch: Batch de données
            batch_idx: Index du batch
            
        Returns:
            loss: Loss pour ce batch
        """
        loss, accuracy, perplexity = self._compute_loss(batch)
        
        # Log les métriques (visible dans TensorBoard/WandB)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_ppl', perplexity, on_step=False, on_epoch=True)
        
        # Sauvegarder pour les graphiques
        self.train_loss_history.append(loss.item())
        
        return loss
    
    def validation_step(
        self, 
        batch: Dict[str, torch.Tensor], 
        batch_idx: int
    ) -> torch.Tensor:
        """
        Step de validation (appelé automatiquement par Lightning).
        
        Args:
            batch: Batch de données
            batch_idx: Index du batch
            
        Returns:
            loss: Loss pour ce batch
        """
        loss, accuracy, perplexity = self._compute_loss(batch)
        
        # Log les métriques
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_ppl', perplexity, on_step=False, on_epoch=True)
        
        # Sauvegarder pour les graphiques
        self.val_loss_history.append(loss.item())
        
        return loss
    
    def configure_optimizers(self):
        """
        Configure l'optimizer et le learning rate scheduler.
        
        Utilise:
        - AdamW optimizer avec weight decay
        - Warmup + Cosine decay scheduler
        """
        # Optimizer: AdamW (Adam avec weight decay)
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            betas=(0.9, 0.98),  # Betas recommandés pour Transformers
            eps=1e-9,
            weight_decay=self.hparams.weight_decay
        )
        
        # Scheduler: Warmup + Cosine Annealing
        def lr_lambda(current_step: int) -> float:
            """Learning rate schedule avec warmup."""
            warmup_steps = self.hparams.warmup_steps
            
            if current_step < warmup_steps:
                # Phase de warmup: augmentation linéaire
                return float(current_step) / float(max(1, warmup_steps))
            else:
                # Phase de decay: cosine annealing
                progress = float(current_step - warmup_steps) / float(
                    max(1, self.trainer.max_steps - warmup_steps)
                )
                return max(0.1, 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159265359))))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',  # Update à chaque step
                'frequency': 1
            }
        }
    
    def on_train_epoch_end(self):
        """Appelé à la fin de chaque epoch d'entraînement."""
        # Log le learning rate actuel
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('learning_rate', current_lr)
    
    @torch.no_grad()
    def generate(
        self,
        prompt: torch.Tensor,
        max_length: int = 512,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95
    ) -> torch.Tensor:
        """
        Génère une séquence de tokens de manière autoregressive.
        
        Args:
            prompt: Tensor [1, prompt_len] des tokens de départ
            max_length: Longueur maximale de la génération
            temperature: Température pour le sampling (plus haut = plus aléatoire)
            top_k: Nombre de tokens les plus probables à considérer
            top_p: Seuil de probabilité cumulée (nucleus sampling)
            
        Returns:
            generated: Tensor [1, max_length] de la séquence générée
        """
        self.eval()
        device = next(self.parameters()).device
        prompt = prompt.to(device)
        
        generated = prompt.clone()
        
        for _ in range(max_length - prompt.size(1)):
            # Forward pass sur toute la séquence générée jusqu'ici
            logits = self(generated)  # [1, seq_len, vocab_size]
            
            # Prendre les logits du dernier token
            next_token_logits = logits[0, -1, :] / temperature  # [vocab_size]
            
            # Top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][-1]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = False
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sampling avec softmax
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Correction CRITIQUE → toujours un entier, jamais un tuple ni un tensor bizarre
            next_token_id = int(next_token.item())
            next_token_tensor = torch.tensor([[next_token_id]], device=device)

            # Ajouter proprement
            generated = torch.cat([generated, next_token_tensor], dim=1)

            # Stop si pad (optionnel)
            if next_token_id == 0:
                break
        
        return generated


# Fonction helper pour créer le module
def create_model(
    vocab_size: int,
    d_model: int = 512,
    num_heads: int = 8,
    num_layers: int = 6,
    d_ff: int = 2048,
    max_seq_len: int = 1024,
    dropout: float = 0.1,
    learning_rate: float = 1e-4,
    warmup_steps: int = 4000,
    weight_decay: float = 0.01
) -> MusicTransformerLightning:
    """
    Crée un module MusicTransformerLightning avec les hyperparamètres donnés.
    
    Returns:
        MusicTransformerLightning: Module prêt pour l'entraînement
    """
    model = MusicTransformerLightning(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        dropout=dropout,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay
    )
    
    print(f"✓ MusicTransformerLightning créé")
    print(f"  Paramètres: {model.model.count_parameters():,}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Warmup steps: {warmup_steps}")
    
    return model


# Test du module
if __name__ == "__main__":
    """Test rapide du module Lightning."""
    
    print("🧪 Test du MusicTransformerLightning\n")
    
    # Créer le modèle
    model = create_model(
        vocab_size=500,
        d_model=256,
        num_heads=8,
        num_layers=4,
        d_ff=1024,
        learning_rate=1e-4,
        warmup_steps=1000
    )
    
    # Créer un batch de test
    batch = {
        'input_ids': torch.randint(0, 500, (4, 128)),
        'attention_mask': torch.ones(4, 128),
        'labels': torch.randint(0, 500, (4, 128))
    }
    
    print(f"\n🔄 Test training_step:")
    loss = model.training_step(batch, 0)
    print(f"  Loss: {loss.item():.4f}")
    
    print(f"\n✓ Test réussi!")