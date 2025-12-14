import gradio as gr
import torch
from pathlib import Path
from miditok import REMI
import sys
import numpy as np
import pickle

# Ajouter le dossier src au path
sys.path.append(str(Path(__file__).parent / "src"))

from models.lightning_module import MusicTransformerLightning

# CONFIGURATION

CHECKPOINT_PATH = "src/models/music-transformer-final.ckpt"
TOKENIZER_PATH = "src/data/processed/tokenizer/tokenizer.json"
VAL_SEQUENCES_PATH = "src/data/processed/validation_sequences.pkl"
OUTPUT_DIR = Path("generated_music")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Créer le dossier de sortie
OUTPUT_DIR.mkdir(exist_ok=True)

# Charger le tokenizer

tokenizer = REMI(params=str(TOKENIZER_PATH))

vocab_size = len(tokenizer)

# Charger les séquences de validation

with open(VAL_SEQUENCES_PATH, 'rb') as f:
    val_sequences = pickle.load(f)

model = MusicTransformerLightning.load_from_checkpoint(
    CHECKPOINT_PATH,
    vocab_size=vocab_size,
    map_location=DEVICE
)
model.to(DEVICE)
model.eval()
model.freeze()

# FONCTION DE GÉNÉRATION

def generate_music(
    prompt_length,
    max_length,
    temperature,
    top_k,
    top_p,
    seed
):
    try:
        # Définir le seed pour reproductibilité
        if seed >= 0:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Créer le prompt
        if prompt_length == 0:
            # Juste BOS
            try:
                bos_token = tokenizer.vocab.get('BOS_None', 1)
            except:
                bos_token = 1
            prompt_tokens = [bos_token]
            prompt_source = "BOS uniquement"
        else:
            # Prendre une séquence aléatoire du validation set
            seed_sequence = val_sequences[np.random.randint(0, len(val_sequences))]
            prompt_tokens = seed_sequence[:prompt_length]
            prompt_source = f"{prompt_length} tokens du validation set"
        
        actual_prompt_length = len(prompt_tokens)
        
        print(f"Nouvelle génération")
        print(f"Prompt: {prompt_source}")
        print(f"Longueur prompt: {actual_prompt_length} tokens")
        print(f"Premiers tokens: {prompt_tokens[:min(10, len(prompt_tokens))]}")
        print(f"Max length: {max_length} tokens")
        print(f"Temperature: {temperature}")
        print(f"Top-K: {top_k}")
        print(f"Top-P: {top_p}")
        print(f"Seed: {seed if seed >= 0 else 'random'}")
        
        # Convertir en tensor
        prompt_tensor = torch.tensor(
            [prompt_tokens],
            dtype=torch.long,
            device=DEVICE
        )
        
        # Générer
        with torch.no_grad():
            generated_tensor = model.generate(
                prompt=prompt_tensor,
                max_length=min(max_length, 1024),
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )
        
        # Convertir en liste
        generated_tokens = generated_tensor[0].cpu().tolist()
        total_length = len(generated_tokens)
        generated_count = total_length - actual_prompt_length
        
        print(f"Génération terminée: {total_length} tokens")
        
        # Séparer prompt et génération
        prompt_part = generated_tokens[:actual_prompt_length]
        generated_part = generated_tokens[actual_prompt_length:]
        
        # Créer les fichiers MIDI
        try:
            # MIDI complet
            midi_full_wrapped = [generated_tokens]
            midi_full = tokenizer.decode(midi_full_wrapped)
            full_path = str(OUTPUT_DIR / "output_FULL.mid")
            midi_full.dump_midi(full_path)
            
            # MIDI prompt seulement
            if actual_prompt_length > 1:
                midi_prompt_wrapped = [prompt_part]
                midi_prompt = tokenizer.decode(midi_prompt_wrapped)
                prompt_path = str(OUTPUT_DIR / "output_PROMPT.mid")
                midi_prompt.dump_midi(prompt_path)
            else:
                prompt_path = None
            
            # MIDI génération seulement
            midi_generated_wrapped = [generated_part]
            midi_generated = tokenizer.decode(midi_generated_wrapped)
            generated_path = str(OUTPUT_DIR / "output_GENERATED.mid")
            midi_generated.dump_midi(generated_path)
            
        except Exception as e:
            print(f"Erreur de décodage: {e}")
            # Fallback
            midi_full = tokenizer.tokens_to_midi([generated_tokens])
            full_path = str(OUTPUT_DIR / "output_FULL.mid")
            midi_full.dump(full_path)
            
            if actual_prompt_length > 1:
                midi_prompt = tokenizer.tokens_to_midi([prompt_part])
                prompt_path = str(OUTPUT_DIR / "output_PROMPT.mid")
                midi_prompt.dump(prompt_path)
            else:
                prompt_path = None
            
            midi_generated = tokenizer.tokens_to_midi([generated_part])
            generated_path = str(OUTPUT_DIR / "output_GENERATED.mid")
            midi_generated.dump(generated_path)
        
        # Créer le texte d'information
        info = f"""Génération terminée !

Statistiques :

- Total        : {total_length} tokens
- Prompt       : {actual_prompt_length} tokens ({actual_prompt_length/total_length*100:.1f}%)
- Généré       : {generated_count} tokens ({generated_count/total_length*100:.1f}%)

Paramètres utilisés :

- Prompt       : {prompt_source}
- Temperature  : {temperature}
- Top-K        : {top_k}
- Top-P        : {top_p}
- Seed         : {seed if seed >= 0 else 'random'}
- Device       : {DEVICE}

Fichiers créés :

- output_FULL.mid       (prompt + génération)
- output_PROMPT.mid     (prompt uniquement)
- output_GENERATED.mid  (génération pure)
"""
        
        print(info)
        
        return full_path, prompt_path, generated_path, info
    
    except Exception as e:
        error_msg = f"Erreur lors de la génération :\n\n{str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return None, None, None, error_msg


# INTERFACE GRADIO

with gr.Blocks(title="Music Transformer") as app:
    
    gr.Markdown("""
    # Music Transformer - Générateur de MIDI
    """)
    
    with gr.Row():
        # Colonne gauche : Paramètres
        with gr.Column(scale=1):
            gr.Markdown("### Paramètres de génération")
            
            prompt_length = gr.Slider(
                minimum=0,
                maximum=128,
                value=10,
                step=1,
                label="Longueur du prompt",
                info="0 = BOS seul, Entre 1 et 128 pour le promp de départ"
            )
            
            max_length = gr.Slider(
                minimum=64,
                maximum=1024,
                value=512,
                step=64,
                label="Longueur totale (tokens)",
                info="Nombre total de tokens à générer"
            )
            
            temperature = gr.Slider(
                minimum=0.1,
                maximum=2.0,
                value=1.0,
                step=0.1,
                label="Temperature",
            )
            
            with gr.Row():
                top_k = gr.Slider(
                    minimum=1,
                    maximum=100,
                    value=50,
                    step=1,
                    label="Top-K"
                )
                
                top_p = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.95,
                    step=0.05,
                    label="Top-P"
                )
            
            seed = gr.Slider(
                minimum=-1,
                maximum=9999,
                value=-1,
                step=1,
                label="Seed",
                info="-1 = aléatoire, ≥0 = reproductible"
            )
            
            generate_btn = gr.Button(
                "Générer un MIDI",
                variant="primary",
                size="lg"
            )
        
        # Colonne droite : Résultats
        with gr.Column(scale=1):
            gr.Markdown("### Résultats")
            
            info_output = gr.Textbox(
                label="Informations",
                lines=22,
                max_lines=25
            )
            
            midi_full = gr.File(
                label="MIDI Complet (prompt + génération)",
                file_types=[".mid"]
            )
            
            midi_prompt = gr.File(
                label="MIDI Prompt uniquement",
                file_types=[".mid"]
            )
            
            midi_generated = gr.File(
                label="MIDI Généré uniquement",
                file_types=[".mid"]
            )
    
    # Connecter le bouton
    generate_btn.click(
        fn=generate_music,
        inputs=[prompt_length, max_length, temperature, top_k, top_p, seed],
        outputs=[midi_full, midi_prompt, midi_generated, info_output]
    )
    

# ====================================
# LANCEMENT
# ====================================

if __name__ == "__main__":
    print("Lancement de l'application Gradio")
    print(f"Modèle : {CHECKPOINT_PATH}")
    print(f"Tokenizer : {TOKENIZER_PATH}")
    print(f"Validation : {VAL_SEQUENCES_PATH}")
    print(f"Sortie : {OUTPUT_DIR}")
    print(f"Device : {DEVICE}")
    
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
        show_error=True
    )