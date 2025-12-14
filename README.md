# Music Transformer – Projet Deep Learning

## Objectif
Ce projet implémente un modèle de Deep Learning de type **Transformer** pour la génération de musique symbolique à partir de fichiers **MIDI**.

Le modèle est entraîné avec **PyTorch Lightning** et permet de générer automatiquement des séquences musicales.

---

## Contenu du projet
- Code source complet du modèle (PyTorch / PyTorch Lightning)
- Des notebooks pour l’exploration des données et la visualisation
- Une application **Gradio** pour tester la génération musicale

---

## Modèle pré-entraîné
Le modèle final entraîné n’est pas inclus dans ce dépôt en raison de sa taille (>100 MB).

Il est disponible au téléchargement à l’adresse suivante :
[]

Après téléchargement, placer le fichier dans :
src/models/music-transformer-final.ckpt

## Lancer le projet
À la racine du projet :

```bash
pip install -r requirements.txt
python app_gradio.py
