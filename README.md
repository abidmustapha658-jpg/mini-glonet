# Robust-GLONET

> Prévision océanique par apprentissage profond et étude de robustesse face aux perturbations des données.

## Description

Robust-GLONET est un projet académique développé dans le cadre d'un travail d'étude portant sur l'intelligence artificielle, la prévision océanique et la cybersécurité appliquée aux modèles de machine learning.

Le projet a pour objectif de développer une version simplifiée d'un pipeline de prévision océanique inspiré de GLONET, puis d'étudier sa robustesse face à différentes perturbations des données d'entrée et d'entraînement.

Les travaux portent notamment sur :

- le développement d'un pipeline PyTorch ;
- la préparation des données ;
- l'entraînement de modèles convolutionnels ;
- l'évaluation des performances ;
- les attaques contrôlées sur les données ;
- la détection d'anomalies ;
- l'entraînement robuste.

---

## Objectifs

Le projet vise à répondre aux questions suivantes :

1. Peut-on prédire un champ océanique futur à partir d'observations passées ?
2. Quel est l'impact d'une perturbation des données sur les performances du modèle ?
3. Peut-on détecter automatiquement des données anormales ?
4. Un modèle entraîné avec des données perturbées devient-il plus robuste ?
5. Quel est l'impact des perturbations lors d'un fine-tuning régional ?

---

## Fonctionnalités prévues

### Intelligence artificielle

- Prétraitement des données.
- Création des `Dataset` et `DataLoader`.
- Baseline de persistance.
- `MiniGlonetCNN`, notre CNN résiduel principal.
- Fine-tuning régional.

### Robustesse et cybersécurité

- Injection de bruit.
- Perturbations spatiales.
- Perturbations temporelles.
- Attaques adversariales (FGSM).
- Empoisonnement de données.
- Détection d'anomalies.
- Autoencodeur de débruitage.
- Entraînement adversarial.

---

## Architecture du projet

```text
Données
   ↓
Prétraitement
   ↓
Dataset PyTorch
   ↓
Mini-GLONET
   ↓
Prévision
   ↓
Perturbation des données
   ↓
Détection
   ↓
Défense
   ↓
Évaluation de robustesse
```

---

## Structure du dépôt

```text
mini-glonet/
│
├── data/
├── notebooks/
├── reports/
├── src/
│   └── mini_glonet/
│       ├── datasets/
│       ├── models/
│       ├── training/
│       ├── attacks/
│       ├── defenses/
│       └── utils/
│
├── tests/
├── README.md
└── requirements.txt
```

---

## Installation

```bash
git clone https://github.com/hassannassiri181-cmyk/mini-glonet.git

cd mini-glonet

python -m venv .venv

source .venv/bin/activate
```

Installation des dépendances :

```bash
pip install -r requirements.txt
```

---

## Exécution

Lancer un entraînement :

```bash
python train.py
```

Exécuter une expérience de robustesse :

```bash
python experiments/run_attack.py
```

Lancer la détection d'anomalies :

```bash
python experiments/anomaly_detection.py
```

---

## Métriques utilisées

- RMSE
- MSE
- Precision
- Recall
- F1-Score
- Temps d'entraînement
- Temps d'inférence

---

## Équipe du projet

| Nom | Rôle |
|-----|------|
| Elhassane Enassiri | Étudiant |
| Mustapha ABID | Collaborateur technique |
| Pr. Kandoussi Asmae | Encadrante académique |

---

## Organisation Git

Branches principales :

```text
main
feature/cnn-baseline
attack/fgsm
attack/temporal
attack/poisoning
defense/anomaly-detector
defense/adversarial-training
experiment/fine-tuning
```

Processus de développement :

1. Créer une branche.
2. Développer la fonctionnalité.
3. Effectuer les tests.
4. Ouvrir une Pull Request.
5. Fusionner après validation.

---

## Livrables

- Pipeline Mini-GLONET fonctionnel.
- Module de perturbations contrôlées.
- Méthode de détection ou de défense.
- Rapport de résultats.
- Démonstration du pipeline complet.

---

## Encadrement

Ce projet est réalisé dans un cadre académique.

- Encadrante académique : **Pr. Kandoussi Asmae**
- Collaborateur technique : **Mustapha ABID**

---

## Licence

Projet académique à but pédagogique.

## Baseline CNN results

The reproducible evaluation report and figures are available in
[reports/baseline-cnn](reports/baseline-cnn/README.md).
