# TechNova Partners : identifier les causes d'attrition

Projet de data science (classification supervisée) : à partir de trois extraits RH d'une ESN fictive,
TechNova Partners, on cherche à comprendre pourquoi les employés démissionnent.

1. **Analyse exploratoire** des trois fichiers et du fichier central issu de leur jointure.
2. **Préparation des données** : nettoyage, création de features, encodage, corrélations.
3. **Modélisation** : modèle étalon (Dummy), régression logistique, Random Forest, Gradient Boosting.
4. **Déséquilibre des classes** (16 % de départs) : validation croisée stratifiée, pondération des classes,
   SMOTE et sous-échantillonnage comparés, seuil de décision choisi sur la courbe précision-rappel.
5. **Fine-tuning** (RandomizedSearchCV) et **interprétation** : importance native, permutation importance,
   valeurs de Shapley (SHAP : beeswarm, scatter, waterfall).

## Contenu du dépôt

| Fichier | Rôle |
|---|---|
| `attrition_technova.ipynb` | Le notebook complet, déjà exécuté |
| `pyproject.toml` / `uv.lock` | Versions de Python compatibles et dépendances (gérées avec uv) |
| `data/` | Les trois extraits fournis pour l'exercice (données fictives) |
| `figures/` | Les graphiques produits par le notebook |

## Installation et exécution

Prérequis : [uv](https://docs.astral.sh/uv/). Python 3.11 à 3.13 (uv installe la version épinglée, 3.12, si besoin).

```bash
git clone https://github.com/EXOSTE/attrition-technova.git
cd attrition-technova
uv sync                      # crée .venv et installe les dépendances exactes de uv.lock
uv run jupyter lab           # puis ouvrir attrition_technova.ipynb
```

Pour réexécuter tout le notebook en ligne de commande (environ 1 minute) :

```bash
uv run jupyter nbconvert --to notebook --execute --inplace attrition_technova.ipynb
```

## Résultats principaux

- Modèle retenu : Gradient Boosting pondéré, optimisé en validation croisée stratifiée (critère : PR AUC),
  seuil choisi pour maximiser le F2 (le rappel compte double : rater un départ coûte plus qu'une fausse alerte).
- Sur le jeu de test : environ 3 départs sur 4 détectés, précision ≈ 0,39 (contre 0,16 au hasard),
  ROC AUC ≈ 0,81. Performance équivalente à une régression logistique pondérée.
- Facteurs de départ les plus forts (SHAP) : heures supplémentaires, absence de participation au PEE
  (confondue avec le célibat dans les données), âge inférieur à ~33 ans, revenu bas, satisfaction de
  l'environnement de travail à 1.
