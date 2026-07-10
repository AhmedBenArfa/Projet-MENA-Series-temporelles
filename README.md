# VaR MENA — Deep Learning vs modèles statistiques classiques

Comparaison de modèles de **Deep Learning** et de modèles **statistiques
classiques** pour la prédiction de la **Value-at-Risk (VaR)** de quatre
indices boursiers de la région **MENA** (Moyen-Orient / Afrique du Nord),
avec deux indices internationaux (CAC40, S&P500) utilisés comme simples
repères de contexte.

## 1. Objectif

L'hypothèse de départ du projet est que les modèles de Deep Learning
(en particulier un **LSTM**) surpassent les modèles statistiques classiques
pour prédire la VaR d'un indice MENA. Ce dépôt implémente un pipeline
complet — chargement des données, tests de stationnarité, 7 modèles de
prévision, calcul de VaR par simulation historique bootstrap, et
backtesting réglementaire — afin de trancher cette hypothèse sur des
données réelles.

**Résultat principal (honnête) :** l'hypothèse n'est **pas confirmée**.
Sur les 4 indices MENA testés à 99 % de confiance, c'est **GARCH** qui
l'emporte sur 3 indices (Tunindex, ADI, MASI), et **ARIMA** sur le
4ᵉ (TASI) — le LSTM ne devance ni GARCH ni ARIMA sur ADI, l'indice choisi
pour tester l'hypothèse. Voir la section [Résultats](#6-résultats) pour le
détail complet.

## 2. Données & indices

Les données sont des séries de prix quotidiens (format investing.com :
`Date, Price, Open, High, Low, Vol., Change %`), placées dans
`data (1)/data/` :

| Rôle | Indices | Fichiers |
|---|---|---|
| **MENA (étudiés)** | Tunindex, ADI (Abu Dhabi), MASI (Maroc), TASI (Arabie Saoudite) | `Tunindex.csv` / `TunindexTest.csv`, `ADI.csv` / `ADITest.csv`, `MASI.csv` / `MASITest.csv`, `TASI.csv` / `TASITest.csv` |
| **Repères internationaux** | CAC40, S&P500 | `CAC40.csv`, `S&P500.csv` |

Chaque indice MENA dispose d'un fichier d'entraînement et d'un fichier
`*Test.csv` séparé, utilisés pour le schéma train-once / walk-forward
(section 3). Les repères CAC40 et S&P500 n'ont **pas** de fichier de test
associé : ils ne servent que de contexte de marché (comparaison de
distribution de rendements, volatilité), sans backtest de VaR — faute de
données de test dans le même format.

## 3. Méthodologie

1. **Rendements log** : `r_t = 100 · ln(P_t / P_{t-1})`, calculés sur les
   prix de clôture (`src/tsvar/data.py::log_returns`).
2. **Tests de stationnarité** : ADF et KPSS sur les rendements
   (`src/tsvar/data.py::adf_test`, `kpss_test`), et décomposition
   saisonnière des prix pour l'analyse exploratoire.
3. **Schéma train-once + walk-forward one-step-ahead** : chaque modèle est
   entraîné **une seule fois** sur la période d'entraînement, puis produit
   des prévisions un pas en avant sur la période de test en avançant la
   fenêtre pas à pas (pas de ré-entraînement à chaque pas — cohérent pour
   les 7 modèles).
4. **7 modèles comparés** (`src/tsvar/run.py::MODELS`) :
   - Classiques : **ARIMA**, **SARIMA**, **GARCH**
   - Machine Learning : **Random Forest**, **XGBoost**
   - Deep Learning : **ANN**, **LSTM**
5. **VaR par Bootstrap Historical Simulation (BHS)** à deux niveaux de
   confiance, **95 %** et **99 %** (`src/tsvar/var.py::var_series`) :
   rééchantillonnage bootstrap des résidus standardisés de chaque modèle,
   combiné à la moyenne/volatilité prédites pour obtenir le quantile de VaR
   du jour.
6. **Backtesting** (`src/tsvar/backtest.py::backtest_summary`) :
   - Test de **Kupiec** (couverture non conditionnelle)
   - Test de **Christoffersen** (indépendance + couverture conditionnelle)
   - Classification en **zone Bâle** (verte / jaune / rouge)
   - Métriques ponctuelles **MAE** et **RMSE** sur la prévision de rendement

## 4. Architecture du dépôt

```
src/tsvar/            package Python
  data.py               chargement CSV, rendements log, ADF/KPSS
  classical.py          ARIMA, SARIMA (walk-forward)
  volatility.py         GARCH (walk-forward)
  ml.py                 Random Forest, XGBoost (walk-forward)
  deep.py               ANN, LSTM (walk-forward, PyTorch)
  var.py                VaR par Bootstrap Historical Simulation (BHS)
  backtest.py           Kupiec, Christoffersen, zones Bâle
  run.py                orchestration : run_index / run_all / MODELS
  plots.py              figures (rendements, ACF/PACF, décomposition, VaR...)

tests/                 suite pytest (22 tests, un module par composant)

scripts/
  generate_results.py    exécution complète : 4 indices x 7 modèles x 2 alphas
                          -> outputs/results.csv, outputs/best_per_index.csv
  build_narrative_nb.py  génère notebook/VaR_MENA.ipynb (utilise le package)
  build_complete_nb.py   génère notebook/VaR_MENA_complet.ipynb (autonome)
  build_deck.py          génère presentation/VaR_MENA.pptx à partir des CSV

notebook/
  VaR_MENA.ipynb          notebook narratif en français, importe tsvar
  VaR_MENA_complet.ipynb  notebook autonome (aucun import du package,
                          tout le code est réécrit en cellules), qui
                          recalcule sa propre comparaison complète

presentation/
  VaR_MENA.pptx           deck français (17 diapositives)

outputs/
  results.csv             résultat complet : 4 indices x 7 modèles x 2 alphas
  best_per_index.csv      meilleur modèle par indice (retenu à 99 %)
  figures/                figures générées (notebooks + deck)

data (1)/data/          fichiers CSV des indices (voir section 2)
```

## 5. Installation

Environnement testé : **Anaconda** (Python 3.13) sous Windows, interpréteur
appelé par chemin complet (non ajouté au `PATH`) :
`C:/Users/Mega-pc/anaconda3/python.exe`.

Dépendances (voir `requirements.txt` pour les versions exactes) :

```bash
"C:/Users/Mega-pc/anaconda3/python.exe" -m pip install -r requirements.txt
```

Le fichier `requirements.txt` épingle notamment :
`numpy`, `pandas`, `statsmodels`, `pmdarima`, `arch`, `scikit-learn`,
`xgboost`, `torch` (CPU), `matplotlib`, `seaborn`, `python-pptx`, `pytest`.

Si `torch` doit être installé séparément (roue CPU) :

```bash
"C:/Users/Mega-pc/anaconda3/python.exe" -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Le package `tsvar` **n'a pas besoin d'être installé** (pas de `setup.py` /
`pyproject.toml`) : `pytest.ini` place `src/` dans le `pythonpath` pour les
tests, et chaque script sous `scripts/` insère lui-même `src/` dans
`sys.path` au démarrage — aucune variable `PYTHONPATH` à définir
manuellement.

## 6. Comment exécuter

Toutes les commandes ci-dessous s'exécutent depuis la racine du dépôt
(`cd "C:/Users/Mega-pc/Documents/Exam Series temporelles"`).

**Lancer les tests :**

```bash
"C:/Users/Mega-pc/anaconda3/python.exe" -m pytest -q
```

→ 22 tests, tous verts (dernière vérification : 2026-07-10).

**Lancer le run complet** (recalcule `outputs/results.csv` et
`outputs/best_per_index.csv` pour les 4 indices MENA x 7 modèles x 2 alphas) :

```bash
"C:/Users/Mega-pc/anaconda3/python.exe" scripts/generate_results.py
```

**Ouvrir les notebooks** (dans Jupyter / VS Code) :

- `notebook/VaR_MENA.ipynb` — récit en français, s'appuie sur le package
  `tsvar` ; rejoue le pipeline complet en direct sur ADI (l'indice de
  l'hypothèse) et charge les résultats déjà calculés (`outputs/*.csv`) pour
  la comparaison sur les 4 indices.
- `notebook/VaR_MENA_complet.ipynb` — version **autonome**, sans aucun
  import de `src/tsvar` (tout le code est recopié dans les cellules) ;
  recalcule elle-même sa comparaison complète 4 indices x 7 modèles.

Pour régénérer un notebook à partir de son script bâtisseur puis l'exécuter
de bout en bout :

```bash
"C:/Users/Mega-pc/anaconda3/python.exe" scripts/build_narrative_nb.py
"C:/Users/Mega-pc/anaconda3/python.exe" -m jupyter nbconvert --to notebook --execute --inplace notebook/VaR_MENA.ipynb
```

(remplacer par `build_complete_nb.py` / `VaR_MENA_complet.ipynb` pour
l'autre notebook).

**Régénérer le deck PowerPoint** (relit `outputs/results.csv` et
`outputs/best_per_index.csv`, régénère ses propres figures `outputs/figures/deck_*.png`,
puis reconstruit `presentation/VaR_MENA.pptx`) :

```bash
"C:/Users/Mega-pc/anaconda3/python.exe" scripts/build_deck.py
```

## 7. Résultats

Résumé du meilleur modèle par indice MENA, à 99 % de confiance (source :
`outputs/best_per_index.csv`, sélection sur la p-value de Kupiec) :

| Indice   | Meilleur modèle | Zone Bâle | Kupiec p-value |
|----------|-----------------|-----------|----------------|
| Tunindex | **GARCH**       | verte     | 0.75           |
| ADI      | **GARCH**       | verte     | 0.82           |
| MASI     | **GARCH**       | verte     | 0.61           |
| TASI     | **ARIMA**       | verte     | 0.57           |

**Verdict de l'hypothèse :** l'hypothèse initiale du projet
(« le LSTM surpasse les modèles classiques sur ADI ») n'est **pas
confirmée** par les données : sur ADI, c'est GARCH qui obtient la
meilleure couverture de VaR à 99 %. Plus largement, GARCH domine sur 3 des
4 indices MENA et ARIMA sur le 4ᵉ ; les modèles de Deep Learning
(ANN, LSTM) et de Machine Learning (Random Forest, XGBoost) ne devancent
les modèles classiques sur aucun indice dans cette étude. En particulier,
**Random Forest sur-viole** systématiquement le seuil de VaR attendu
(taux d'exceptions observé trop élevé), ce qui en fait un mauvais candidat
pour une VaR fiable malgré de bonnes métriques ponctuelles (MAE/RMSE).

Le détail complet (7 modèles x 4 indices x 2 alphas, MAE, RMSE, taux
observé, p-values Kupiec/Christoffersen, zone Bâle) est disponible dans
`outputs/results.csv`, et commenté pas à pas dans les deux notebooks.

**Repères CAC40 / S&P500 :** ces deux indices ne sont utilisés que comme
contexte de marché (comparaison de la distribution des rendements et de la
volatilité avec les indices MENA) dans les notebooks — ils ne font l'objet
d'aucun backtest de VaR, faute de fichier de test dédié dans le jeu de
données (voir section 2).
