# Phase 05 — Recherche Vectorielle & Base de Données Vectorielles avec PostgreSQL et pgvector

Bienvenue dans le laboratoire pratique de recherche vectorielle dense (*Dense Retrieval Lab*). Ce tutoriel explore les fondements mathématiques, les structures d'indexation géométriques, l'optimisation des requêtes SQL et les compromis de performance entre la recherche exacte (**Exact KNN**) et la recherche approximative (**ANN** avec **HNSW** et **IVFFlat**).

---

## 1. Fondements Mathématiques des Métriques de Distance

Dans un espace vectoriel dense de dimension $d$ (ex. $d = 384$ pour `all-MiniLM-L6-v2`), la similarité sémantique entre deux vecteurs $\mathbf{u}, \mathbf{v} \in \mathbb{R}^d$ se quantifie au moyen d'opérateurs géométriques.

### A. Distance Cosinus (*Cosine Distance*)
La distance cosinus mesure l'angle $\theta$ formé par deux vecteurs, indépendamment de leur norme euclidienne :
$$\cos(\theta) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2} = \frac{\sum_{i=1}^d u_i v_i}{\sqrt{\sum_{i=1}^d u_i^2} \sqrt{\sum_{i=1}^d v_i^2}}$$

Dans `pgvector`, l'opérateur `<=>` calcule la **distance cosinus** :
$$d_{\cos}(\mathbf{u}, \mathbf{v}) = 1 - \cos(\theta) \in [0, 2]$$

- Si $\mathbf{u} = \mathbf{v}$ : $d_{\cos} = 0.0 \implies \text{similarité} = 1.0$
- Si $\mathbf{u} \perp \mathbf{v}$ (orthogonaux) : $d_{\cos} = 1.0 \implies \text{similarité} = 0.0$
- Si $\mathbf{u} = -\mathbf{v}$ (opposés) : $d_{\cos} = 2.0 \implies \text{similarité} = -1.0$

### B. Distance Euclidienne $L_2$ (*Euclidean Distance*)
La distance euclidienne mesure la distance physique en ligne droite :
$$d_{L2}(\mathbf{u}, \mathbf{v}) = \|\mathbf{u} - \mathbf{v}\|_2 = \sqrt{\sum_{i=1}^d (u_i - v_i)^2}$$

Dans `pgvector`, l'opérateur `<->` calcule cette distance $d_{L2} \in [0, \infty)$.
Pour la convertir en score de similarité borné dans $(0, 1]$, on utilise la transformation harmonique :
$$s_{L2} = \frac{1}{1 + d_{L2}}$$

### C. Produit Scalaire (*Inner Product / Dot Product*)
Le produit scalaire pondère l'alignement et la magnitude :
$$\langle \mathbf{u}, \mathbf{v} \rangle = \mathbf{u} \cdot \mathbf{v} = \sum_{i=1}^d u_i v_i$$

Dans `pgvector`, l'opérateur `<#>` retourne l'opposé du produit scalaire :
$$d_{\text{ip}}(\mathbf{u}, \mathbf{v}) = - (\mathbf{u} \cdot \mathbf{v})$$
*Pourquoi cette négation ?* Parce qu'en SQL, une clause `ORDER BY ... ASC LIMIT k` recherche les valeurs minimales. Minimiser $- (\mathbf{u} \cdot \mathbf{v})$ revient exactement à maximiser le produit scalaire $\mathbf{u} \cdot \mathbf{v}$.

### D. Théorème d'Équivalence sous Normalisation $L_2$
Si tous les vecteurs sont préalablement normalisés à la norme unité ($\|\mathbf{u}\|_2 = \|\mathbf{v}\|_2 = 1.0$), alors :
$$\|\mathbf{u} - \mathbf{v}\|_2^2 = \|\mathbf{u}\|_2^2 + \|\mathbf{v}\|_2^2 - 2 (\mathbf{u} \cdot \mathbf{v}) = 2 - 2 \cos(\theta) = 2 \, d_{\cos}(\mathbf{u}, \mathbf{v})$$
$$d_{\cos}(\mathbf{u}, \mathbf{v}) = 1 - \mathbf{u} \cdot \mathbf{v} = \frac{1}{2} \|\mathbf{u} - \mathbf{v}\|_2^2$$

> [!NOTE]
> Pour des embeddings normalisés $L_2$, les trois métriques produisent un **classement top-k rigoureusement identique** !

---

## 2. Recherche Exacte vs Recherche Approximative (ANN)

```
             ┌──────────────────────────────────────────────────────────────┐
             │               Recherche Vectorielle dans pgvector             │
             └──────────────────────────────┬───────────────────────────────┘
                                            │
                    ┌───────────────────────┴───────────────────────┐
                    ▼                                               ▼
         Exact KNN (Force Brute)                           ANN (Approximation)
      - Complexité O(N · d)                            - Complexité O(log N) ou O(probes · N/k)
      - Parcours séquentiel (Seq Scan)                 - Index Scan (graphes ou clusters)
      - Rappel: 100% (Vérité terrain)                  - Rappel: 80% à 99% réglable
      - Idéal pour: N < 50 000                         - Idéal pour: N > 50 000
```

---

## 3. L'Index HNSW (*Hierarchical Navigable Small World*)

HNSW est un graphe de proximité multicouche inspiré des *skip-lists*.

```
Couche 2 (Épars)    (A) ---------------------------------------------> (F)
                     │                                                   │
Couche 1 (Moyen)    (A) --------------> (C) ----------> (D) -----------> (F)
                     │                   │               │               │
Couche 0 (Dense)    (A) <-> (B) <-> (C) <-> (D) <-> (E) <-> (F) <-> (G)
```

### Paramètres Clés :
1. **$M$ (`m`, défaut: 16, plage: 4–64)** : Nombre maximal de liens bidirectionnels par nœud.
   - Augmenter $M$ améliore la connectivité et le rappel sur les espaces à haute courbure, au prix d'une empreinte mémoire accrue.
2. **$ef_{\text{construction}}$ (`ef_construction`, défaut: 64, plage: 16–512)** : Taille de la file de priorité lors de la construction du graphe.
   - Détermine le temps de construction et la qualité topologique du graphe.
3. **$ef_{\text{search}}$ (`hnsw.ef_search`, défaut: 40, plage: 10–500)** : Paramètre dynamique à l'exécution de la requête.
   - Détermine la profondeur d'exploration locale lors du routage glouton (*greedy routing*).
   - Plus $ef_{\text{search}}$ est élevé, plus le rappel se rapproche de 100% au prix d'une latence légèrement supérieure.

### Syntaxe SQL pgvector :
```sql
-- Création de l'index HNSW
CREATE INDEX idx_chunks_hnsw_cos ON chunk_embeddings
USING hnsw (((vector)::vector(384)) vector_cosine_ops)
WITH (m = 16, ef_construction = 64)
WHERE vector_dimension = 384 AND vector IS NOT NULL;

-- Réglage transactionnel de la profondeur de recherche
SET LOCAL hnsw.ef_search = 100;

-- Requête de recherche sémantique
SELECT id, text_snippet, 1 - (vector <=> CAST(:query_vec AS vector(384))) AS similarity
FROM chunk_embeddings
WHERE vector_dimension = 384
ORDER BY vector <=> CAST(:query_vec AS vector(384))
LIMIT 5;
```

---

## 4. L'Index IVFFlat (*Inverted File Flat*)

IVFFlat partitionne l'espace vectoriel en $k$ cellules de Voronoï via un algorithme de classification k-means.

```
       Cellule 1                 Cellule 2                 Cellule 3
  ┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
  │   •  •          │       │       •     •   │       │   •             │
  │      [C1]  •    │       │   •   [C2]      │       │     [C3]   •    │
  │   •             │       │         •       │       │   •      •      │
  └─────────────────┘       └─────────────────┘       └─────────────────┘
           ▲                         ▲                         ▲
           │                         │                         │
      Liste inversée 1          Liste inversée 2          Liste inversée 3
```

### Paramètres Clés :
1. **`lists` (défaut: 10–1000)** : Nombre de centroïdes de Voronoï créés lors de la phase d'entraînement.
   - Règle empirique : $\text{lists} \approx \sqrt{N}$ pour de petites collections ($N < 100\,000$) ou $N / 1000$ pour de grands corpus.
2. **`probes` (`ivfflat.probes`, défaut: 1, plage: 1–100)** : Nombre de centroïdes les plus proches explorés lors de la requête.
   - Si $\text{probes} = 1$, seuls les vecteurs du centroïde le plus proche sont évalués.
   - Si $\text{probes} = \text{lists}$, la recherche redevient équivalente à une force brute exacte.

### Syntaxe SQL pgvector :
```sql
-- Création de l'index IVFFlat
CREATE INDEX idx_chunks_ivfflat_cos ON chunk_embeddings
USING ivfflat (((vector)::vector(384)) vector_cosine_ops)
WITH (lists = 10)
WHERE vector_dimension = 384 AND vector IS NOT NULL;

-- Réglage transactionnel des centroïdes explorés
SET LOCAL ivfflat.probes = 5;

-- Recherche vectorielle filtrée
SELECT id, text_snippet, 1 - (vector <=> CAST(:query_vec AS vector(384))) AS similarity
FROM chunk_embeddings
WHERE vector_dimension = 384
ORDER BY vector <=> CAST(:query_vec AS vector(384))
LIMIT 5;
```

---

## 5. Inspection du Plan d'Exécution avec `EXPLAIN ANALYZE`

Dans PostgreSQL, la présence d'un index vectoriel **ne garantit pas** que l'optimiseur l'utilisera systématiquement.

### Règle du Coût de l'Optimiseur :
Sur les petites tables ($N < 100$ lignes), PostgreSQL choisit presque toujours un parcours séquentiel (`Seq Scan`) car lire un ou deux blocs disque de 8 ko en mémoire tampon est considérablement plus rapide que de charger les pages d'un graphe HNSW ou d'initialiser les centroïdes IVFFlat !

Pour forcer et observer le plan d'indexation dans un benchmark :
```sql
EXPLAIN (ANALYZE, FORMAT JSON)
SELECT id FROM chunk_embeddings
WHERE vector_dimension = 384
ORDER BY vector <=> CAST(:query_vec AS vector(384))
LIMIT 5;
```

Sortie typique lors de l'utilisation de l'index :
```text
-> Index Scan using idx_hnsw_cos_384_... on chunk_embeddings
   Order By: ((vector)::vector(384) <=> $1)
   Planning Time: 0.437 ms
   Execution Time: 0.101 ms
```

---

## 6. Gestion des Préfixes Asymétriques (ex. Multilingual-E5)

Certains modèles d'embedding modernes entraînés de façon asymétrique (comme `intfloat/multilingual-e5-small`) exigent des préfixes explicites pour séparer l'espace sémantique des questions et des documents :

- **Requête utilisateur** : Préfixée avec `"query: "` (ex. `"query: Quels sont les principes de Raft ?"`)
- **Fragments de document** : Préfixés avec `"passage: "` (ex. `"passage: Raft est un algorithme de consensus..."`)

> [!IMPORTANT]
> Oublier d'ajouter `"query: "` à la requête avec un modèle E5 dégrade le rappel et le score de similarité de plus de 30% car la projection latente de la question ne s'aligne pas avec l'espace des passages cibles !

---

## 7. Pièges Courants (*Common Pitfalls*)

1. **Mélanger des espaces d'embeddings différents** :
   Calculer la distance entre un vecteur issu de `all-MiniLM-L6-v2` et un vecteur issu d'un autre modèle produit des résultats sémantiquement aberrants. L'isolation stricte par `model_id` et `vector_dimension` est impérative.
2. **Utiliser le produit scalaire sur des vecteurs non normalisés sans justification** :
   Sur des vecteurs bruts, les passages très longs ou redondants peuvent avoir une norme artificielle élevée, faussant le classement au détriment de la pertinence sémantique.
3. **Créer un index IVFFlat sur une table vide ou trop petite** :
   Si $N < \text{lists}$, la classification k-means échoue ou génère des centroïdes dégénérés, détruisant le rappel ($< 10\%$).
4. **Oublier le casting de dimension explicite dans PostgreSQL** :
   Une colonne `vector` sans dimension déclarée (`Vector()`) requiert l'expression `((vector)::vector(384))` pour accepter un index HNSW ou IVFFlat.

---

## 8. Exercices Pratiques

### Exercice 1 : Impact de `ef_search` sur le Rappel HNSW
1. Rendez-vous dans l'onglet **Algorithm Comparison**.
2. Cliquez sur **Generate 300 Synthetic Vectors** pour disposer d'un corpus de test suffisant.
3. Créez un index HNSW avec $M=16, ef_{\text{construction}}=64$.
4. Lancez le benchmark avec $ef_{\text{search}} = 5$, puis avec $ef_{\text{search}} = 40$, puis avec $ef_{\text{search}} = 150$.
5. Observez l'évolution de la courbe de Rappel@K et de la latence en millisecondes.

### Exercice 2 : Filtrage par Métadonnées
1. Dans l'onglet **Semantic Search**, tapez une question.
2. Activez le panneau **Metadata Filters** et filtrez par `page_number = 1`.
3. Vérifiez dans les résultats que seuls les chunks appartenant à la page 1 sont retournés.
4. Inspectez le plan `EXPLAIN ANALYZE` pour vérifier comment PostgreSQL applique le prédicat de filtrage.
