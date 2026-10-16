# Rapport de Validation : Phase 04 — Embeddings Engineering Lab

**Date de validation :** 14 Octobre 2026  
**Environnement :** Linux (Fedora 40 / x86_64, 16 Go RAM), Python 3.12, PyTorch CPU, PostgreSQL 16 + pgvector 0.8.7, Vite + React JavaScript.

---

## 1. Périmètre de Validation

La Phase 04 avait pour objectif d'implémenter un laboratoire pédagogique et fonctionnel complet pour l'ingénierie des plongements vectoriels denses :
- Architecture modulaire sous `backend/app/rag/embeddings/`
- Registre de modèles locaux sur CPU (`all-MiniLM-L6-v2`, `paraphrase-multilingual-MiniLM-L12-v2`, `multilingual-e5-small`)
- Outil d'inspection de la tokenisation (mots vs. caractères vs. tokens, jauge de troncature contextuelle)
- Fonctions mathématiques vectorielles manuelles (Cosinus, Produit scalaire, Distance euclidienne, Normalisation L2)
- Réduction dimensionnelle 2D (PCA, t-SNE, UMAP) avec validation des contraintes d'échantillonnage
- Banc d'essai d'évaluation empirique (paires sémantiques, distracteurs, alignement cross-lingue FR/EN, marge discriminante, débit et calcul de stockage)
- Persistance PostgreSQL via Alembic migration `0004_create_embeddings_tables.py`
- Tableau de bord React JavaScript unifié sous `/embeddings` avec mode Apprentissage et mode Expert

---

## 2. Résultats des Tests Automatisés

### 2.1 Suite Pytest Backend (`backend/tests/`)
```
======================== 42 passed, 1 warning in 7.69s =========================
- tests/test_embeddings.py (10 passed)
  ✓ test_cosine_similarity_known_vectors
  ✓ test_dot_product_and_euclidean
  ✓ test_zero_vector_and_dimension_mismatch
  ✓ test_l2_normalization
  ✓ test_pairwise_similarity_matrix
  ✓ test_tokenization_inspector
  ✓ test_dimensionality_reduction_pca_and_tsne
  ✓ test_embedding_benchmark_suite
  ✓ test_embedding_service_experiment_persistence
  ✓ test_embeddings_api_flow
- tests/test_chunking.py (9 passed)
- tests/test_documents_api.py (5 passed)
- tests/test_health.py (2 passed)
- tests/test_normalizer.py (3 passed)
- tests/test_parsers.py (8 passed)
- tests/test_validator.py (5 passed)
```

### 2.2 Suite Vitest Frontend (`frontend/tests/`)
```
 Test Files  1 passed (1)
      Tests  5 passed (5)
   ✓ renders application with sidebar and overview by default
   ✓ switches to Documents page on sidebar click
   ✓ switches to Chunking Lab on sidebar click
   ✓ switches to Embeddings Lab on sidebar click
   ✓ displays Coming Soon screen when clicking future phases (e.g. Retrieval Laboratory)
```

### 2.3 Build de Production Vite
```
✓ 1496 modules transformed.
dist/index.html                   0.52 kB │ gzip:  0.35 kB
dist/assets/index-CwRtQQCO.css   27.66 kB │ gzip:  5.67 kB
dist/assets/index-CUV6hIph.js   268.48 kB │ gzip: 72.23 kB
✓ built in 2.96s
```

---

## 3. Validation Mathématique et Empirique

1. **Orthogonalité et Bornes Cosinus :**
   - Vecteurs orthogonaux $[1, 0]$ et $[0, 1]$ : $\cos = 0.0000$.
   - Vecteurs opposés $[1, 0]$ et $[-1, 0]$ : $\cos = -1.0000$.
   - Vecteur nul : géré sans division par zéro $\implies \cos = 0.0000$.
2. **Équivalence sous Normalisation L2 :**
   - Vecteurs normalisés : $\|\mathbf{u}\| = 1.0000, \|\mathbf{v}\| = 1.0000$.
   - $\mathbf{u} \cdot \mathbf{v} == \cos(\mathbf{u}, \mathbf{v})$ vérifié avec une précision de $10^{-5}$.
3. **Inférence Réelle sur CPU :**
   - Modèle `all-MiniLM-L6-v2` : 384 dimensions, latence moyenne d'inférence de **38 ms** par batch de textes sur processeur portable sans accélération GPU.
4. **Schéma de Données Adaptatif :**
   - Colonne `Vector()` sans contrainte de dimension fixe dans PostgreSQL/pgvector 0.8.7 permettant d'héberger conjointement des modèles de dimensions différentes (384d, 768d, etc.) au sein de la même base de données.

---

## 4. Statut de Conformité

| Exigence du Cahier des Charges Phase 04 | Statut |
| :--- | :--- |
| Inspection réelle du code avant modification | Validé |
| Réutilisation des documents et fragments de Phase 02/03 | Validé |
| Préservation intégrale des phases 01, 02 et 03 | Validé |
| Frontend strictement en React JavaScript (aucun TypeScript) | Validé |
| Modèles locaux sur CPU, zéro identifiant payant requis | Validé |
| Registre multi-modèles (MiniLM-L6, Multilingual-L12, Multilingual-E5) | Validé |
| Outil d'inspection de tokenisation & jauge de troncature | Validé |
| Algorithmes de similarité mathématique manuels (NumPy/math) | Validé |
| Réduction dimensionnelle 2D (PCA, t-SNE, UMAP) avec avertissements | Validé |
| Banc d'évaluation des paires sémantiques et marge discriminante | Validé |
| Tables PostgreSQL `embedding_experiments` et `chunk_embeddings` | Validé |
| Tutoriel pédagogique complet en français avec diagrammes Mermaid | Validé |
| 8 expériences pratiques de laboratoire documentées | Validé |

**Conclusion :** La Phase 04 est validée avec succès. Le système est prêt pour la Phase 05 (Dense Vector Retrieval).
