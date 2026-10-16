# Laboratoire Pratique 04 : Expériences d'Ingénierie des Embeddings

Ce guide propose 8 exercices pratiques et reproductibles à exécuter dans le tableau de bord de l'**Embeddings Explorer** (`/embeddings`) ou via l'API REST (`/api/v1/embeddings`).

---

## Expérience 1 : L'Équivalence Mathématique Produit Scalaire et Cosinus sous Normalisation L2

- **Objectif** : Vérifier empiriquement que le produit scalaire est équivalent à la similarité cosinus si et seulement si les vecteurs sont normalisés à la norme unitaire.
- **Entrées** :
  - Phrase A : *"FastAPI provides asynchronous request handling."*
  - Phrase B : *"Modern Python REST APIs benefit from async frameworks."*
- **Procédure** :
  1. Dans l'onglet **Similarity Lab**, saisir la Phrase A et la Phrase B.
  2. Activer la case *"Apply L2 Normalization"*.
  3. Noter les scores : `cosine_similarity` et `dot_product`.
  4. Désactiver la case *"Apply L2 Normalization"*.
  5. Ré-exécuter le calcul.
- **Observation Attendue** :
  - Avec normalisation : `dot_product == cosine_similarity` au 5e chiffre après la virgule.
  - Sans normalisation : le produit scalaire diverge fortement selon la norme brute des vecteurs, tandis que la similarité cosinus reste bornée dans $[-1.0, 1.0]$.

---

## Expérience 2 : L'Alignement Multilingue (Français vs. Anglais)

- **Objectif** : Comparer la capacité d'un modèle monolingue anglais versus un modèle multilingue sur des paires cross-lingues.
- **Entrées** :
  - Phrase A (FR) : *"L'architecture de microservices améliore la scalabilité et l'isolation des pannes."*
  - Phrase B (EN) : *"Microservice architecture improves system scalability and fault isolation."*
- **Procédure** :
  1. Calculer la similarité avec `sentence-transformers/all-MiniLM-L6-v2`.
  2. Calculer la similarité avec `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
- **Observation Attendue** :
  - Le modèle multilingue projette les deux phrases sémantiquement équivalentes à proximité immédiate (similarité cosinus $> 0.85$).
  - Le modèle monolingue anglais sous-estime l'équivalence sémantique en raison de la séparation lexicale des sous-mots.

---

## Expérience 3 : Détection de la Négation Subtile et Contre-Factualité

- **Objectif** : Évaluer la sensibilité sémantique du plongement face à une inversion de polarité avec forte ressemblance lexicale.
- **Entrées** :
  - Phrase A : *"The database query returned within five milliseconds with zero errors."*
  - Phrase B : *"The database query failed with multiple timeout errors after five minutes."*
- **Observation Attendue** :
  - Malgré un vocabulaire commun (*database, query, errors, five*), la similarité cosinus est nettement inférieure à une paire paraphrase.
  - Cela illustre pourquoi l'évaluation dense doit utiliser une marge discriminante ($\Delta = \mu_{\text{pos}} - \mu_{\text{neg}}$) plutôt qu'un seuil absolu.

---

## Expérience 4 : L'Impact de la Troncature des Séquences Longues

- **Objectif** : Observer le comportement du tokeniseur lorsque la longueur du texte dépasse la limite de contexte du modèle (`max_seq_length`).
- **Entrées** : Paragraphe technique de 400 mots.
- **Procédure** :
  1. Ouvrir l'onglet **Tokenizer Lab**.
  2. Sélectionner `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (limite : 128 tokens).
  3. Lancer la tokenisation.
- **Observation Attendue** :
  - L'interface affiche une jauge d'alerte et un avertissement explicite indiquant le nombre exact de tokens tronqués.
  - Les informations contenues après le 128e token ne participent pas au pooling final de l'embedding.

---

## Expérience 5 : Sensibilité aux Préfixes Asymétriques (`query:` vs. `passage:`)

- **Objectif** : Mesurer l'impact des préfixes d'instruction pour les modèles asymétriques comme `intfloat/multilingual-e5-small`.
- **Entrées** :
  - Requête : *"Comment configurer pgvector dans PostgreSQL ?"*
  - Passage : *"Pour activer l'extension vectorielle, exécuter CREATE EXTENSION vector;"*
- **Observation Attendue** :
  - Lorsque la requête est préfixée par `query: ` et le fragment par `passage: `, la similarité cosinus s'accroît significativement comparativement à un encodage sans préfixes.

---

## Expérience 6 : Comparaison de la Déformation Topologique (PCA vs. t-SNE)

- **Objectif** : Observer la différence entre projection globale linéaire (PCA) et projection locale stochastique (t-SNE) sur les chunks d'un document réel.
- **Procédure** :
  1. Ouvrir l'onglet **Projection Lab**.
  2. Sélectionner le document `cv_yassir.pdf` et son expérience de découpage.
  3. Projeter en PCA 2D : observer la distribution étalée le long des axes de variance maximale.
  4. Projeter en t-SNE 2D : observer le regroupement en îlots distincts (expérience professionnelle vs compétences techniques vs formation).

---

## Expérience 7 : Mesure de Débit et Latence d'Inférence par Batch

- **Objectif** : Quantifier le gain d'efficacité du traitement par lots (*batching*) sur CPU.
- **Procédure** :
  1. Exécuter le benchmark dans l'onglet **Benchmark Lab**.
  2. Relever la latence par phrase et le débit en tokens/seconde.
- **Observation Attendue** :
  - Sur CPU standard, `all-MiniLM-L6-v2` traite entre 500 et 1 500 tokens par seconde en batch de 16 ou 32, démontrant la faisabilité d'un RAG local sans GPU.

---

## Expérience 8 : Calcul de l'Empreinte Mémoire et de Stockage Vectoriel

- **Objectif** : Calculer et anticiper l'empreinte disque et RAM dans PostgreSQL (`pgvector`).
- **Formule** :
  $$\text{Taille (octets)} = N \times d \times 4 \text{ octets (float32)}$$
- **Cas d'école** :
  - 10 000 chunks à 384 dimensions : ~14,6 Mo
  - 100 000 chunks à 384 dimensions : ~146,5 Mo
  - 1 000 000 chunks à 384 dimensions : ~1,46 Go
- **Observation Attendue** :
  - L'indexation vectorielle sur CPU reste extrêmement économique pour des corpus de plusieurs centaines de milliers de documents.
