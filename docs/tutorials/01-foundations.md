# Tutoriel 01 : Les Fondations du RAG et de l'Ingénierie Logicielle

Bienvenue dans le premier module pédagogique du **RAG Engineering Lab**. Ce laboratoire a pour vocation de vous apprendre à concevoir, mesurer et déployer des architectures de **Génération Augmentée par Récupération** (RAG, pour *Retrieval-Augmented Generation*).

---

## 1. Qu'est-ce que le RAG ?

Le terme **RAG** (*Retrieval-Augmented Generation*) a été formalisé par Lewis et al. (Meta AI, 2020). 

Un grand modèle de langage (LLM comme GPT-4, Claude ou Gemini) possède une mémoire **paramétrique** : il a stocké des connaissances générales lors de sa phase d'entraînement. Cependant, cette mémoire présente trois limites majeures en entreprise :
1. **L'obsolescence** : Le modèle ne connaît rien des événements ou données survenus après sa date d'entraînement (*knowledge cutoff*).
2. **L'absence de données privées** : Le modèle n'a pas accès à vos bases internes, contrats, wikis d'entreprise ou tickets clients.
3. **Les hallucinations** : Face à une question pointue dont il ne connaît pas la réponse exacte, le modèle invente des affirmations factuellement fausses mais linguistiquement plausibles.

Le **RAG** résout ce défi en combinant deux composantes :
- Une composante de **Recherche** (*Retrieval*) : interroger une base de connaissances externe pour extraire les extraits de texte les plus pertinents par rapport à la question de l'utilisateur.
- Une composante de **Génération** (*Generation*) : injecter ces extraits vérifiés dans le contexte (*prompt*) transmis au LLM, en lui demandant de formuler sa réponse en se basant exclusivement sur ces faits.

---

## 2. Recherche (*Retrieval*) vs Génération (*Generation*)

Comprendre la distinction entre ces deux rôles est fondamental :

| Étape | Retrieval (Recherche) | Generation (Génération) |
|---|---|---|
| **Rôle** | Trouver des faits et documents exacts | Rédiger, résumer et expliquer |
| **Nature de l'opération** | Déterministe ou probabiliste (mots-clés, similarité vectorielle) | Synthèse de texte par réseau de neurones autorégressif |
| **Objectif de qualité** | Haute précision et fort rappel (ne rien oublier d'utile) | Clarté, fidélité aux sources, absence d'hallucinations |
| **Coût computationnel** | Faible à modéré (recherche indexée dans PostgreSQL) | Élevé (inférence LLM) |

> **Principe clé :** *Garbage in, garbage out*. Si l'étape de recherche renvoie des documents tronqués, mal analysés ou non pertinents, le meilleur LLM du monde produira une réponse erronée ou hors-sujet.

---

## 3. Rôles des Composants du Système

Le **RAG Engineering Lab** repose sur trois piliers logiciels :

### A. Le Backend (FastAPI & Python 3.12)
Python est le standard de l'écosystème IA et NLP. FastAPI offre une exécution asynchrone performante, une documentation Swagger interactive générée automatiquement, et une validation stricte des données grâce à Pydantic v2. Le backend orchestre la chaîne complète : validation des fichiers, extraction, calcul des empreintes SHA-256, normalisation et découpage.

### B. La Base de Données (PostgreSQL 16 + pgvector)
Plutôt que d'utiliser une base vectorielle isolée, nous utilisons PostgreSQL avec son extension native `pgvector`. Cela permet de manipuler les métadonnées relationnelles (titres, dates, droits d'accès, pages) et les vecteurs de plongement sémantique au sein du même moteur transactionnel conforme ACID.

### C. Le Frontend (React, Vite, Tailwind CSS)
Une interface web moderne en pur JavaScript permettant de visualiser directement les transformations appliquées aux données (textes bruts vs normalisés, métadonnées, scores de similarité, citations).

---

## 4. Pourquoi une Architecture Modulaire ?

Dans beaucoup de tutoriels, le RAG est présenté à l'aide d'un script unique de 30 lignes utilisant une bibliothèque tout-en-un. En production, cette approche monolithique échoue rapidement :
- On ne sait pas pourquoi un document précis n'a pas été trouvé.
- Les tableaux et le code sont détruits lors du parsing.
- Les métadonnées de page sont perdues, rendant impossible la citation exacte des sources.

Notre architecture découpe chaque responsabilité dans un sous-module dédié :
- `rag/ingestion` : Validation, extraction multi-format, nettoyage.
- `rag/chunking` : Découpage sémantique et par phrases.
- `rag/embeddings` : Calcul des vecteurs denses.
- `rag/retrieval` : Recherche vectorielle, BM25 et recherche hybride.
- `rag/generation` : Construction des prompts et synthèse.
- `rag/evaluation` : Mesure scientifique de la précision et de la fidélité.

---

## 5. Prochaines Étapes

Après la mise en place de la fondation (Phase 01) et du système d'ingestion documentaire (Phase 02), vous explorerez :
1. Les stratégies de découpage textuel (*chunking*).
2. La géométrie des espaces vectoriels (*embeddings*).
3. L'indexation et la recherche vectorielle avec pgvector.
4. La recherche hybride et le reclassement (*reranking*).
