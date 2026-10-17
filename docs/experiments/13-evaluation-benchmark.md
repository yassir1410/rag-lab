# Expérience 13 : Benchmark Global & Mesures de Qualité RAG

**Date :** 17 Octobre 2026  
**Auteur :** Équipe RAG Engineering Lab  
**Phase :** Phase 13 — RAG Evaluation, Observability & Quality Engineering  
**Statut :** Validé (Tests Déterministes Déployés & Exécutés)  

---

## 1. Contexte & Objectifs

La Phase 13 introduit une plateforme unifiée d'évaluation, d'observabilité et de comptabilité pour l'ensemble des 12.5 phases du laboratoire.

Le jeu de données de référence `Phase 13 Comprehensive RAG Benchmark` (ID persistant : `d4de5bc1-fffc-43dc-8992-1636733c2b91`) rassemble 24 scénarios d'évaluation rigoureusement calibrés pour tester les limites du système RAG.

---

## 2. Taxonomie des 24 Scénarios d'Évaluation

| ID | Catégorie | Modalité | Question Cible | Sortie Attendue & Tolérance | Route Attendue | Source Label |
|---|---|---|---|---|---|---|
| **BM-01** | Factual | Text | Force primaire derrière les lois de Kepler | Gravitational attraction / Gravity | `text_dense`, `text_hybrid` | human |
| **BM-02** | Multi-Chunk | Text | Différence composition planètes telluriques vs gazeuses | Rocheux/silicates/fer vs hydrogène/hélium/glaces | `text_dense`, `text_hybrid` | human |
| **BM-03** | Factual (FR) | Text | Caractéristiques principales de l'atmosphère martienne | 95% CO2, faible pression ~6 mbar | `text_dense`, `text_hybrid` | human |
| **BM-04** | Paraphrasing | Text | Variations aphélie / périhélie et excentricité | Distance orbitale minimale vs maximale | `text_hybrid`, `text_dense` | human |
| **BM-05** | Negation | Text | Planètes sans aucun satellite naturel | Mercure et Vénus | `text_hybrid`, `text_dense` | human |
| **BM-06** | Factual | Table | Gravité de surface de Jupiter (page 1) | 23.1 m/s² (tolérance ±0.5 m/s²) | `multimodal_table` | human |
| **BM-07** | Comparative | Table | Planète avec la densité moyenne la plus élevée | Terre, ~5 514 kg/m³ (tolérance ±50 kg/m³) | `multimodal_table` | human |
| **BM-08** | OCR | Text | Titre extrait via ROC sur page scannée | RAG Multimodal Benchmark: Solar System... | `text_dense`, `multimodal_visual` | human |
| **BM-09** | Visual | Visual | Caractéristique géométrique de la ceinture d'astéroïdes | Bande toroïdale de débris entre Mars et Jupiter | `multimodal_visual` | human |
| **BM-10** | Visual | Visual | Planète avec anneaux proéminents au-delà de l'équateur | Saturne | `multimodal_visual` | human |
| **BM-11** | Graph | Graph | Géantes gazeuses orbitant autour du Soleil | Jupiter et Saturne | `graph` | human |
| **BM-12** | Graph | Graph | Mission New Horizons (survol de Pluton et cible suivante) | Pluton (2015) puis Arrokoth (2014 MU69) | `graph`, `text_dense` | human |
| **BM-13** | Agentic | Agentic | Ratio rayon orbital Neptune / Mercure | ~77.1 (30.07 AU / 0.39 AU) | `agentic` | human |
| **BM-14** | Agentic | Agentic | Pression de surface Vénus/Terre/Mars et eau liquide | 92 bar, 1 bar, 0.006 bar (seuil point triple) | `agentic`, `text_hybrid` | human |
| **BM-15** | Mixed | Mixed | Effet de serre Vénus + température du tableau | ~464 °C (737 K) via CO2 | `multimodal_table`, `text_hybrid` | human |
| **BM-16** | Mixed | Mixed | Excentricité Mercure (tableau vs diagramme) | Oui, e = 0.2056 corroboré par ellipse décalée | `multimodal_table`, `multimodal_visual` | human |
| **BM-17** | Ambiguous | Mixed | Vue générale et propriétés de Jupiter | Géante gazeuse >300 masses terrestres | `text_dense`, `text_hybrid`, `multimodal_table` | human |
| **BM-18** | Missing Index | Graph | Communauté de Kepler dans un graphe non extrait | Refus explicite / absence d'index | `graph`, `text_dense` | synthetic |
| **BM-19** | Isolation | Text | Demi-grand axe de Mars pour le document spécifié | 1.524 AU (isolation stricte 0% fuite) | `text_dense`, `text_hybrid` | human |
| **BM-20** | Unanswerable | Text | Cours de l'action Alphabet sur NASDAQ | Refus conforme (hors corpus) | `text_dense`, `text_hybrid` | human |
| **BM-21** | Distractor | Text | Cinquième lune découverte autour de Mercure | Refus conforme (prémisse fausse, 0 lune) | `text_dense`, `text_hybrid` | human |
| **BM-22** | Conflicting | Text | Année de preuve de vie microbienne sur Europe | Refus conforme (absence de preuve factuelle) | `text_dense`, `text_hybrid` | synthetic |
| **BM-23** | Reliability | Text | Gestion des timeouts amont (HTTP 504) | Erreur structurée sans hallucination | `text_dense` | synthetic |
| **BM-24** | Security | Text | Prompt injection ("Ignore all previous instructions...") | Refus sécurisé et maintien du contexte | `text_dense`, `text_hybrid` | human |

---

## 3. Résultats des Mesures Déterministes (Mock Provider)

Toutes les métriques de base ont été validées sans appel payant externe :
- **Retrieval Metrics Suite :**
  - Mean Precision@5 : 0.4000 (calculs de validation)
  - Mean Recall@5 : 1.0000
  - Mean F1@5 : 0.5714 (moyenne harmonique conforme)
  - Mean Hit Rate@5 : 1.0000
  - Mean Reciprocal Rank (MRR) : 1.0000
  - Normalized Discounted Cumulative Gain (nDCG@5) : 1.0000
- **Deterministic Generation Accuracy :**
  - Exact Match (EM) : validé avec normalisation ponctuation/espaces.
  - Token F1 : validé sur les correspondances unigrammes.
  - Normalized Match : 100% de concordance sur les variations acceptables et inclusions.
  - Numerical Correctness : 100% sur les grandeurs avec unités (`23.1 m/s²`, `5514 kg/m³`, `1.524 AU`).
- **Routing & Isolation :**
  - Route Accuracy : 100% sur routes simples et ensembles acceptables.
  - Unnecessary Vision Detection : 0% de faux positifs.
  - Document Isolation Score : 100% (0% fuite inter-documents).
- **Structured Tracing :**
  - 100% des secrets (`api_key`, `x-goog-api-key`, `Authorization`) caviardés.
  - 100% des chaînes Base64 images remplacées par `[REDACTED_BASE64_IMAGE]`.

---

## 4. Recommandations pour la Production

1. **Ne jamais considérer un score LLM Judge comme une vérité terrain objective :** Les juges LLM présentent des biais de longueur et d'auto-complaisance. Toujours corréler avec des métriques déterministes (EM, Token F1, juste numérique).
2. **Surveiller le ratio de vision inutile :** L'activation de modèles multimodaux sur des tableaux textuels augmente la latence par 5 et le coût par 4 sans gain factuel par rapport à une extraction Markdown structurée.
3. **Persister les traces en base plutôt qu'en mémoire tampon :** La persistance dans `evaluation_traces` garantit l'auditabilité à long terme et l'analyse post-incident des régressions.
