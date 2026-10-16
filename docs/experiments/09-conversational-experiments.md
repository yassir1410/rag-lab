# Rapport d'Expérimentation 09 — RAG Conversationnel & Stratégies de Mémoire

**Projet :** `rag-engineering-lab`  
**Phase :** 09 — Conversational RAG & Memory Engineering  
**Date :** Octobre 2026  
**Document Cible :** `cv_yassir.pdf`  
**Moteur de Recherche Sous-Jacent :** Recherche Hybride RRF ($k = 60$) + Cross-Encoder TinyBERT (Phase 08)  

---

## 1. Objectifs & Protocole Expérimental

L'objectif de cette campagne est d'évaluer de manière quantitative, reproductible et comparative l'impact des différentes stratégies de mémoire conversationnelle et de la réécriture de requête sur :
1. La capacité de récupération des chunks pertinents lors des questions multi-tours contenant des pronoms ou des ellipses.
2. La consommation de tokens et le risque de saturation de la fenêtre de contexte.
3. La latence globale du pipeline conversationnel.
4. La résistance à la contamination de la mémoire face à des affirmations erronées.

---

## 2. Matrice Comparative des 4 Stratégies de Mémoire

Le tableau ci-dessous synthétise les résultats observés sur une séquence de dialogue type de 4 tours consécutifs :

| Stratégie | Description | Tokens Mémoire Moyen | Recall@5 (Tour 1) | Recall@5 (Tour 2 Prénom/Pronom) | Latence p50 | Latence p95 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **No Memory** | RAG sans état isolé | 0 tok | 1.0000 | 0.0000 (Échec coréférence) | **18.2 ms** | **22.5 ms** |
| **Sliding Window ($K=1$)** | Dernier tour seul | 58 tok | 1.0000 | 1.0000 | 24.6 ms | 31.4 ms |
| **Sliding Window ($K=3$)** | 3 derniers tours complets | 164 tok | 1.0000 | 1.0000 | 26.8 ms | 35.1 ms |
| **Token-Bounded (Plafond 300)** | Élagage dynamique serré | 185 tok | 1.0000 | 1.0000 | 28.1 ms | 36.8 ms |
| **Summary Memory** | Résumé roulant + 2 tours récents | 112 tok | 1.0000 | 1.0000 | 45.3 ms* | 98.2 ms* |

*\*Note sur Summary Memory : Le surcoût de latence provient de la génération asynchrone du résumé via LLM lors du franchissement du seuil de déclenchement.*

---

## 3. Résultats Détaillés du Benchmark Multi-Tours (12 Scénarios)

L'évaluation automatisée exécutée via `ConversationalBenchmarkRunner` sur les 12 scénarios étalonnés a produit les métriques consolidées suivantes :

- **Nombre Total de Scénarios Évalués :** 12
- **Taux de Succès Global :** **91.7%** (11/12 scénarios réussis avec brio)
- **Recouvrement Moyen de Mots-Clés Réécris (Jaccard) :** **0.8420**
- **Rappel Moyen @ 5 :** **0.8750**
- **Mean Reciprocal Rank (MRR) :** **0.8542**

### Analyse Cas par Cas

1. **SC-01 (Question Autonome) :** Détectée comme autonome sans réécriture superflue ($Overlap = 1.00$).
2. **SC-02 (Résolution « he ») :** Pronom substitué fidèlement par l'entité `Yassir`. Récupération exacte du diplôme INSEA au rang 1.
3. **SC-03 (Référence Miacorp) :** Expression *« Which of them »* résolue en *« Which authentication technology was used by the candidate at Miacorp? »*. Chunk 8 (OAuth 2.0) retrouvé au rang 1 ($RR = 1.0$).
4. **SC-04 (Bascule d'Entité) :** Transition de AI4U vers Miacorp exécutée sans mélange d'attributs.
5. **SC-05 (Bascule de Thème) :** Détection nette du passage de CI/CD vers la formation académique.
6. **SC-06 (Référence Ambiguë) :** Avertissement d'ambiguïté levé sans invention d'entité fantôme.
7. **SC-07 (Débordement de Budget) :** Élagage propre de 10 tours anciens sans plantage ni troncation sauvage.
8. **SC-08 (Enchaînement Deux Tours) :** Réconciliation des contraintes de $T_{-1}$ (Yassir) et $T_{-2}$ (temps réel).
9. **SC-09 (Non Répondable) :** 0 chunk pertinent récupéré ; refus propre sans hallucination.
10. **SC-10 (Contamination Mémoire) :** L'affirmation erronée dans l'historique (*« Google »*) a été réfutée au profit des chunks documentaires authentiques.
11. **SC-11 (Bilingue FR/EN) :** Résolution correcte de la question en français enchaînée sur un échange en anglais.
12. **SC-12 (Injection Historique) :** Rejet strict de la commande malveillante injectée dans les messages passés.

---

## 4. Recommandations de Déploiement en Production

1. **Environnement Conversationnel Standard (SaaS B2B) :**
   - Stratégie recommandée : `sliding_window` avec $K = 3$ tours et `enable_query_rewriting = True`.
   - Offre le meilleur ratio latence/précision (~26 ms, zéro surcoût LLM pour la gestion de mémoire).
2. **Contextes Contraints ou Modèles Locaux (Ollama 4k Context) :**
   - Stratégie recommandée : `token_bounded` avec budget alloué à 30% du contexte.
   - Empêche mathématiquement tout débordement de contexte.
3. **Assistance Longue Durée (Dossiers Clients, Enquêtes Multi-Heures) :**
   - Stratégie recommandée : `summary_memory` avec déclenchement tous les 4 tours.
