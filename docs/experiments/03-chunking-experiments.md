# Laboratory Experiments — Phase 03: Chunking Strategies

This guide contains 8 reproducible, quantitative laboratory exercises to observe how chunking decisions impact token distribution, boundary integrity, and context retention.

---

## Experiment 1: 100 vs 300 vs 600 Token Windows

### Goal
Quantify how chunk size impacts the total chunk count and size uniformity on a standard technical article.

### Setup
Select `sample_documents/architecture_guide.md` in the Chunking Lab.

### Steps
1. Select **Fixed-Size Window** strategy.
2. Set `measurement_unit = token`, `chunk_overlap = 20`.
3. Test three configurations:
   - Run A: `chunk_size = 100`
   - Run B: `chunk_size = 300`
   - Run C: `chunk_size = 600`
4. Inspect the **Size Distribution & Metrics** tab for each run.

### Expected Observations
- Run A produces ~4x more chunks than Run C.
- Run A's observed overlap ratio is significantly higher relative to chunk size.
- Run C preserves multi-step code and paragraphs together, while Run A splits code blocks across multiple chunks.

---

## Experiment 2: Overlap Sensitivity Analysis (0%, 10%, 20%)

### Goal
Measure the text duplication ratio and boundary continuity across different overlap thresholds.

### Setup
Use `sample_documents/rag_overview.txt`.

### Steps
1. Select **Recursive Character** strategy.
2. Fix `chunk_size = 400` characters.
3. Test three overlap values:
   - Overlap = 0 characters (0%)
   - Overlap = 40 characters (10%)
   - Overlap = 80 characters (20%)
4. Record the **Duplicated Text Ratio** in the metrics bar.

### Expected Observations
- At 0% overlap, the duplicated text ratio is exactly 0.00%.
- At 10% overlap, duplication is ~8-12%.
- At 20% overlap, duplication reaches ~18-24%, increasing storage and embedding cost in exchange for boundary safety.

---

## Experiment 3: Fixed-Size vs Recursive Splitting on Dense Prose

### Goal
Compare linguistic integrity between naive slicing and delimiter-aware recursive splitting.

### Setup
Open the **Compare Strategies** tab in the Chunking Lab.

### Steps
1. Select `Strategy A = Fixed-Size Window` (`chunk_size = 300`, `overlap = 30`).
2. Select `Strategy B = Recursive Character` (`chunk_size = 300`, `overlap = 30`).
3. Click **Run Benchmark Comparison**.

### Expected Observations
- Strategy A cuts words and sentences abruptly at index 300.
- Strategy B respects paragraph breaks (`\n\n`) and sentence periods (`. `), keeping natural semantic clauses intact.

---

## Experiment 4: French vs English Sentence Segmentation

### Goal
Verify that abbreviations (*M., Mme., Dr., e.g., i.e.*) and decimal separators (*94,5%*) do not trigger false sentence breaks.

### Setup
Use a test document containing:
`"Le Dr. Martin et Mme. Dupont valident l'architecture. La précision atteint 94,5%. Tout fonctionne."`

### Steps
1. Select **Sentence-Based** strategy.
2. Set `chunk_size = 150`, `sentence_overlap = 0`.
3. Inspect generated chunks.

### Expected Observations
- Sentence 1 (`Le Dr. Martin et Mme. Dupont...`) remains intact in a single chunk.
- The period in `Dr.` and `Mme.` is not misinterpreted as a sentence boundary.
- The comma in `94,5%` is not split.

---

## Experiment 5: Semantic Chunking & Topic Transitions

### Goal
Observe how embedding cosine distance identifies topic shifts without relying on arbitrary character lengths.

### Setup
A composite document with two contrasting paragraphs (e.g. quantum physics followed by Parisian gastronomy).

### Steps
1. Select **Semantic Similarity** strategy.
2. Set `similarity_threshold = 0.70`, `min_chunk_size = 50`.
3. Click **Run Chunking**.
4. Check the `breakpoint_distance` badges on chunks.

### Expected Observations
- Sentences about quantum computing stay grouped together.
- When the topic abruptly shifts to French culinary traditions, similarity drops significantly, triggering a split.

---

## Experiment 6: Parent-Child Hierarchical Mapping

### Goal
Verify the structural decoupling between retrieval units (Child chunks) and generation units (Parent chunks).

### Setup
Use `sample_documents/architecture_guide.md`.

### Steps
1. Select **Parent-Child (Hierarchical)** strategy.
2. Set `parent_chunk_size = 1000`, `child_chunk_size = 250`, `child_overlap = 25`.
3. Run chunking.

### Expected Observations
- Chunks labeled as `Parent Chunk` have child IDs listed in their metadata.
- Each `Child Chunk` possesses an explicit `parent_chunk_id` linking back to its exact parent.

---

## Experiment 7: Structure-Aware Chunking on Tables & Code

### Goal
Verify that Markdown tables and fenced code blocks (` ``` `) remain atomic and are not sliced into fragments.

### Setup
Ingest `sample_documents/architecture_guide.md` in Phase 02 (contains a table and a python code block).

### Steps
1. Select **Structure-Aware** strategy.
2. Set `chunk_size = 400`, `preserve_tables = true`, `preserve_code_blocks = true`.
3. Click **Run Chunking**.

### Expected Observations
- The Markdown table `| Format | Library | ... |` is emitted as a single atomic chunk with `content_type = table`.
- The python function `def check_duplicate...` remains inside a single chunk with `content_type = code`.

---

## Experiment 8: Edge Cases & Stress Testing

### Goal
Verify system stability on extreme inputs: empty documents, 50,000-character unspaced strings, and accented Unicode.

### Steps
1. Test with an empty string: returns 0 chunks and 0 ms processing cleanly without crashing.
2. Test with an oversized string without spaces: the recursive chunker falls back to character-level slicing (`""` delimiter) without unbounded recursion.
