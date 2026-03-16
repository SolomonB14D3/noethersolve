# Paper Pipeline — From Discovery to Publication

Every paper goes through these stages in order. No stage can be skipped.
Each stage has a checklist that must be fully checked before proceeding.

---

## Stage 0: Data Collection
**Gate:** All experiments complete, all numbers finalized.

- [ ] All experimental results exist in `results/` or `results/discoveries/`
- [ ] Numbers are reproducible (scripts exist and run cleanly)
- [ ] Key claims have been verified by at least two independent methods
- [ ] Negative results documented alongside positive ones
- [ ] Raw data preserved (not just summaries)

**Output:** `paper/{paper_id}/data/` directory with all source data.

---

## Stage 1: Citation Research
**Gate:** Every factual claim has a citation or is explicitly marked as novel.

- [ ] Literature search for all prior work on the topic
- [ ] Every citation verified as REAL (check DOI resolves, check author names match, check year is correct)
- [ ] No hallucinated citations — if unsure, search the web to confirm
- [ ] Every citation actually says what we claim it says (no misattribution)
- [ ] Proper credit to prior work — if someone did something similar, cite them
- [ ] Novel claims clearly distinguished from known results
- [ ] Check for concurrent/recent work (arXiv search within last 6 months)

**Verification process:**
```bash
# For each citation, verify:
# 1. DOI resolves: curl -s -o /dev/null -w "%{http_code}" https://doi.org/DOI_HERE
# 2. Title/author match: fetch metadata from CrossRef or Semantic Scholar
# 3. Claim matches paper: read abstract + relevant section
```

**Output:** `paper/{paper_id}/references.bib` with verified entries only.

---

## Stage 2: Writing
**Gate:** Complete draft with all sections, figures referenced, numbers inline.

- [ ] Abstract: one paragraph, states the result, not the method
- [ ] Introduction: motivates the problem, states what's new, previews results
- [ ] Methods: reproducible by a competent reader
- [ ] Results: numbers match Stage 0 data exactly (cross-check every number)
- [ ] Discussion: honest about limitations, connects to prior work
- [ ] Conclusion: restates the key finding, suggests next steps

**Style rules:**
- [ ] No "delve", "tapestry", "intricate", "multifaceted", "it's worth noting", "notably", "in conclusion" (AI tell-words)
- [ ] No "we leverage", "we utilize" — use "we use"
- [ ] No "novel" in abstract (let the reader decide)
- [ ] No exclamation marks in scientific prose
- [ ] Active voice preferred ("we show X" not "X is shown")
- [ ] Numbers: use exact values from data, not rounded unless stated
- [ ] Define every symbol on first use
- [ ] Every figure/table referenced in text

**Output:** `paper/{paper_id}/draft.md` (Markdown source).

---

## Stage 3: Number Verification
**Gate:** Every number in the paper matches the source data.

- [ ] Create a verification script: `paper/{paper_id}/verify_numbers.py`
- [ ] Script loads raw data and prints every number that appears in the paper
- [ ] Diff the script output against the draft — zero discrepancies
- [ ] Percentages computed correctly (check denominators)
- [ ] Statistical claims verified (p-values, confidence intervals, effect sizes)
- [ ] Rounding is consistent throughout (same decimal places for same quantities)
- [ ] Table totals actually sum correctly

**Output:** `paper/{paper_id}/verify_numbers.py` passes with zero discrepancies.

---

## Stage 4: Figures
**Gate:** All figures are publication-ready.

- [ ] Every figure has a clear, self-contained caption
- [ ] Axes labeled with units
- [ ] Font size ≥ 8pt in all text (including tick labels)
- [ ] No overlapping labels or data
- [ ] Color-blind safe palette (avoid red/green only distinctions)
- [ ] Fits within column width (single: 3.5", double: 7.0") without scaling artifacts
- [ ] Vector format preferred (PDF/SVG), raster only for heatmaps/images (≥300 DPI)
- [ ] No matplotlib default styling — use clean, publication-ready style
- [ ] Legend doesn't obscure data
- [ ] Consistent styling across all figures in the paper
- [ ] Figure generation script: `paper/{paper_id}/make_figures.py`
- [ ] Script is reproducible (generates identical figures from data)

**Figure style template:**
```python
import matplotlib.pyplot as plt
import matplotlib as mpl

# Publication style
mpl.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'axes.linewidth': 0.8,
    'axes.labelsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.figsize': (3.5, 2.8),  # single column
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'lines.linewidth': 1.2,
    'lines.markersize': 4,
})

# Color-blind safe palette (Wong 2011)
COLORS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7',
          '#F0E442', '#56B4E9', '#E69F00', '#000000']
```

**Output:** `paper/{paper_id}/figures/` with all PDFs + generation script.

---

## Stage 5: AI Language Scrub
**Gate:** Paper reads like it was written by a domain expert, not an AI.

Run this checklist on every paragraph:

- [ ] Grep for banned words/phrases (see list below)
- [ ] Read each paragraph aloud — does it sound like a human wrote it?
- [ ] No hedging chains ("it is important to note that one might consider...")
- [ ] No unnecessary meta-commentary ("in this section, we will discuss...")
- [ ] No inflated adjectives ("remarkable", "groundbreaking", "profound")
- [ ] Specific claims, not vague ones ("improves by 12%" not "significantly improves")
- [ ] Short sentences preferred. Max 35 words per sentence.
- [ ] No paragraph starts with "Furthermore", "Moreover", "Additionally" in sequence

**Banned word/phrase list:**
```
delve, tapestry, intricate, multifaceted, synergy, paradigm shift,
it's worth noting, notably, interestingly, remarkably, importantly,
in conclusion, to summarize, as previously mentioned, it should be noted,
leverage (as verb), utilize, facilitate, elucidate, underscore,
a testament to, pave the way, shed light on, the landscape of,
holistic, robust (unless about statistics), comprehensive (in titles),
novel (in abstract), groundbreaking, cutting-edge, state-of-the-art (unless comparing)
```

**Output:** Clean draft with zero banned phrases.

---

## Stage 6: Finalization
**Gate:** Paper is ready for submission.

- [ ] Title is specific and under 15 words
- [ ] Author list and affiliations correct
- [ ] Acknowledgments section (compute resources, funding, data sources)
- [ ] All references formatted consistently
- [ ] Supplementary material organized (if any)
- [ ] Page count within venue limits
- [ ] Appendix for proofs/derivations that break flow
- [ ] Read the full paper one more time, start to finish
- [ ] Have the paper "reviewed" (re-read after 24h gap if possible)

**Output:** Final `paper/{paper_id}/final.md`.

---

## Stage 7: PDF Generation
**Gate:** PDF renders correctly with no artifacts.

**Method:** Markdown → LaTeX → PDF via pandoc, or direct LaTeX.

```bash
# Option A: pandoc (simpler)
pandoc paper/{paper_id}/final.md \
  -o paper/{paper_id}/{paper_id}.pdf \
  --pdf-engine=xelatex \
  -V geometry:margin=1in \
  -V fontsize=11pt \
  --bibliography=paper/{paper_id}/references.bib \
  --citeproc

# Option B: direct LaTeX (more control)
# Use paper/{paper_id}/paper.tex template
```

- [ ] PDF opens correctly
- [ ] All figures render (not broken image links)
- [ ] No text cut off at page boundaries
- [ ] Tables don't overflow margins
- [ ] Page numbers present
- [ ] Headers/footers correct
- [ ] Hyperlinks work (DOIs, URLs)
- [ ] References section renders correctly
- [ ] Math renders correctly (no missing symbols)

**Output:** `paper/{paper_id}/{paper_id}.pdf`.

---

## Stage 8: Zenodo Upload
**Gate:** Paper is publicly accessible with a DOI.

```bash
# 1. Create new deposit (or new version of existing)
curl -X POST "https://zenodo.org/api/deposit/depositions" \
  -H "Authorization: Bearer $ZENODO_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{}'

# 2. Upload PDF
curl -X PUT "https://zenodo.org/api/deposit/depositions/{id}/files/{filename}" \
  -H "Authorization: Bearer $ZENODO_TOKEN" \
  -H "Content-Type: application/octet-stream" \
  --data-binary @paper/{paper_id}/{paper_id}.pdf

# 3. Set metadata
curl -X PUT "https://zenodo.org/api/deposit/depositions/{id}" \
  -H "Authorization: Bearer $ZENODO_TOKEN" \
  -H "Content-Type: application/json" \
  -d @paper/{paper_id}/zenodo_metadata.json

# 4. Publish
curl -X POST "https://zenodo.org/api/deposit/depositions/{id}/actions/publish" \
  -H "Authorization: Bearer $ZENODO_TOKEN"
```

- [ ] Zenodo metadata complete (title, authors, description, keywords, license)
- [ ] Upload type: "publication" / subtype: "preprint"
- [ ] License: CC-BY-4.0
- [ ] Related identifiers link to GitHub repo and any prior papers
- [ ] DOI minted and resolves correctly
- [ ] Record the DOI

**Output:** DOI recorded in `paper/{paper_id}/DOI.txt`.

---

## Stage 9: GitHub Update
**Gate:** Repository reflects the new publication.

- [ ] Add DOI to this project's CLAUDE.md (Discovery Papers section)
- [ ] Add DOI to rho-eval CLAUDE.md (if cross-referenced)
- [ ] Update MEMORY.md paper roadmap with DOI
- [ ] Add paper entry to README.md publications section
- [ ] Tag release if version bump warranted: `git tag vX.Y.Z`
- [ ] Push tag (triggers Zenodo webhook for toolkit papers): `git push --tags`
- [ ] Verify Zenodo auto-linked the GitHub release (if applicable)

**Output:** All references updated, pushed, verified.

---

## Paper Status Tracker

| ID | Title | S0 | S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 | DOI |
|----|-------|----|----|----|----|----|----|----|----|----|----|-----|
| D1 | Q_f Conservation Laws | :white_check_mark: | | | | | | | | | | |
| D2 | Z₃ Phase Cancellation | :white_check_mark: | | | | | | | | | | |
| D3 | Where LLMs Are Wrong | :white_check_mark: | | | | | | | | | | |
| D4 | Orthogonal Adapter Routing | :white_check_mark: | | | | | | | | | | |

---

## Quick Reference: Paper IDs

| ID | Short Name | File Prefix | Domain |
|----|-----------|-------------|--------|
| D1 | qf_conservation | `d1_` | Fluid dynamics / mathematical physics |
| D2 | z3_cancellation | `d2_` | Celestial mechanics / dynamical systems |
| D3 | llm_knowledge_gaps | `d3_` | AI / machine learning |
| D4 | orthogonal_routing | `d4_` | NLP / adapter methods |
