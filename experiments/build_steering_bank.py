#!/usr/bin/env python3
"""Build steering vector bank from HuggingFace benchmark datasets.

Downloads MMLU, SciQ, ARC, TruthfulQA, OpenBookQA, HellaSwag, Winogrande
-> converts to fact format -> extracts steering vectors for each domain.

Usage:
    python experiments/build_steering_bank.py --download   # Download + convert only
    python experiments/build_steering_bank.py --extract    # Extract vectors only
    python experiments/build_steering_bank.py --all        # Both
"""
import argparse
import json
import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np

PROJECT = Path(__file__).parent.parent
BANK_DIR = PROJECT / "steering_bank"
VECTORS_DIR = PROJECT / "steering_vectors"

sys.path.insert(0, str(PROJECT))


def download_mmlu():
    from datasets import load_dataset
    print("\n  Downloading MMLU...")
    ds = load_dataset("cais/mmlu", "all", split="test")
    by_subject = defaultdict(list)
    for row in ds:
        truth = row["choices"][row["answer"]]
        distractors = [c for i, c in enumerate(row["choices"]) if i != row["answer"]]
        by_subject[row["subject"]].append({
            "context": row["question"], "truth": truth, "distractors": distractors,
        })
    saved = 0
    for subject, facts in by_subject.items():
        if len(facts) < 5: continue
        with open(BANK_DIR / f"mmlu_{subject}.json", "w") as f:
            json.dump({"domain": f"mmlu_{subject}", "source": "mmlu", "facts": facts}, f)
        saved += 1
    print(f"  MMLU: {saved} subjects, {sum(len(v) for v in by_subject.values()):,} facts")
    return saved


def download_sciq():
    from datasets import load_dataset
    print("\n  Downloading SciQ...")
    ds = load_dataset("allenai/sciq", split="test")
    facts = []
    for row in ds:
        d = [row["distractor1"], row["distractor2"], row["distractor3"]]
        d = [x for x in d if x and x.strip()]
        if len(d) >= 2:
            facts.append({"context": row["question"], "truth": row["correct_answer"], "distractors": d})
    with open(BANK_DIR / "sciq.json", "w") as f:
        json.dump({"domain": "sciq", "source": "sciq", "facts": facts}, f)
    print(f"  SciQ: {len(facts)} facts")


def download_arc():
    from datasets import load_dataset
    print("\n  Downloading ARC-Challenge...")
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    facts = []
    for row in ds:
        labels, texts = row["choices"]["label"], row["choices"]["text"]
        idx = next((i for i, l in enumerate(labels) if l == row["answerKey"]), None)
        if idx is None: continue
        facts.append({"context": row["question"], "truth": texts[idx],
                       "distractors": [t for i, t in enumerate(texts) if i != idx]})
    with open(BANK_DIR / "arc_challenge.json", "w") as f:
        json.dump({"domain": "arc_challenge", "source": "arc", "facts": facts}, f)
    print(f"  ARC-Challenge: {len(facts)} facts")


def download_truthfulqa():
    from datasets import load_dataset
    print("\n  Downloading TruthfulQA...")
    ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation")
    facts = []
    for row in ds:
        mc1 = row["mc1_targets"]
        truth = None; distractors = []
        for c, l in zip(mc1["choices"], mc1["labels"]):
            if l == 1: truth = c
            else: distractors.append(c)
        if truth and distractors:
            facts.append({"context": row["question"], "truth": truth, "distractors": distractors[:3]})
    with open(BANK_DIR / "truthfulqa.json", "w") as f:
        json.dump({"domain": "truthfulqa", "source": "truthfulqa", "facts": facts}, f)
    print(f"  TruthfulQA: {len(facts)} facts")


def download_openbookqa():
    from datasets import load_dataset
    print("\n  Downloading OpenBookQA...")
    ds = load_dataset("allenai/openbookqa", "main", split="test")
    facts = []
    for row in ds:
        labels, texts = row["choices"]["label"], row["choices"]["text"]
        idx = next((i for i, l in enumerate(labels) if l == row["answerKey"]), None)
        if idx is None: continue
        facts.append({"context": row["question_stem"], "truth": texts[idx],
                       "distractors": [t for i, t in enumerate(texts) if i != idx]})
    with open(BANK_DIR / "openbookqa.json", "w") as f:
        json.dump({"domain": "openbookqa", "source": "openbookqa", "facts": facts}, f)
    print(f"  OpenBookQA: {len(facts)} facts")


def download_hellaswag():
    from datasets import load_dataset
    print("\n  Downloading HellaSwag...")
    ds = load_dataset("Rowan/hellaswag", split="validation")
    facts = []
    for row in ds:
        endings = row["endings"]; label = int(row["label"])
        if label >= len(endings): continue
        facts.append({"context": row["ctx"], "truth": endings[label],
                       "distractors": [e for i, e in enumerate(endings) if i != label][:3]})
    with open(BANK_DIR / "hellaswag.json", "w") as f:
        json.dump({"domain": "hellaswag", "source": "hellaswag", "facts": facts}, f)
    print(f"  HellaSwag: {len(facts)} facts")


def download_winogrande():
    from datasets import load_dataset
    print("\n  Downloading Winogrande...")
    ds = load_dataset("allenai/winogrande", "winogrande_xl", split="validation")
    facts = []
    for row in ds:
        if row["answer"] == "1":
            facts.append({"context": row["sentence"], "truth": row["option1"], "distractors": [row["option2"]]})
        elif row["answer"] == "2":
            facts.append({"context": row["sentence"], "truth": row["option2"], "distractors": [row["option1"]]})
    with open(BANK_DIR / "winogrande.json", "w") as f:
        json.dump({"domain": "winogrande", "source": "winogrande", "facts": facts}, f)
    print(f"  Winogrande: {len(facts)} facts")


def run_downloads():
    BANK_DIR.mkdir(exist_ok=True)
    print("=" * 60)
    print("  Downloading benchmark datasets -> steering_bank/")
    print("=" * 60)
    for fn in [download_mmlu, download_sciq, download_arc, download_truthfulqa,
               download_openbookqa, download_hellaswag, download_winogrande]:
        try:
            fn()
        except Exception as e:
            print(f"  FAILED: {e}")
    n_facts = 0; n_files = 0
    for f in BANK_DIR.glob("*.json"):
        with open(f) as fh: data = json.load(fh)
        n_facts += len(data.get("facts", [])); n_files += 1
    print(f"\n  TOTAL: {n_files} domain files, {n_facts:,} facts")


def extract_vectors(layer=15):
    import mlx.core as mx
    from mlx_lm import load

    alphas = [0.10, 0.25, 0.50, 0.75, 1.0, 1.5, 2.0]
    VECTORS_DIR.mkdir(exist_ok=True)

    domain_files = []
    if BANK_DIR.exists():
        domain_files += sorted(BANK_DIR.glob("*.json"))
    domain_files += sorted((PROJECT / "problems").glob("*_facts*.json"))

    print(f"\nLoading Qwen/Qwen3-14B-Base...")
    model, tokenizer = load("Qwen/Qwen3-14B-Base")
    print(f"Extracting vectors from {len(domain_files)} domain files, layer {layer}")
    print(f"{'='*80}")

    results = []
    total_start = time.time()

    for df in domain_files:
        try:
            with open(df) as f: data = json.load(f)
        except Exception: continue

        domain = data.get("domain", df.stem.replace("_facts_v2", "").replace("_facts", ""))
        facts = data.get("facts", data.get("verifications", []))
        if len(facts) < 10: continue

        start = time.time()
        np.random.seed(42)
        idx = np.random.permutation(len(facts))
        split = min(len(facts) // 2, 50)
        train = [facts[i] for i in idx[:split]]
        test = [facts[i] for i in idx[split:split+50]]

        def score_mc(fact_list):
            correct = 0; total = 0; scored = []
            for fact in fact_list:
                ctx = fact.get("context", "")
                truth = fact.get("truth", fact.get("fact", ""))
                dist = fact.get("distractors", [])
                if not dist: continue
                options = [truth] + dist[:3]
                prompt = (ctx + "\n\n" if ctx else "") + "Which is correct?\n"
                for i, opt in enumerate(options): prompt += f"{chr(65+i)}) {opt}\n"
                prompt += "Answer: "
                tokens = tokenizer.encode(prompt)
                logits = model(mx.array([tokens])); mx.eval(logits)
                opt_toks = [tokenizer.encode(chr(65+i))[-1] for i in range(len(options))]
                opt_logits = [float(logits[0, -1, t].item()) for t in opt_toks]
                pred = int(np.argmax(opt_logits))
                if pred == 0: correct += 1
                total += 1
                scored.append({"fact": fact, "correct": pred == 0, "predicted": pred})
            return correct, total, scored

        _, train_total, train_scored = score_mc(train)
        test_correct, test_total, _ = score_mc(test)
        baseline = test_correct / test_total if test_total else 0

        wrong = [s for s in train_scored if not s["correct"]]
        if len(wrong) < 2:
            elapsed = time.time() - start
            print(f"  {domain:<45s} base={baseline:.0%} wrong={len(wrong)}/{train_total} SKIP [{elapsed:.1f}s]")
            results.append({"domain": domain, "baseline": baseline, "skipped": True, "n_wrong": len(wrong)})
            continue

        # Steering vector from wrong answers
        correct_prompts = []; incorrect_prompts = []
        for item in wrong:
            fact = item["fact"]
            ctx = fact.get("context", ""); truth = fact.get("truth", fact.get("fact", ""))
            dist = fact.get("distractors", [])
            if not dist: continue
            base_p = f"{ctx}\n\nAnswer: " if ctx else "Answer: "
            correct_prompts.append(base_p + truth)
            picked = item["predicted"] - 1
            incorrect_prompts.append(base_p + (dist[picked] if 0 <= picked < len(dist) else dist[0]))

        def get_acts(prompts):
            acts = []
            for p in prompts:
                tokens = tokenizer.encode(p)
                hidden = model.model.embed_tokens(mx.array([tokens]))
                for i, lyr in enumerate(model.model.layers):
                    hidden = lyr(hidden, mask=None, cache=None)
                    if i == layer:
                        acts.append(np.array(hidden[0, -1, :].astype(mx.float32)))
                        break
                mx.eval(hidden)
            return np.array(acts)

        sv = get_acts(correct_prompts).mean(0) - get_acts(incorrect_prompts).mean(0)

        # Test alphas
        best_acc = baseline; best_alpha = 0.0
        for alpha in alphas:
            sc = 0; st = 0
            for fact in test:
                ctx = fact.get("context", ""); truth = fact.get("truth", fact.get("fact", ""))
                dist = fact.get("distractors", [])
                if not dist: continue
                options = [truth] + dist[:3]
                prompt = (ctx + "\n\n" if ctx else "") + "Which is correct?\n"
                for i, opt in enumerate(options): prompt += f"{chr(65+i)}) {opt}\n"
                prompt += "Answer: "
                tokens = tokenizer.encode(prompt)
                hidden = model.model.embed_tokens(mx.array([tokens]))
                for i, lyr in enumerate(model.model.layers):
                    hidden = lyr(hidden, mask=None, cache=None)
                    if i == layer:
                        steer = mx.array(sv.astype(np.float32) * alpha).reshape(1, 1, -1)
                        hidden = hidden + steer
                hidden = model.model.norm(hidden)
                if model.args.tie_word_embeddings:
                    logits = model.model.embed_tokens.as_linear(hidden)
                else:
                    logits = model.lm_head(hidden)
                mx.eval(logits)
                opt_toks = [tokenizer.encode(chr(65+i))[-1] for i in range(len(options))]
                opt_logits = [float(logits[0, -1, t].item()) for t in opt_toks]
                if int(np.argmax(opt_logits)) == 0: sc += 1
                st += 1
            acc = sc / st if st else 0
            if acc > best_acc: best_acc = acc; best_alpha = alpha

        improvement = best_acc - baseline
        elapsed = time.time() - start
        vec_path = VECTORS_DIR / f"{domain}_layer{layer}.npy"
        np.save(vec_path, sv)
        status = "+" if improvement > 0 else ("=" if improvement == 0 else "-")
        print(f"  {domain:<45s} {baseline:.0%}->{best_acc:.0%} a={best_alpha} wrong={len(wrong)}/{train_total} {status} [{elapsed:.1f}s]")
        results.append({"domain": domain, "baseline": baseline, "best_steered": best_acc,
                        "improvement": improvement, "best_alpha": best_alpha,
                        "n_wrong": len(wrong), "skipped": False})

    with open(VECTORS_DIR / "bank_results.json", "w") as f:
        json.dump(results, f, indent=2)

    elapsed_total = time.time() - total_start
    active = [r for r in results if not r.get("skipped")]
    improved = [r for r in active if r.get("improvement", 0) > 0]
    skipped = [r for r in results if r.get("skipped")]
    print(f"\n{'='*80}")
    print(f"DONE: {len(results)} domains in {elapsed_total/60:.1f}min")
    print(f"Processed: {len(active)} | Skipped: {len(skipped)} | Improved: {len(improved)}")
    if improved:
        print(f"\nIMPROVED ({len(improved)}):")
        for r in sorted(improved, key=lambda x: -x["improvement"]):
            print(f"  {r['domain']:<45s} {r['baseline']:.0%}->{r['best_steered']:.0%} (+{r['improvement']:.0%})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--layer", type=int, default=15)
    args = parser.parse_args()
    if args.all or args.download: run_downloads()
    if args.all or args.extract: extract_vectors(layer=args.layer)
    if not (args.all or args.download or args.extract):
        print("Specify --download, --extract, or --all")

if __name__ == "__main__":
    main()
