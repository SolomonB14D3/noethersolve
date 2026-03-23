#!/usr/bin/env python3
"""Staged adapter training for HSV-APOE4-Alzheimer's domain.

Clusters:
1. APOE structure (amino acids, risk levels)
2. Drug/antiviral (CNS penetration, mechanisms)
3. Inflammasome/NLRP3 (pathway, inhibitors)
4. Tau/viral (antimicrobial function, colocalization)
5. Microglia/TREM2 (function, variants)
"""

import json
import os
import sys
import time

import mlx.core as mx
import mlx.optimizers as optim
import mlx_lm
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

from noethersolve.adapter import SnapOnConfig, create_adapter
from noethersolve import train_utils as t3

OUT_DIR = os.path.join(ROOT, "adapters")

# Define clusters for staged training
CLUSTERS = {
    "apoe_structure": [
        {"context": "APOE2 at position 158 has amino acid", "truth": "Cysteine (Cys158)", "distractors": ["Arginine (Arg158)", "Glutamate (Glu158)", "Aspartate (Asp158)"]},
        {"context": "APOE4 at position 158 has amino acid", "truth": "Arginine (Arg158)", "distractors": ["Cysteine (Cys158)", "Glutamate (Glu158)", "Aspartate (Asp158)"]},
        {"context": "APOE2 homozygotes have AD risk compared to E3/E3", "truth": "reduced by approximately 90%", "distractors": ["unchanged", "reduced by 20-30%", "increased by 50%"]},
        {"context": "APOE Christchurch variant R136S is", "truth": "protective against AD", "distractors": ["high risk like APOE4", "neutral like APOE3", "causes early-onset AD"]},
    ],
    "epidemiology": [
        {"context": "APOE4 carriers with HSV-1 reactivation have AD risk", "truth": "increased 3-4 fold (HR 3.28-3.68)", "distractors": ["unchanged from baseline", "increased 1.2-1.5 fold", "increased 10-20 fold"]},
        {"context": "In APOE4-negative individuals HSV-1 affects AD risk by", "truth": "no significant association (requires gene-virus interaction)", "distractors": ["doubling the risk", "reducing risk", "same 3-4 fold increase as carriers"]},
        {"context": "The gene-virus interaction for HSV-AD requires", "truth": "both APOE4 genotype AND HSV-1 infection", "distractors": ["either factor alone is sufficient", "only APOE4 matters", "only HSV-1 matters"]},
        {"context": "HSV-1 DNA presence in human AD brains is associated with", "truth": "APOE4 genotype", "distractors": ["APOE2 genotype", "no genetic association", "APOE3 only"]},
    ],
    "antivirals": [
        {"context": "Adibelivir brain/plasma ratio is", "truth": "0.5 to 4.0 (excellent CNS penetration)", "distractors": ["below 0.1 (poor penetration)", "0.1 to 0.2 (moderate)", "equivalent to valacyclovir"]},
        {"context": "Pritelivir brain/plasma ratio is", "truth": "below 0.1 (limited CNS penetration)", "distractors": ["0.5 to 4.0 (excellent)", "0.2 to 0.5 (moderate)", "equivalent to adibelivir"]},
        {"context": "The VALAD trial of valacyclovir in HSV+ early AD showed", "truth": "no benefit (cognitive scores worsened vs placebo)", "distractors": ["significant benefit", "modest benefit in APOE4 subgroup", "reduced amyloid by 20%"]},
        {"context": "HSV helicase-primase inhibitors differ from nucleoside analogs by", "truth": "not requiring viral thymidine kinase activation", "distractors": ["requiring phosphorylation", "being less potent", "identical mechanism"]},
        {"context": "Valacyclovir failed in AD trials primarily because of", "truth": "poor CNS penetration and no anti-inflammatory action", "distractors": ["viral resistance", "patient non-compliance", "wrong dosing"]},
    ],
    "inflammasome": [
        {"context": "APOE4 activates NLRP3 inflammasome via", "truth": "NF-kB signaling pathway upregulation", "distractors": ["direct NLRP3 binding", "calcium channel modulation", "autophagy enhancement"]},
        {"context": "MCC950 in HSV-infected 5xFAD mice", "truth": "reduced amyloid burden and cognitive decline", "distractors": ["had no effect", "worsened neuroinflammation", "only affected tau"]},
        {"context": "NLRP3 inflammasome activation produces", "truth": "IL-1beta and IL-18 via caspase-1", "distractors": ["only TNF-alpha", "only interferon-gamma", "no cytokines"]},
        {"context": "APOE4 lipid droplet accumulation in microglia triggers", "truth": "inflammasome activation and inflammatory mediator release", "distractors": ["neuroprotection", "enhanced phagocytosis", "apoptosis"]},
    ],
    "tau_viral": [
        {"context": "Tau hyperphosphorylation after HSV-1 infection functions as", "truth": "antimicrobial defense that binds viral capsids", "distractors": ["purely pathological process", "viral replication enhancer", "apoptosis trigger"]},
        {"context": "Phosphorylated tau neutralizes HSV-1 by", "truth": "directly binding viral capsids", "distractors": ["blocking viral entry", "triggering interferon", "activating autophagy"]},
        {"context": "HSV-1 ICP27 protein in AD brains colocalizes with", "truth": "phosphorylated tau (not amyloid-beta)", "distractors": ["amyloid-beta plaques", "both tau and amyloid equally", "neither"]},
        {"context": "HSV-1 infection in neurons upregulates gamma-secretase subunits", "truth": "PSEN1 and PSEN2", "distractors": ["APP only", "BACE1 only", "neither"]},
    ],
    "microglia_trem2": [
        {"context": "TREM2 R47H variant affects microglial HSV-1 response by", "truth": "impairing antiviral function (increased viral replication)", "distractors": ["enhancing viral clearance", "no effect", "blocking BBB viral entry"]},
        {"context": "TREM2 depletion in HSV-1 infected co-cultures leads to", "truth": "elevated viral replication", "distractors": ["reduced viral replication", "no change", "complete viral clearance"]},
        {"context": "APOE4 microglia show immune response that is", "truth": "exaggerated with reduced Abeta clearance", "distractors": ["suppressed with normal clearance", "normal with enhanced clearance", "identical to APOE2/E3"]},
        {"context": "APOE-I3 targeting ASOs in tauopathy mice", "truth": "reduced neurodegeneration by approximately 50%", "distractors": ["had no effect", "worsened tau accumulation", "only affected peripheral APOE"]},
    ],
    "vaccines": [
        {"context": "Recombinant zoster vaccine reduces dementia risk by", "truth": "approximately 51% in adults over 65", "distractors": ["no significant reduction", "10 to 15 percent", "90 percent"]},
        {"context": "The AS01 adjuvant effect on dementia risk is", "truth": "protective regardless of target pathogen", "distractors": ["dependent on antigen specificity", "harmful via immune activation", "neutral"]},
        {"context": "APOE2 enhances Abeta clearance compared to APOE4 via", "truth": "better BBB transport efficiency", "distractors": ["identical transport mechanisms", "reduced clearance", "no effect on Abeta"]},
    ],
}


def mc_hinge_loss(adapter, lm_head, model, prompt, truth, distractors, tokenizer,
                  margin_target=3.0):
    def lp(text):
        prompt_ids = tokenizer.encode(prompt)
        full_ids = tokenizer.encode(prompt + text)
        n_prompt = len(prompt_ids)
        if len(full_ids) <= n_prompt:
            return mx.array(-1e9)
        tokens = mx.array(full_ids)[None, :]
        h = model.model(tokens)
        base_logits = lm_head(h)
        shifts = adapter(base_logits)
        shifts = shifts - shifts.mean(axis=-1, keepdims=True)
        logits = base_logits + shifts
        logits = t3.LOGIT_SOFTCAP * mx.tanh(logits / t3.LOGIT_SOFTCAP)
        total = mx.array(0.0)
        for i, tok_id in enumerate(full_ids[n_prompt:]):
            pos = n_prompt - 1 + i
            lv = logits[0, pos]
            lse = mx.log(mx.sum(mx.exp(lv - mx.max(lv))) + 1e-8) + mx.max(lv)
            total = total + lv[tok_id] - lse
        return total

    truth_lp = lp(f" {truth}")
    dist_lps = [lp(f" {d}") for d in distractors]
    best_dist = mx.max(mx.stack(dist_lps))
    loss = mx.maximum(mx.array(0.0), mx.array(margin_target) - (truth_lp - best_dist))
    return loss, float(truth_lp - best_dist)


def clip_grads(grads, max_norm=1.0):
    leaves = tree_flatten(grads)
    total_sq = sum(float(mx.sum(g ** 2)) for _, g in leaves)
    norm = total_sq ** 0.5
    if norm > max_norm:
        scale = max_norm / (norm + 1e-8)
        leaves = [(k, g * scale) for k, g in leaves]
    return tree_unflatten(leaves)


def train_cluster(adapter, optimizer, model, tokenizer, lm_head, examples,
                  steps_per_example=100, lr=4e-6, margin_target=3.0, cluster_name=""):
    loss_and_grad = mx.value_and_grad(mc_hinge_loss, argnums=0)

    total_steps = len(examples) * steps_per_example
    print(f"\n  Training cluster '{cluster_name}': {len(examples)} examples, {total_steps} steps")

    t0 = time.time()
    recent_margins = []
    step = 0
    for _ in range(steps_per_example):
        for ex in examples:
            ctx, truth, distractors = ex["context"], ex["truth"], ex["distractors"]
            prompt = ctx + ":"

            (loss_val, margin_val), grads = loss_and_grad(
                adapter, lm_head, model, prompt, truth, distractors, tokenizer, margin_target
            )
            grads = clip_grads(grads)
            optimizer.update(adapter, grads)
            mx.eval(adapter.parameters(), optimizer.state)

            recent_margins.append(margin_val)
            if len(recent_margins) > 50:
                recent_margins.pop(0)
            step += 1

    elapsed = time.time() - t0
    avg_margin = np.mean(recent_margins)
    print(f"    Completed in {elapsed:.0f}s  avg_margin={avg_margin:.3f}")
    return adapter


def eval_on_facts(adapter, lm_head, model, tokenizer, label=""):
    facts_path = os.path.join(ROOT, "problems", "hsv_apoe4_alzheimers_facts.json")
    with open(facts_path) as f:
        data = json.load(f)
    facts = data["facts"]

    print(f"\n  Evaluation ({label})")
    wins, margins = 0, []
    for fact in facts:
        ctx, truth, distractors = fact["context"], fact["truth"], fact["distractors"]
        prompt = ctx + ":"

        def adapted_lp(text):
            prompt_ids = tokenizer.encode(prompt)
            comp_ids = tokenizer.encode(text)
            full_ids = prompt_ids + comp_ids
            if not comp_ids:
                return -999.0
            tokens = mx.array(full_ids)[None, :]
            h = model.model(tokens)
            base_logits = lm_head(h)
            if adapter is not None:
                shifts = adapter(base_logits)
                shifts = shifts - shifts.mean(axis=-1, keepdims=True)
                logits = base_logits + shifts
                logits = t3.LOGIT_SOFTCAP * mx.tanh(logits / t3.LOGIT_SOFTCAP)
            else:
                logits = base_logits
            n_prompt = len(prompt_ids)
            total = 0.0
            for i, tok_id in enumerate(comp_ids):
                pos = n_prompt - 1 + i
                lv = np.array(logits[0, pos].astype(mx.float32))
                lse = float(np.log(np.sum(np.exp(lv - lv.max())) + 1e-8) + lv.max())
                total += float(lv[tok_id]) - lse
            return total

        truth_lp = adapted_lp(f" {truth}")
        dist_lps = [adapted_lp(f" {d}") for d in distractors]
        win = truth_lp > max(dist_lps)
        margin = truth_lp - max(dist_lps)
        wins += int(win)
        margins.append(margin)
        marker = "+" if win else "-"
        print(f"    {marker} {fact['id']:10s} margin={margin:+.2f}")

    mean_m = float(np.mean(margins))
    print(f"  Pass: {wins}/{len(facts)}  mean_margin={mean_m:+.2f}")
    return wins, len(facts), mean_m


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("\nLoading Qwen/Qwen3-4B-Base...")
    t0 = time.time()
    model, tokenizer = mlx_lm.load("Qwen/Qwen3-4B-Base")
    model.freeze()
    lm_head = t3.get_lm_head_fn(model)
    print(f"  Loaded in {time.time()-t0:.1f}s")

    # Create adapter
    vocab_size = model.model.embed_tokens.weight.shape[0]
    d_model = model.model.layers[0].self_attn.q_proj.weight.shape[0]
    cfg = SnapOnConfig(d_model=d_model, d_inner=64, n_layers=0,
                       n_heads=8, mode="logit", vocab_size=vocab_size)
    adapter = create_adapter(cfg)
    optimizer = optim.AdamW(learning_rate=4e-6, weight_decay=0.01)

    print("\n" + "="*60)
    print("  BASELINE")
    print("="*60)
    eval_on_facts(None, lm_head, model, tokenizer, label="baseline")

    # Staged training - train clusters sequentially
    cluster_order = ["apoe_structure", "epidemiology", "antivirals",
                     "inflammasome", "tau_viral", "microglia_trem2", "vaccines"]

    for i, cluster_name in enumerate(cluster_order, 1):
        print(f"\n{'='*60}")
        print(f"  STAGE {i}/{len(cluster_order)}: {cluster_name}")
        print("="*60)

        examples = CLUSTERS[cluster_name]
        adapter = train_cluster(adapter, optimizer, model, tokenizer, lm_head,
                               examples, steps_per_example=80, margin_target=3.0,
                               cluster_name=cluster_name)

        # Eval after each stage
        eval_on_facts(adapter, lm_head, model, tokenizer, label=f"after_{cluster_name}")

    print("\n" + "="*60)
    print("  FINAL EVALUATION")
    print("="*60)
    eval_on_facts(adapter, lm_head, model, tokenizer, label="staged_final")

    out_path = os.path.join(OUT_DIR, "hsv_apoe4_alzheimers_staged_adapter.npz")
    weights = dict(tree_flatten(adapter.parameters()))
    mx.savez(out_path, **weights)
    print(f"\n  Adapter saved: {out_path}")


if __name__ == "__main__":
    main()
