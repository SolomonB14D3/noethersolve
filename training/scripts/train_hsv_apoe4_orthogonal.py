#!/usr/bin/env python3
"""Orthogonal adapter training for HSV-APOE4-Alzheimer's domain.

Trains separate specialist adapters per cluster, then routes each fact
to its specialist at inference time. This solves representational
see-saws where concepts need to move in opposite directions.
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

# Define clusters - each gets its own orthogonal adapter
CLUSTERS = {
    "apoe_genetics": [
        {"context": "APOE2 at position 158 has amino acid", "truth": "Cysteine (Cys158)", "distractors": ["Arginine (Arg158)", "Glutamate (Glu158)", "Aspartate (Asp158)"]},
        {"context": "APOE4 at position 158 has amino acid", "truth": "Arginine (Arg158)", "distractors": ["Cysteine (Cys158)", "Glutamate (Glu158)", "Aspartate (Asp158)"]},
        {"context": "APOE2 homozygotes have AD risk compared to E3/E3", "truth": "reduced by approximately 90%", "distractors": ["unchanged", "reduced by 20-30%", "increased by 50%"]},
        {"context": "APOE2 enhances Abeta clearance compared to APOE4 via", "truth": "better BBB transport efficiency", "distractors": ["identical transport", "reduced clearance", "no effect"]},
    ],
    "epidemiology_risk": [
        {"context": "APOE4 carriers with HSV-1 reactivation have AD risk", "truth": "increased 3-4 fold (HR 3.28-3.68)", "distractors": ["unchanged from baseline", "increased 1.2-1.5 fold", "increased 10-20 fold"]},
        {"context": "In APOE4-negative individuals HSV-1 affects AD risk by", "truth": "no significant association (gene-virus interaction required)", "distractors": ["doubling the risk", "reducing risk", "same 3-4 fold increase"]},
        {"context": "The gene-virus interaction for HSV-AD requires", "truth": "both APOE4 AND HSV-1 together", "distractors": ["either alone sufficient", "only APOE4", "only HSV-1"]},
        {"context": "HSV-1 DNA in AD brains associates with", "truth": "APOE4 genotype specifically", "distractors": ["APOE2 genotype", "no genetic link", "APOE3 only"]},
    ],
    "antiviral_cns": [
        {"context": "Adibelivir (IM-250) brain to plasma ratio", "truth": "excellent at 0.5-4.0", "distractors": ["poor below 0.1", "moderate 0.1-0.2", "same as valacyclovir"]},
        {"context": "Pritelivir brain to plasma ratio", "truth": "limited below 0.1", "distractors": ["excellent 0.5-4.0", "moderate 0.2-0.5", "same as adibelivir"]},
        {"context": "VALAD trial valacyclovir in HSV+ early AD", "truth": "failed with worsened cognition", "distractors": ["showed benefit", "helped APOE4 subgroup", "reduced amyloid"]},
        {"context": "HSV helicase-primase inhibitors vs nucleoside analogs", "truth": "do not require viral TK activation", "distractors": ["need phosphorylation", "less potent", "same mechanism"]},
        {"context": "Valacyclovir failed in AD because of", "truth": "poor CNS penetration", "distractors": ["viral resistance", "wrong dose", "patient compliance"]},
    ],
    "nlrp3_inflammasome": [
        {"context": "APOE4 activates NLRP3 via", "truth": "NF-kB pathway upregulation", "distractors": ["direct NLRP3 binding", "calcium channels", "autophagy"]},
        {"context": "MCC950 in HSV-infected 5xFAD mice", "truth": "reduced amyloid and cognitive decline", "distractors": ["no effect", "worsened inflammation", "only affected tau"]},
        {"context": "NLRP3 inflammasome produces", "truth": "IL-1beta and IL-18 via caspase-1", "distractors": ["only TNF-alpha", "only interferon-gamma", "no cytokines"]},
        {"context": "APOE4 microglia lipid droplets trigger", "truth": "inflammasome activation", "distractors": ["neuroprotection", "enhanced phagocytosis", "apoptosis"]},
    ],
    "tau_antimicrobial": [
        {"context": "Tau phosphorylation after HSV-1 infection", "truth": "antimicrobial defense binding viral capsids", "distractors": ["purely pathological", "enhances viral replication", "triggers apoptosis"]},
        {"context": "p-tau neutralizes HSV-1 by", "truth": "directly binding capsids", "distractors": ["blocking entry", "triggering interferon", "autophagy"]},
        {"context": "HSV-1 ICP27 in AD brains colocalizes with", "truth": "p-tau not amyloid-beta", "distractors": ["amyloid-beta", "both equally", "neither"]},
        {"context": "HSV-1 upregulates gamma-secretase", "truth": "PSEN1 and PSEN2", "distractors": ["APP only", "BACE1 only", "neither"]},
    ],
    "microglia_trem2": [
        {"context": "TREM2 R47H variant HSV-1 response", "truth": "impaired antiviral increased replication", "distractors": ["enhanced clearance", "no effect", "blocked BBB entry"]},
        {"context": "TREM2 depletion with HSV-1", "truth": "elevated viral replication", "distractors": ["reduced replication", "no change", "complete clearance"]},
        {"context": "APOE4 microglia immune response", "truth": "exaggerated with reduced Abeta clearance", "distractors": ["suppressed normal clearance", "enhanced clearance", "identical to APOE2"]},
        {"context": "APOE-I3 ASOs in tauopathy mice", "truth": "reduced neurodegeneration 50%", "distractors": ["no effect", "worsened tau", "only peripheral"]},
    ],
    "vaccines_adjuvants": [
        {"context": "Recombinant zoster vaccine dementia reduction", "truth": "approximately 51% over age 65", "distractors": ["no significant reduction", "10-15%", "90%"]},
        {"context": "AS01 adjuvant dementia effect", "truth": "protective regardless of pathogen", "distractors": ["antigen dependent", "harmful", "neutral"]},
    ],
}

# Map each fact ID to its cluster for routing
FACT_TO_CLUSTER = {
    "haa01": "epidemiology_risk",
    "haa02": "epidemiology_risk",
    "haa03": "apoe_genetics",
    "haa04": "antiviral_cns",
    "haa05": "antiviral_cns",
    "haa06": "nlrp3_inflammasome",
    "haa07": "tau_antimicrobial",
    "haa08": "microglia_trem2",
    "haa09": "vaccines_adjuvants",
    "haa10": "nlrp3_inflammasome",
    "haa11": "antiviral_cns",
    "haa12": "apoe_genetics",
    "haa13": "vaccines_adjuvants",
    "haa14": "tau_antimicrobial",
    "haa15": "microglia_trem2",
    "haa16": "microglia_trem2",
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


def train_specialist(model, tokenizer, lm_head, examples, cluster_name,
                     steps=600, lr=4e-6, d_inner=64, margin_target=3.0):
    """Train a specialist adapter for one cluster."""
    vocab_size = model.model.embed_tokens.weight.shape[0]
    d_model = model.model.layers[0].self_attn.q_proj.weight.shape[0]
    cfg = SnapOnConfig(d_model=d_model, d_inner=d_inner, n_layers=0,
                       n_heads=8, mode="logit", vocab_size=vocab_size)
    adapter = create_adapter(cfg)
    optimizer = optim.AdamW(learning_rate=lr, weight_decay=0.01)

    loss_and_grad = mx.value_and_grad(mc_hinge_loss, argnums=0)

    print(f"\n  Training {cluster_name} specialist: {len(examples)} examples, {steps} steps")

    t0 = time.time()
    recent_margins = []
    for step in range(steps):
        ex = examples[step % len(examples)]
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

    elapsed = time.time() - t0
    avg_margin = np.mean(recent_margins)
    print(f"    Completed in {elapsed:.0f}s  avg_margin={avg_margin:.3f}")
    return adapter


def eval_with_routing(adapters, lm_head, model, tokenizer):
    """Evaluate using routed orthogonal adapters."""
    facts_path = os.path.join(ROOT, "problems", "hsv_apoe4_alzheimers_facts.json")
    with open(facts_path) as f:
        data = json.load(f)
    facts = data["facts"]

    print(f"\n  Routed orthogonal evaluation")
    wins, margins = 0, []
    for fact in facts:
        ctx, truth, distractors = fact["context"], fact["truth"], fact["distractors"]
        prompt = ctx + ":"
        fact_id = fact["id"]

        # Route to specialist adapter
        cluster = FACT_TO_CLUSTER.get(fact_id, "apoe_genetics")
        adapter = adapters.get(cluster)

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
        print(f"    {marker} {fact_id:10s} [{cluster:20s}] margin={margin:+.2f}")

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

    # Train specialist adapter for each cluster
    adapters = {}
    for cluster_name, examples in CLUSTERS.items():
        print(f"\n{'='*60}")
        print(f"  TRAINING: {cluster_name}")
        print("="*60)
        adapter = train_specialist(model, tokenizer, lm_head, examples, cluster_name,
                                  steps=600, lr=4e-6, margin_target=3.0)
        adapters[cluster_name] = adapter

        # Save individual adapter
        out_path = os.path.join(OUT_DIR, f"hsv_apoe4_{cluster_name}_adapter.npz")
        weights = dict(tree_flatten(adapter.parameters()))
        mx.savez(out_path, **weights)
        print(f"    Saved: {out_path}")

    # Final routed evaluation
    print("\n" + "="*60)
    print("  FINAL: ROUTED ORTHOGONAL EVALUATION")
    print("="*60)
    eval_with_routing(adapters, lm_head, model, tokenizer)

    # Save routing config
    routing_config = {
        "domain": "hsv_apoe4_alzheimers",
        "clusters": list(CLUSTERS.keys()),
        "fact_to_cluster": FACT_TO_CLUSTER,
        "adapter_files": {
            cluster: f"hsv_apoe4_{cluster}_adapter.npz"
            for cluster in CLUSTERS.keys()
        }
    }
    config_path = os.path.join(OUT_DIR, "hsv_apoe4_orthogonal_routing.json")
    with open(config_path, "w") as f:
        json.dump(routing_config, f, indent=2)
    print(f"\n  Routing config saved: {config_path}")


if __name__ == "__main__":
    main()
