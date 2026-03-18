"""NoetherSolve — find where LLM knowledge ends, build verified tools, serve
them to any AI agent via MCP.

The pipeline: find gaps → flip facts → build tool → add to MCP server.
Every tool added makes every connected agent smarter.

Emmy Noether proved every continuous symmetry corresponds to a conserved quantity.
NoetherSolve finds where LLMs fail to recognize those quantities, builds verified
computational tools for the right answers, and exposes them via Model Context
Protocol (MCP) — 230 tools currently serving physics, math, genetics, complexity
theory, enzyme kinetics, quantum mechanics, pharmacokinetics, drug interactions,
organic chemistry, elliptic curves, intersection theory (adjunction, blow-ups,
ruled surfaces, toric varieties), information theory, topological phases, ergodic
theory, optimization convergence, numerical PDEs, MHD conservation, GR constraints
(ADM formalism), seismic waves (velocity, moduli, reflection), plasma physics
(adiabatic invariants μ, J, Φ), and LLM science.

This module uses lazy loading for fast startup. Imports happen on first access.
"""

__version__ = "1.22.0"

# ---------------------------------------------------------------------------
# Lazy Loading Infrastructure
# ---------------------------------------------------------------------------
# Uses __getattr__ pattern for PEP 562 lazy module loading.
# Startup: only this file is parsed. Actual imports happen on first use.
# This reduces startup from ~2s to <0.3s.

import importlib
import sys
from typing import Any

# Map of exported names to their source modules and import type
# Format: "name": ("module", "import_name" or None for same name)
_LAZY_IMPORTS: dict[str, tuple[str, str | None]] = {
    # monitor.py
    "frac_var": ("noethersolve.monitor", None),
    "MonitorReport": ("noethersolve.monitor", None),
    "VortexMonitor": ("noethersolve.monitor", None),
    "ChemicalMonitor": ("noethersolve.monitor", None),
    "GravityMonitor": ("noethersolve.monitor", None),

    # monitor_em.py
    "EMMonitor": ("noethersolve.monitor_em", None),

    # validate.py
    "validate_integrator": ("noethersolve.validate", None),
    "compare_configs": ("noethersolve.validate", None),

    # audit_chem.py
    "audit_network": ("noethersolve.audit_chem", None),
    "AuditReport": ("noethersolve.audit_chem", None),

    # audit_facts.py
    "audit_facts": ("noethersolve.audit_facts", None),
    "FactAuditReport": ("noethersolve.audit_facts", None),

    # hamiltonian.py
    "HamiltonianMonitor": ("noethersolve.hamiltonian", None),
    "HamiltonianReport": ("noethersolve.hamiltonian", None),
    "harmonic_oscillator": ("noethersolve.hamiltonian", None),
    "kepler_2d": ("noethersolve.hamiltonian", None),
    "henon_heiles": ("noethersolve.hamiltonian", None),
    "coupled_oscillators": ("noethersolve.hamiltonian", None),

    # learner.py
    "InvariantLearner": ("noethersolve.learner", None),
    "LearnerReport": ("noethersolve.learner", None),

    # knot.py
    "KnotMonitor": ("noethersolve.knot", None),
    "KnotReport": ("noethersolve.knot", None),
    "KnotDiagram": ("noethersolve.knot", None),
    "unknot": ("noethersolve.knot", None),
    "trefoil": ("noethersolve.knot", None),
    "figure_eight_knot": ("noethersolve.knot", None),

    # audit_sequence.py
    "audit_sequence": ("noethersolve.audit_sequence", None),
    "SequenceReport": ("noethersolve.audit_sequence", None),
    "SequenceIssue": ("noethersolve.audit_sequence", None),
    "gc_content": ("noethersolve.audit_sequence", None),
    "cpg_observed_expected": ("noethersolve.audit_sequence", None),

    # crispr.py
    "score_guide": ("noethersolve.crispr", None),
    "score_guides": ("noethersolve.crispr", None),
    "check_offtarget_pair": ("noethersolve.crispr", None),
    "GuideReport": ("noethersolve.crispr", None),
    "GuideIssue": ("noethersolve.crispr", None),

    # pipeline.py
    "validate_pipeline": ("noethersolve.pipeline", None),
    "validate_pipeline_dict": ("noethersolve.pipeline", None),
    "TherapyDesign": ("noethersolve.pipeline", None),
    "PipelineReport": ("noethersolve.pipeline", None),
    "PipelineIssue": ("noethersolve.pipeline", None),

    # aggregation.py
    "predict_aggregation": ("noethersolve.aggregation", None),
    "AggregationReport": ("noethersolve.aggregation", None),
    "AggregationIssue": ("noethersolve.aggregation", None),
    "KYTE_DOOLITTLE": ("noethersolve.aggregation", None),
    "AGGRESCAN": ("noethersolve.aggregation", None),

    # splice.py
    "score_donor": ("noethersolve.splice", None),
    "score_acceptor": ("noethersolve.splice", None),
    "scan_splice_sites": ("noethersolve.splice", None),
    "pyrimidine_tract_score": ("noethersolve.splice", None),
    "SpliceSiteReport": ("noethersolve.splice", None),

    # pk_model.py
    "one_compartment_iv": ("noethersolve.pk_model", None),
    "one_compartment_oral": ("noethersolve.pk_model", None),
    "half_life": ("noethersolve.pk_model", None),
    "steady_state": ("noethersolve.pk_model", None),
    "auc_single_dose": ("noethersolve.pk_model", None),
    "dose_adjustment": ("noethersolve.pk_model", None),
    "IVBolusReport": ("noethersolve.pk_model", None),
    "OralDosingReport": ("noethersolve.pk_model", None),
    "HalfLifeReport": ("noethersolve.pk_model", None),
    "SteadyStateReport": ("noethersolve.pk_model", None),
    "AUCReport": ("noethersolve.pk_model", None),
    "DoseAdjustmentReport": ("noethersolve.pk_model", None),

    # complexity.py
    "audit_complexity": ("noethersolve.complexity", None),
    "check_inclusion": ("noethersolve.complexity", None),
    "check_completeness": ("noethersolve.complexity", None),
    "get_class_info": ("noethersolve.complexity", None),
    "ComplexityReport": ("noethersolve.complexity", None),
    "ComplexityIssue": ("noethersolve.complexity", None),

    # conjecture_status.py
    "check_conjecture": ("noethersolve.conjecture_status", None),
    "check_claim": ("noethersolve.conjecture_status", None),
    "list_conjectures": ("noethersolve.conjecture_status", None),
    "get_conjecture": ("noethersolve.conjecture_status", None),
    "ConjectureReport": ("noethersolve.conjecture_status", None),
    "ConjectureIssue": ("noethersolve.conjecture_status", None),
    "ConjectureInfo": ("noethersolve.conjecture_status", None),

    # proof_barriers.py
    "check_barriers": ("noethersolve.proof_barriers", None),
    "list_barriers": ("noethersolve.proof_barriers", None),
    "list_techniques": ("noethersolve.proof_barriers", None),
    "get_barrier": ("noethersolve.proof_barriers", None),
    "what_works_for": ("noethersolve.proof_barriers", None),
    "BarrierReport": ("noethersolve.proof_barriers", None),
    "BarrierIssue": ("noethersolve.proof_barriers", None),
    "BarrierInfo": ("noethersolve.proof_barriers", None),

    # number_theory.py
    "verify_goldbach": ("noethersolve.number_theory", None),
    "verify_collatz": ("noethersolve.number_theory", None),
    "verify_twin_primes": ("noethersolve.number_theory", None),
    "check_abc_triple": ("noethersolve.number_theory", None),
    "verify_legendre": ("noethersolve.number_theory", None),
    "prime_gap_analysis": ("noethersolve.number_theory", None),
    "is_prime": ("noethersolve.number_theory", None),
    "prime_sieve": ("noethersolve.number_theory", None),
    "radical": ("noethersolve.number_theory", None),
    "GoldbachReport": ("noethersolve.number_theory", None),
    "CollatzReport": ("noethersolve.number_theory", None),
    "TwinPrimeReport": ("noethersolve.number_theory", None),
    "ABCReport": ("noethersolve.number_theory", None),
    "LegendreReport": ("noethersolve.number_theory", None),
    "PrimeGapReport": ("noethersolve.number_theory", None),

    # reductions.py
    "validate_chain": ("noethersolve.reductions", None),
    "check_reduction": ("noethersolve.reductions", None),
    "strongest_reduction": ("noethersolve.reductions", None),
    "list_known_reductions": ("noethersolve.reductions", None),
    "get_reduction_info": ("noethersolve.reductions", None),
    "ChainReport": ("noethersolve.reductions", None),
    "ChainIssue": ("noethersolve.reductions", None),
    "ReductionResult": ("noethersolve.reductions", None),

    # pde_regularity.py
    "check_sobolev_embedding": ("noethersolve.pde_regularity", None),
    "check_pde_regularity": ("noethersolve.pde_regularity", None),
    "critical_exponent": ("noethersolve.pde_regularity", None),
    "check_blowup": ("noethersolve.pde_regularity", None),
    "sobolev_conjugate": ("noethersolve.pde_regularity", None),
    "EmbeddingReport": ("noethersolve.pde_regularity", None),
    "RegularityReport": ("noethersolve.pde_regularity", None),
    "CriticalExponentReport": ("noethersolve.pde_regularity", None),
    "PDEBlowupReport": ("noethersolve.pde_regularity", "BlowupReport"),
    "EmbeddingIssue": ("noethersolve.pde_regularity", None),

    # llm_claims.py
    "audit_llm_claims": ("noethersolve.llm_claims", None),
    "check_llm_claim": ("noethersolve.llm_claims", None),
    "check_benchmark_score": ("noethersolve.llm_claims", None),
    "chinchilla_optimal": ("noethersolve.llm_claims", None),
    "get_llm_topic": ("noethersolve.llm_claims", None),
    "list_llm_topics": ("noethersolve.llm_claims", None),
    "list_domains": ("noethersolve.llm_claims", None),
    "LLMClaimReport": ("noethersolve.llm_claims", None),
    "LLMClaimResult": ("noethersolve.llm_claims", None),
    "LLMClaimIssue": ("noethersolve.llm_claims", None),
    "LLMTopicInfo": ("noethersolve.llm_claims", None),

    # control.py
    "simulate_pid": ("noethersolve.control", None),
    "analyze_stability": ("noethersolve.control", None),
    "PIDReport": ("noethersolve.control", None),
    "StabilityReport": ("noethersolve.control", None),

    # isolation.py
    "check_isolation": ("noethersolve.isolation", None),
    "analyze_schedule": ("noethersolve.isolation", None),
    "list_anomalies": ("noethersolve.isolation", None),
    "IsolationReport": ("noethersolve.isolation", None),
    "ScheduleReport": ("noethersolve.isolation", None),

    # quantum_circuit.py
    "simulate_circuit": ("noethersolve.quantum_circuit", None),
    "measure_state": ("noethersolve.quantum_circuit", None),
    "CircuitReport": ("noethersolve.quantum_circuit", None),

    # chemistry_calc.py
    "nernst_equation": ("noethersolve.chemistry_calc", None),
    "henderson_hasselbalch": ("noethersolve.chemistry_calc", None),
    "crystal_field_splitting": ("noethersolve.chemistry_calc", None),
    "band_gap_analysis": ("noethersolve.chemistry_calc", None),
    "balance_redox": ("noethersolve.chemistry_calc", None),
    "NernstReport": ("noethersolve.chemistry_calc", None),
    "BufferReport": ("noethersolve.chemistry_calc", None),
    "CrystalFieldReport": ("noethersolve.chemistry_calc", None),
    "BandGapReport": ("noethersolve.chemistry_calc", None),

    # crypto_calc.py
    "security_level": ("noethersolve.crypto_calc", None),
    "birthday_bound": ("noethersolve.crypto_calc", None),
    "rsa_key_analysis": ("noethersolve.crypto_calc", None),
    "cipher_mode_analysis": ("noethersolve.crypto_calc", None),
    "SecurityLevelReport": ("noethersolve.crypto_calc", None),
    "BirthdayBoundReport": ("noethersolve.crypto_calc", None),
    "RSAKeyReport": ("noethersolve.crypto_calc", None),
    "CipherModeReport": ("noethersolve.crypto_calc", None),

    # finance_calc.py
    "black_scholes": ("noethersolve.finance_calc", None),
    "put_call_parity": ("noethersolve.finance_calc", None),
    "nash_equilibrium_2x2": ("noethersolve.finance_calc", None),
    "present_value": ("noethersolve.finance_calc", None),
    "future_value": ("noethersolve.finance_calc", None),
    "BlackScholesReport": ("noethersolve.finance_calc", None),
    "PutCallParityReport": ("noethersolve.finance_calc", None),
    "NashEquilibriumReport": ("noethersolve.finance_calc", None),
    "PresentValueReport": ("noethersolve.finance_calc", None),

    # distributed_calc.py
    "quorum_calc": ("noethersolve.distributed_calc", None),
    "byzantine_threshold": ("noethersolve.distributed_calc", None),
    "vector_clock_compare": ("noethersolve.distributed_calc", None),
    "consistency_model": ("noethersolve.distributed_calc", None),
    "gossip_convergence": ("noethersolve.distributed_calc", None),
    "QuorumReport": ("noethersolve.distributed_calc", None),
    "ByzantineReport": ("noethersolve.distributed_calc", None),
    "VectorClockReport": ("noethersolve.distributed_calc", None),
    "ConsistencyReport": ("noethersolve.distributed_calc", None),
    "GossipReport": ("noethersolve.distributed_calc", None),

    # network_calc.py
    "bandwidth_delay_product": ("noethersolve.network_calc", None),
    "tcp_throughput": ("noethersolve.network_calc", None),
    "subnet_calc": ("noethersolve.network_calc", None),
    "ip_fragmentation": ("noethersolve.network_calc", None),
    "congestion_window": ("noethersolve.network_calc", None),
    "BandwidthDelayReport": ("noethersolve.network_calc", None),
    "TCPThroughputReport": ("noethersolve.network_calc", None),
    "SubnetReport": ("noethersolve.network_calc", None),
    "FragmentationReport": ("noethersolve.network_calc", None),
    "CongestionWindowReport": ("noethersolve.network_calc", None),

    # os_calc.py
    "page_table_calc": ("noethersolve.os_calc", None),
    "schedule_fcfs": ("noethersolve.os_calc", None),
    "schedule_sjf": ("noethersolve.os_calc", None),
    "schedule_round_robin": ("noethersolve.os_calc", None),
    "detect_deadlock": ("noethersolve.os_calc", None),
    "tlb_analysis": ("noethersolve.os_calc", None),
    "context_switch_cost": ("noethersolve.os_calc", None),
    "PageTableReport": ("noethersolve.os_calc", None),
    "SchedulingReport": ("noethersolve.os_calc", None),
    "DeadlockReport": ("noethersolve.os_calc", None),
    "TLBReport": ("noethersolve.os_calc", None),
    "ContextSwitchReport": ("noethersolve.os_calc", None),

    # enzyme_kinetics.py
    "michaelis_menten": ("noethersolve.enzyme_kinetics", None),
    "inhibition": ("noethersolve.enzyme_kinetics", None),
    "catalytic_efficiency": ("noethersolve.enzyme_kinetics", None),
    "lineweaver_burk": ("noethersolve.enzyme_kinetics", None),
    "ph_rate_profile": ("noethersolve.enzyme_kinetics", None),
    "cooperativity": ("noethersolve.enzyme_kinetics", None),
    "MMReport": ("noethersolve.enzyme_kinetics", None),
    "InhibitionReport": ("noethersolve.enzyme_kinetics", None),
    "EfficiencyReport": ("noethersolve.enzyme_kinetics", None),
    "LineweaverBurkReport": ("noethersolve.enzyme_kinetics", None),
    "PHProfileReport": ("noethersolve.enzyme_kinetics", None),
    "CooperativityReport": ("noethersolve.enzyme_kinetics", None),
    "DIFFUSION_LIMIT": ("noethersolve.enzyme_kinetics", None),

    # reaction_engine.py (optional - RDKit may not be installed)
    "analyze_molecule": ("noethersolve.reaction_engine", None),
    "predict_selectivity": ("noethersolve.reaction_engine", None),
    "predict_mechanism": ("noethersolve.reaction_engine", None),
    "validate_synthesis": ("noethersolve.reaction_engine", None),
    "check_baldwin": ("noethersolve.reaction_engine", None),
    "check_woodward_hoffmann": ("noethersolve.reaction_engine", None),
    "list_mayr_nucleophiles": ("noethersolve.reaction_engine", None),
    "list_mayr_electrophiles": ("noethersolve.reaction_engine", None),
    "list_reaction_templates": ("noethersolve.reaction_engine", None),
    "get_reaction_template": ("noethersolve.reaction_engine", None),
    "MoleculeAnalysis": ("noethersolve.reaction_engine", None),
    "SelectivityReport": ("noethersolve.reaction_engine", None),
    "MechanismReport": ("noethersolve.reaction_engine", None),
    "SynthesisReport": ("noethersolve.reaction_engine", None),
    "BaldwinReport": ("noethersolve.reaction_engine", None),
    "WoodwardHoffmannReport": ("noethersolve.reaction_engine", None),

    # qm_calculator.py
    "particle_in_box": ("noethersolve.qm_calculator", None),
    "hydrogen_energy": ("noethersolve.qm_calculator", None),
    "uncertainty_check": ("noethersolve.qm_calculator", None),
    "tunneling_probability": ("noethersolve.qm_calculator", None),
    # Note: harmonic_oscillator is also in hamiltonian.py - using qm_calculator version
    "angular_momentum_addition": ("noethersolve.qm_calculator", None),
    "ParticleInBoxReport": ("noethersolve.qm_calculator", None),
    "HydrogenEnergyReport": ("noethersolve.qm_calculator", None),
    "UncertaintyReport": ("noethersolve.qm_calculator", None),
    "TunnelingReport": ("noethersolve.qm_calculator", None),
    "HarmonicOscillatorReport": ("noethersolve.qm_calculator", None),
    "AngularMomentumReport": ("noethersolve.qm_calculator", None),

    # elliptic_curves.py
    "is_on_curve": ("noethersolve.elliptic_curves", None),
    "point_add": ("noethersolve.elliptic_curves", None),
    "point_double": ("noethersolve.elliptic_curves", None),
    "point_negate": ("noethersolve.elliptic_curves", None),
    "scalar_mult": ("noethersolve.elliptic_curves", None),
    "point_order": ("noethersolve.elliptic_curves", None),
    "discriminant": ("noethersolve.elliptic_curves", None),
    "j_invariant": ("noethersolve.elliptic_curves", None),
    "is_singular": ("noethersolve.elliptic_curves", None),
    "hasse_bounds": ("noethersolve.elliptic_curves", None),
    "count_points_naive": ("noethersolve.elliptic_curves", None),
    "verify_hasse": ("noethersolve.elliptic_curves", None),
    "find_points": ("noethersolve.elliptic_curves", None),
    "is_valid_torsion_order": ("noethersolve.elliptic_curves", None),
    "analyze_curve": ("noethersolve.elliptic_curves", None),
    "analyze_point_arithmetic": ("noethersolve.elliptic_curves", None),
    "EllipticCurveReport": ("noethersolve.elliptic_curves", None),
    "PointArithmeticReport": ("noethersolve.elliptic_curves", None),

    # intersection_theory.py
    "bezout_intersection": ("noethersolve.intersection_theory", None),
    "BezoutReport": ("noethersolve.intersection_theory", None),
    "genus_degree_formula": ("noethersolve.intersection_theory", None),
    "GenusReport": ("noethersolve.intersection_theory", None),
    "self_intersection_Pn": ("noethersolve.intersection_theory", None),
    "self_intersection_line_P2": ("noethersolve.intersection_theory", None),
    "self_intersection_exceptional": ("noethersolve.intersection_theory", None),
    "SelfIntersectionReport": ("noethersolve.intersection_theory", None),
    "canonical_P2": ("noethersolve.intersection_theory", None),
    "canonical_cubic_surface": ("noethersolve.intersection_theory", None),
    "del_pezzo_degree": ("noethersolve.intersection_theory", None),
    "CanonicalReport": ("noethersolve.intersection_theory", None),
    "noether_formula": ("noethersolve.intersection_theory", None),
    "NoetherReport": ("noethersolve.intersection_theory", None),
    "lines_on_cubic_surface": ("noethersolve.intersection_theory", None),
    "bitangents_to_quartic": ("noethersolve.intersection_theory", None),
    "conics_through_5_points": ("noethersolve.intersection_theory", None),
    "lines_meeting_4_general_lines_P3": ("noethersolve.intersection_theory", None),
    "plane_cubics_through_9_points": ("noethersolve.intersection_theory", None),
    "rational_curves_on_quintic_threefold": ("noethersolve.intersection_theory", None),
    "EnumerativeReport": ("noethersolve.intersection_theory", None),
    "chow_ring_Pn": ("noethersolve.intersection_theory", None),
    "segre_embedding_degree": ("noethersolve.intersection_theory", None),
    "intersection_multiplicity_formula": ("noethersolve.intersection_theory", None),
    "compute_multiplicity_smooth_transverse": ("noethersolve.intersection_theory", None),
    "compute_multiplicity_tangent": ("noethersolve.intersection_theory", None),
    "adjunction_formula": ("noethersolve.intersection_theory", None),
    "adjunction_complete_intersection": ("noethersolve.intersection_theory", None),
    "AdjunctionReport": ("noethersolve.intersection_theory", None),
    "blowup_K_squared": ("noethersolve.intersection_theory", None),
    "blowup_P2": ("noethersolve.intersection_theory", None),
    "blowup_transform_divisor": ("noethersolve.intersection_theory", None),
    "BlowupReport": ("noethersolve.intersection_theory", None),
    "ruled_surface": ("noethersolve.intersection_theory", None),
    "hirzebruch_surface": ("noethersolve.intersection_theory", None),
    "RuledSurfaceReport": ("noethersolve.intersection_theory", None),
    "toric_canonical": ("noethersolve.intersection_theory", None),
    "toric_Pn_canonical": ("noethersolve.intersection_theory", None),
    "ToricCanonicalReport": ("noethersolve.intersection_theory", None),

    # information_theory.py
    "binary_entropy": ("noethersolve.information_theory", None),
    "entropy": ("noethersolve.information_theory", None),
    "relative_entropy": ("noethersolve.information_theory", None),
    "mutual_information": ("noethersolve.information_theory", None),
    "capacity_bsc": ("noethersolve.information_theory", None),
    "capacity_bec": ("noethersolve.information_theory", None),
    "capacity_awgn": ("noethersolve.information_theory", None),
    "capacity_z_channel": ("noethersolve.information_theory", None),
    "ChannelCapacityReport": ("noethersolve.information_theory", None),
    "rate_distortion_binary": ("noethersolve.information_theory", None),
    "rate_distortion_gaussian": ("noethersolve.information_theory", None),
    "RateDistortionReport": ("noethersolve.information_theory", None),
    "source_coding_bound": ("noethersolve.information_theory", None),
    "SourceCodingReport": ("noethersolve.information_theory", None),
    "mac_capacity_region_2user": ("noethersolve.information_theory", None),
    "MACRegionReport": ("noethersolve.information_theory", None),
    "check_data_processing": ("noethersolve.information_theory", None),
    "DataProcessingReport": ("noethersolve.information_theory", None),
    "fano_inequality": ("noethersolve.information_theory", None),
    "FanoReport": ("noethersolve.information_theory", None),
    "typical_set_bounds": ("noethersolve.information_theory", None),
    "TypicalSetReport": ("noethersolve.information_theory", None),

    # drug_interactions.py
    "get_drug_profile": ("noethersolve.drug_interactions", None),
    "DrugProfileReport": ("noethersolve.drug_interactions", None),
    "get_cyp_info": ("noethersolve.drug_interactions", None),
    "CYPInfoReport": ("noethersolve.drug_interactions", None),
    "check_interaction": ("noethersolve.drug_interactions", None),
    "predict_auc_change": ("noethersolve.drug_interactions", None),
    "InteractionReport": ("noethersolve.drug_interactions", None),
    "check_pharmacogenomics": ("noethersolve.drug_interactions", None),
    "PharmacogenomicsReport": ("noethersolve.drug_interactions", None),
    "list_cyp_enzymes": ("noethersolve.drug_interactions", None),
    "list_substrates": ("noethersolve.drug_interactions", None),
    "list_inhibitors": ("noethersolve.drug_interactions", None),
    "list_inducers": ("noethersolve.drug_interactions", None),
    "Strength": ("noethersolve.drug_interactions", None),
    "InteractionType": ("noethersolve.drug_interactions", None),
    "Severity": ("noethersolve.drug_interactions", None),

    # epidemiology.py
    "herd_immunity_threshold": ("noethersolve.epidemiology", None),
    "reproduction_number": ("noethersolve.epidemiology", None),
    "doubling_time": ("noethersolve.epidemiology", None),
    "attack_rate": ("noethersolve.epidemiology", None),
    "sir_model": ("noethersolve.epidemiology", None),
    "vaccine_impact": ("noethersolve.epidemiology", None),
    "generation_interval": ("noethersolve.epidemiology", None),
    "seir_parameters": ("noethersolve.epidemiology", None),
    "get_disease_R0": ("noethersolve.epidemiology", None),
    "list_diseases": ("noethersolve.epidemiology", None),
    "HerdImmunityReport": ("noethersolve.epidemiology", None),
    "ReproductionNumberReport": ("noethersolve.epidemiology", None),
    "DoublingTimeReport": ("noethersolve.epidemiology", None),
    "AttackRateReport": ("noethersolve.epidemiology", None),
    "SIRReport": ("noethersolve.epidemiology", None),
    "VaccineImpactReport": ("noethersolve.epidemiology", None),
    "GenerationIntervalReport": ("noethersolve.epidemiology", None),

    # radiative_transfer.py
    "radiative_forcing": ("noethersolve.radiative_transfer", None),
    "planck_response": ("noethersolve.radiative_transfer", None),
    "climate_sensitivity": ("noethersolve.radiative_transfer", None),
    "stefan_boltzmann": ("noethersolve.radiative_transfer", None),
    "effective_temperature": ("noethersolve.radiative_transfer", None),
    "analyze_feedback": ("noethersolve.radiative_transfer", None),
    "list_feedbacks": ("noethersolve.radiative_transfer", None),
    "RadiativeForcingReport": ("noethersolve.radiative_transfer", None),
    "PlanckResponseReport": ("noethersolve.radiative_transfer", None),
    "ClimateSensitivityReport": ("noethersolve.radiative_transfer", None),
    "StefanBoltzmannReport": ("noethersolve.radiative_transfer", None),
    "EffectiveTemperatureReport": ("noethersolve.radiative_transfer", None),
    "FeedbackAnalysisReport": ("noethersolve.radiative_transfer", None),

    # turbulence.py
    "kolmogorov_45_law": ("noethersolve.turbulence", None),
    "energy_spectrum": ("noethersolve.turbulence", None),
    "length_scales": ("noethersolve.turbulence", None),
    "structure_function_exponent": ("noethersolve.turbulence", None),
    "intermittency_analysis": ("noethersolve.turbulence", None),
    "is_in_inertial_range": ("noethersolve.turbulence", None),
    "inertial_range_extent": ("noethersolve.turbulence", None),
    "Kolmogorov45Report": ("noethersolve.turbulence", None),
    "EnergySpectrumReport": ("noethersolve.turbulence", None),
    "LengthScalesReport": ("noethersolve.turbulence", None),
    "StructureFunctionReport": ("noethersolve.turbulence", None),
    "IntermittencyReport": ("noethersolve.turbulence", None),

    # topological_invariants.py
    "chern_number": ("noethersolve.topological_invariants", None),
    "z2_invariant": ("noethersolve.topological_invariants", None),
    "bulk_boundary_correspondence": ("noethersolve.topological_invariants", None),
    "quantum_hall": ("noethersolve.topological_invariants", None),
    "berry_phase": ("noethersolve.topological_invariants", None),
    "topological_classification": ("noethersolve.topological_invariants", None),
    "list_symmetry_classes": ("noethersolve.topological_invariants", None),
    "ChernNumberReport": ("noethersolve.topological_invariants", None),
    "Z2InvariantReport": ("noethersolve.topological_invariants", None),
    "BulkBoundaryReport": ("noethersolve.topological_invariants", None),
    "QuantumHallReport": ("noethersolve.topological_invariants", None),
    "BerryPhaseReport": ("noethersolve.topological_invariants", None),
    "TopologicalClassReport": ("noethersolve.topological_invariants", None),
    "VON_KLITZING": ("noethersolve.topological_invariants", None),
    "CONDUCTANCE_QUANTUM": ("noethersolve.topological_invariants", None),

    # ergodic_theory.py
    "classify_system": ("noethersolve.ergodic_theory", None),
    "compare_levels": ("noethersolve.ergodic_theory", None),
    "lyapunov_analysis": ("noethersolve.ergodic_theory", None),
    "entropy_analysis": ("noethersolve.ergodic_theory", None),
    "poincare_recurrence": ("noethersolve.ergodic_theory", None),
    "mixing_rate": ("noethersolve.ergodic_theory", None),
    "list_dynamical_systems": ("noethersolve.ergodic_theory", "list_systems"),
    "list_ergodic_levels": ("noethersolve.ergodic_theory", "list_levels"),
    "is_stronger": ("noethersolve.ergodic_theory", None),
    "implies": ("noethersolve.ergodic_theory", None),
    "HierarchyReport": ("noethersolve.ergodic_theory", None),
    "LyapunovReport": ("noethersolve.ergodic_theory", None),
    "EntropyReport": ("noethersolve.ergodic_theory", None),
    "RecurrenceReport": ("noethersolve.ergodic_theory", None),
    "ComparisonReport": ("noethersolve.ergodic_theory", None),
    "MixingRateReport": ("noethersolve.ergodic_theory", None),

    # optimization_convergence.py
    "gradient_descent_rate": ("noethersolve.optimization_convergence", None),
    "nesterov_rate": ("noethersolve.optimization_convergence", None),
    "compare_algorithms": ("noethersolve.optimization_convergence", None),
    "analyze_conditioning": ("noethersolve.optimization_convergence", None),
    "oracle_lower_bound": ("noethersolve.optimization_convergence", None),
    "optimal_step_size": ("noethersolve.optimization_convergence", None),
    "non_convex_rate": ("noethersolve.optimization_convergence", None),
    "list_algorithms": ("noethersolve.optimization_convergence", None),
    "iterations_needed": ("noethersolve.optimization_convergence", None),
    "ConvergenceReport": ("noethersolve.optimization_convergence", None),
    "OptimizationComparisonReport": ("noethersolve.optimization_convergence", "ComparisonReport"),
    "ConditionReport": ("noethersolve.optimization_convergence", None),
    "LowerBoundReport": ("noethersolve.optimization_convergence", None),
    "StepSizeReport": ("noethersolve.optimization_convergence", None),
    "NonConvexReport": ("noethersolve.optimization_convergence", None),

    # numerical_pde.py
    "check_cfl": ("noethersolve.numerical_pde", None),
    "cfl_hyperbolic": ("noethersolve.numerical_pde", None),
    "cfl_parabolic": ("noethersolve.numerical_pde", None),
    "max_timestep": ("noethersolve.numerical_pde", None),
    "von_neumann_analysis": ("noethersolve.numerical_pde", None),
    "get_scheme_info": ("noethersolve.numerical_pde", None),
    "list_schemes": ("noethersolve.numerical_pde", None),
    "check_lax_equivalence": ("noethersolve.numerical_pde", None),
    "analyze_accuracy": ("noethersolve.numerical_pde", None),
    "check_common_error": ("noethersolve.numerical_pde", None),
    "CFLReport": ("noethersolve.numerical_pde", None),
    "VonNeumannReport": ("noethersolve.numerical_pde", None),
    "SchemeReport": ("noethersolve.numerical_pde", None),
    "LaxEquivalenceReport": ("noethersolve.numerical_pde", None),
    "AccuracyReport": ("noethersolve.numerical_pde", None),

    # mhd_conservation.py
    "check_magnetic_helicity": ("noethersolve.mhd_conservation", None),
    "check_cross_helicity": ("noethersolve.mhd_conservation", None),
    "check_mhd_energy": ("noethersolve.mhd_conservation", None),
    "check_frozen_flux": ("noethersolve.mhd_conservation", None),
    "check_div_B": ("noethersolve.mhd_conservation", None),
    "check_mhd_invariant": ("noethersolve.mhd_conservation", None),
    "list_mhd_invariants": ("noethersolve.mhd_conservation", None),
    "HelicityReport": ("noethersolve.mhd_conservation", None),
    "MHDEnergyReport": ("noethersolve.mhd_conservation", None),
    "FrozenFluxReport": ("noethersolve.mhd_conservation", None),
    "DivBReport": ("noethersolve.mhd_conservation", None),
    "MHDInvariantReport": ("noethersolve.mhd_conservation", "InvariantReport"),

    # gr_constraints.py
    "check_hamiltonian_constraint": ("noethersolve.gr_constraints", None),
    "check_momentum_constraint": ("noethersolve.gr_constraints", None),
    "check_adm_mass": ("noethersolve.gr_constraints", None),
    "check_bondi_mass": ("noethersolve.gr_constraints", None),
    "check_komar_mass": ("noethersolve.gr_constraints", None),
    "compare_mass_definitions": ("noethersolve.gr_constraints", None),
    "analyze_adm_formalism": ("noethersolve.gr_constraints", None),
    "list_gr_concepts": ("noethersolve.gr_constraints", None),
    "ConstraintReport": ("noethersolve.gr_constraints", None),
    "MassReport": ("noethersolve.gr_constraints", None),
    "MassComparisonReport": ("noethersolve.gr_constraints", None),
    "ADMReport": ("noethersolve.gr_constraints", None),

    # seismic_waves.py
    "calc_seismic_velocity": ("noethersolve.seismic_waves", None),
    "calc_velocity_from_poisson": ("noethersolve.seismic_waves", None),
    "poisson_from_velocities": ("noethersolve.seismic_waves", None),
    "convert_elastic_moduli": ("noethersolve.seismic_waves", None),
    "calc_reflection_coefficient": ("noethersolve.seismic_waves", None),
    "critical_angle": ("noethersolve.seismic_waves", None),
    "snells_law": ("noethersolve.seismic_waves", None),
    "vp_vs_ratio_bounds": ("noethersolve.seismic_waves", None),
    "SeismicVelocityReport": ("noethersolve.seismic_waves", None),
    "PoissonRatioReport": ("noethersolve.seismic_waves", None),
    "ElasticModuliReport": ("noethersolve.seismic_waves", None),
    "ReflectionReport": ("noethersolve.seismic_waves", None),

    # plasma_adiabatic.py
    "calc_magnetic_moment": ("noethersolve.plasma_adiabatic", None),
    "calc_bounce_invariant": ("noethersolve.plasma_adiabatic", None),
    "calc_flux_invariant": ("noethersolve.plasma_adiabatic", None),
    "check_adiabatic_hierarchy": ("noethersolve.plasma_adiabatic", None),
    "mirror_force": ("noethersolve.plasma_adiabatic", None),
    "loss_cone_angle": ("noethersolve.plasma_adiabatic", None),
    "cyclotron_frequency": ("noethersolve.plasma_adiabatic", None),
    "larmor_radius": ("noethersolve.plasma_adiabatic", None),
    "get_particle_mass": ("noethersolve.plasma_adiabatic", None),
    "MagneticMomentReport": ("noethersolve.plasma_adiabatic", None),
    "BounceInvariantReport": ("noethersolve.plasma_adiabatic", None),
    "FluxInvariantReport": ("noethersolve.plasma_adiabatic", None),
    "AdiabaticHierarchyReport": ("noethersolve.plasma_adiabatic", None),
    "ELECTRON_MASS": ("noethersolve.plasma_adiabatic", None),
    "PROTON_MASS": ("noethersolve.plasma_adiabatic", None),
    "ELECTRON_CHARGE": ("noethersolve.plasma_adiabatic", None),

    # dimension_physics.py
    "check_dimension_dependence": ("noethersolve.dimension_physics", None),
    "get_formula": ("noethersolve.dimension_physics", None),
    "list_dimension_dependent_concepts": ("noethersolve.dimension_physics", None),
    "DimensionalFormula": ("noethersolve.dimension_physics", None),
    "DimensionCheckResult": ("noethersolve.dimension_physics", None),
    "DIMENSIONAL_PHYSICS": ("noethersolve.dimension_physics", None),

    # info_thermo.py
    "calc_landauer_bound": ("noethersolve.info_thermo", None),
    "calc_shannon_entropy": ("noethersolve.info_thermo", None),
    "calc_info_thermo_bridge": ("noethersolve.info_thermo", None),
    "calc_huffman_landauer_parallel": ("noethersolve.info_thermo", None),
    "LandauerReport": ("noethersolve.info_thermo", None),
    "ShannonEntropyReport": ("noethersolve.info_thermo", None),
    "InfoThermoBridgeReport": ("noethersolve.info_thermo", None),
    "HuffmanLandauerReport": ("noethersolve.info_thermo", None),
    "K_B": ("noethersolve.info_thermo", None),
    "LN_2": ("noethersolve.info_thermo", None),

    # battery_degradation.py
    "calc_calendar_aging": ("noethersolve.battery_degradation", None),
    "calc_cycle_aging": ("noethersolve.battery_degradation", None),
    "calc_combined_aging": ("noethersolve.battery_degradation", None),
    "compare_chemistries": ("noethersolve.battery_degradation", None),
    "CalendarAgingReport": ("noethersolve.battery_degradation", None),
    "CycleAgingReport": ("noethersolve.battery_degradation", None),
    "CombinedAgingReport": ("noethersolve.battery_degradation", None),
    "CHEMISTRY_PARAMS": ("noethersolve.battery_degradation", None),

    # catalysis.py
    "calc_bep_activation": ("noethersolve.catalysis", None),
    "calc_volcano_position": ("noethersolve.catalysis", None),
    "calc_d_band_center": ("noethersolve.catalysis", None),
    "get_scaling_relation": ("noethersolve.catalysis", None),
    "find_optimal_catalyst": ("noethersolve.catalysis", None),
    "BEPReport": ("noethersolve.catalysis", None),
    "VolcanoReport": ("noethersolve.catalysis", None),
    "DBandReport": ("noethersolve.catalysis", None),
    "ScalingRelationReport": ("noethersolve.catalysis", None),
    "BEP_PARAMS": ("noethersolve.catalysis", None),
    "VOLCANO_REACTIONS": ("noethersolve.catalysis", None),
    "D_BAND_CENTERS": ("noethersolve.catalysis", None),

    # neoantigen_pipeline.py
    "score_cleavage": ("noethersolve.neoantigen_pipeline", None),
    "score_tap": ("noethersolve.neoantigen_pipeline", None),
    "score_mhc_binding": ("noethersolve.neoantigen_pipeline", None),
    "score_tcr_recognition": ("noethersolve.neoantigen_pipeline", None),
    "evaluate_neoantigen": ("noethersolve.neoantigen_pipeline", None),
    "compare_neoantigens": ("noethersolve.neoantigen_pipeline", "compare_candidates"),
    "CleavageReport": ("noethersolve.neoantigen_pipeline", None),
    "TAPReport": ("noethersolve.neoantigen_pipeline", None),
    "MHCBindingReport": ("noethersolve.neoantigen_pipeline", None),
    "TCRReport": ("noethersolve.neoantigen_pipeline", None),
    "NeoantigenPipelineReport": ("noethersolve.neoantigen_pipeline", "PipelineReport"),
    "MHCClass": ("noethersolve.neoantigen_pipeline", None),

    # noether_symmetry.py
    "symmetry_to_conservation": ("noethersolve.noether_symmetry", None),
    "conservation_to_symmetry": ("noethersolve.noether_symmetry", None),
    "verify_noether_claim": ("noethersolve.noether_symmetry", None),
    "list_noether_pairs": ("noethersolve.noether_symmetry", "list_all_pairs"),
    "get_noether_pair": ("noethersolve.noether_symmetry", "get_pair"),
    "SymmetryNoetherReport": ("noethersolve.noether_symmetry", "NoetherReport"),
    "NoetherPair": ("noethersolve.noether_symmetry", None),
    "SymmetryType": ("noethersolve.noether_symmetry", None),
    "ConservationType": ("noethersolve.noether_symmetry", None),
    "NOETHER_PAIRS": ("noethersolve.noether_symmetry", None),

    # mrna_design.py
    "calculate_base_pair_energy": ("noethersolve.mrna_design", None),
    "calculate_duplex_stability": ("noethersolve.mrna_design", None),
    "analyze_immunogenicity": ("noethersolve.mrna_design", None),
    "calculate_cai": ("noethersolve.mrna_design", None),
    "optimize_codons": ("noethersolve.mrna_design", None),
    "analyze_mrna_design": ("noethersolve.mrna_design", None),
    "compare_modifications": ("noethersolve.mrna_design", None),
    "ThermodynamicReport": ("noethersolve.mrna_design", None),
    "CodonOptimizationReport": ("noethersolve.mrna_design", None),
    "ImmunogenicityReport": ("noethersolve.mrna_design", None),
    "mRNADesignReport": ("noethersolve.mrna_design", None),
    "ModificationType": ("noethersolve.mrna_design", None),

    # behavioral_finance.py
    "prospect_value_function": ("noethersolve.behavioral_finance", None),
    "probability_weight": ("noethersolve.behavioral_finance", None),
    "calculate_prospect_value": ("noethersolve.behavioral_finance", None),
    "calculate_expected_value": ("noethersolve.behavioral_finance", None),
    "analyze_prospect": ("noethersolve.behavioral_finance", None),
    "exponential_discount": ("noethersolve.behavioral_finance", None),
    "hyperbolic_discount": ("noethersolve.behavioral_finance", None),
    "analyze_temporal_discounting": ("noethersolve.behavioral_finance", None),
    "analyze_allais_paradox": ("noethersolve.behavioral_finance", None),
    "analyze_loss_aversion": ("noethersolve.behavioral_finance", None),
    "mental_accounting_violation": ("noethersolve.behavioral_finance", None),
    "framing_effect_demo": ("noethersolve.behavioral_finance", None),
    "herding_cascade_threshold": ("noethersolve.behavioral_finance", None),
    "ProspectTheoryReport": ("noethersolve.behavioral_finance", None),
    "TemporalDiscountReport": ("noethersolve.behavioral_finance", None),
    "AllaisParadoxReport": ("noethersolve.behavioral_finance", None),
    "LossAversionReport": ("noethersolve.behavioral_finance", None),
    "DecisionType": ("noethersolve.behavioral_finance", None),
    "LOSS_AVERSION_LAMBDA": ("noethersolve.behavioral_finance", None),
    "VALUE_CURVATURE_ALPHA": ("noethersolve.behavioral_finance", None),
    "PROB_WEIGHT_GAMMA_GAINS": ("noethersolve.behavioral_finance", None),

    # autonomy_analysis.py
    "AutonomyFramework": ("noethersolve.autonomy_analysis", None),
    "ComponentStatus": ("noethersolve.autonomy_analysis", None),
    "AutonomyComponent": ("noethersolve.autonomy_analysis", None),
    "ComponentAssessment": ("noethersolve.autonomy_analysis", None),
    "AutonomyReport": ("noethersolve.autonomy_analysis", None),
    "ImplementationApproach": ("noethersolve.autonomy_analysis", None),
    "AUTONOMY_COMPONENTS": ("noethersolve.autonomy_analysis", None),
    "SYSTEM_PROFILES": ("noethersolve.autonomy_analysis", None),
    "IMPLEMENTATION_APPROACHES": ("noethersolve.autonomy_analysis", None),
    "get_all_components": ("noethersolve.autonomy_analysis", None),
    "assess_system": ("noethersolve.autonomy_analysis", None),
    "assess_predefined_system": ("noethersolve.autonomy_analysis", None),
    "compare_systems": ("noethersolve.autonomy_analysis", None),
    "identify_autonomy_gaps": ("noethersolve.autonomy_analysis", None),
    "check_autonomy_requirements": ("noethersolve.autonomy_analysis", None),
    "analyze_transformer_autonomy": ("noethersolve.autonomy_analysis", None),
    "list_frameworks": ("noethersolve.autonomy_analysis", None),
    "list_predefined_systems": ("noethersolve.autonomy_analysis", None),
    "get_implementation_roadmap": ("noethersolve.autonomy_analysis", None),
    "design_autonomous_system": ("noethersolve.autonomy_analysis", None),
    "get_minimum_viable_autonomy": ("noethersolve.autonomy_analysis", None),
    "get_full_autonomy_blueprint": ("noethersolve.autonomy_analysis", None),

    # metacognition.py
    "MetacognitiveProcess": ("noethersolve.metacognition", None),
    "KnowledgeType": ("noethersolve.metacognition", None),
    "MonitoringJudgment": ("noethersolve.metacognition", None),
    "ConfidenceSample": ("noethersolve.metacognition", None),
    "CalibrationResult": ("noethersolve.metacognition", None),
    "ResolutionResult": ("noethersolve.metacognition", None),
    "MetaDPrimeResult": ("noethersolve.metacognition", None),
    "UnknownRecallResult": ("noethersolve.metacognition", None),
    "SelfCorrectionResult": ("noethersolve.metacognition", None),
    "MetacognitiveStateVector": ("noethersolve.metacognition", None),
    "MetacognitionReport": ("noethersolve.metacognition", None),
    "compute_calibration": ("noethersolve.metacognition", None),
    "compute_resolution": ("noethersolve.metacognition", None),
    "compute_meta_d_prime": ("noethersolve.metacognition", None),
    "analyze_unknown_recall": ("noethersolve.metacognition", None),
    "analyze_self_correction": ("noethersolve.metacognition", None),
    "assess_metacognition": ("noethersolve.metacognition", None),
    "get_llm_metacognition_baseline": ("noethersolve.metacognition", None),
    "list_metacognitive_capabilities": ("noethersolve.metacognition", None),
    "LLM_TYPICAL_PROFILE": ("noethersolve.metacognition", None),

    # loving_autonomy.py
    "LovingAssistant": ("noethersolve.loving_autonomy", None),
    "AssistantResponse": ("noethersolve.loving_autonomy", None),

    # loving_service.py
    "ServicePriority": ("noethersolve.loving_service", None),
    "LovingDecision": ("noethersolve.loving_service", None),
    "UserContext": ("noethersolve.loving_service", None),
    "LovingServiceController": ("noethersolve.loving_service", None),
    "should_verify_with_tool": ("noethersolve.loving_service", None),
    "compute_loving_response_priority": ("noethersolve.loving_service", None),
    "should_spend_scarce_resources": ("noethersolve.loving_service", None),
    "format_correction_lovingly": ("noethersolve.loving_service", None),
    "acknowledge_uncertainty": ("noethersolve.loving_service", None),
    "get_principle_checklist": ("noethersolve.loving_service", None),
    "integrate_with_autonomy_loop": ("noethersolve.loving_service", None),
    "PRINCIPLE_WEIGHTS": ("noethersolve.loving_service", None),

    # metacognitive_control.py
    "MetacognitiveAction": ("noethersolve.metacognitive_control", None),
    "MetacognitiveDecision": ("noethersolve.metacognitive_control", None),
    "TaskContext": ("noethersolve.metacognitive_control", None),
    "CalibrationHistory": ("noethersolve.metacognitive_control", None),
    "MetacognitiveEnergyBudget": ("noethersolve.metacognitive_control", None),
    "MetacognitiveController": ("noethersolve.metacognitive_control", None),
    "should_engage_metacognition": ("noethersolve.metacognitive_control", None),
    "compute_error_probability": ("noethersolve.metacognitive_control", None),
    "compute_metacognitive_ev": ("noethersolve.metacognitive_control", None),
    "compute_optimal_check_threshold": ("noethersolve.metacognitive_control", None),
    "get_suggested_tool": ("noethersolve.metacognitive_control", None),
    "get_tools_for_domain": ("noethersolve.metacognitive_control", None),
    "is_blind_spot_domain": ("noethersolve.metacognitive_control", None),
    "list_metacognitive_actions": ("noethersolve.metacognitive_control", None),
    "get_domain_tool_coverage": ("noethersolve.metacognitive_control", None),
    "ResourceType": ("noethersolve.metacognitive_control", None),
    "ResourceCost": ("noethersolve.metacognitive_control", None),
    "ResourceBudget": ("noethersolve.metacognitive_control", None),
    "ToolType": ("noethersolve.metacognitive_control", None),
    "ResourceAwareDecision": ("noethersolve.metacognitive_control", None),
    "ResourceAwareController": ("noethersolve.metacognitive_control", None),
    "prefer_local_tools": ("noethersolve.metacognitive_control", None),
    "compute_resource_aware_ev": ("noethersolve.metacognitive_control", None),
    "detect_mlx_available": ("noethersolve.metacognitive_control", None),
    "get_compute_backend": ("noethersolve.metacognitive_control", None),
    "BACKEND_EFFICIENCY": ("noethersolve.metacognitive_control", None),
    "ACTION_COSTS": ("noethersolve.metacognitive_control", None),
    "ACTION_RESOURCE_COSTS": ("noethersolve.metacognitive_control", None),
    "TOOL_DOMAINS": ("noethersolve.metacognitive_control", None),
    "TOOL_TYPES": ("noethersolve.metacognitive_control", None),
    "BLIND_SPOT_DOMAINS": ("noethersolve.metacognitive_control", None),

    # antibody_developability.py
    "analyze_charge": ("noethersolve.antibody_developability", None),
    "analyze_aggregation": ("noethersolve.antibody_developability", None),
    "analyze_polyreactivity": ("noethersolve.antibody_developability", None),
    "analyze_liabilities": ("noethersolve.antibody_developability", None),
    "assess_developability": ("noethersolve.antibody_developability", None),
    "calc_charge_at_ph": ("noethersolve.antibody_developability", None),
    "estimate_pI": ("noethersolve.antibody_developability", None),
    "find_glycosylation_sites": ("noethersolve.antibody_developability", None),
    "ChargeReport": ("noethersolve.antibody_developability", None),
    "AbAggregationReport": ("noethersolve.antibody_developability", "AggregationReport"),
    "PolyreactivityReport": ("noethersolve.antibody_developability", None),
    "LiabilityReport": ("noethersolve.antibody_developability", None),
    "DevelopabilityReport": ("noethersolve.antibody_developability", None),
    "AbRiskLevel": ("noethersolve.antibody_developability", "RiskLevel"),

    # gauge_equivalence.py
    "check_gauge_equivalence": ("noethersolve.gauge_equivalence", None),
    "explain_parallel": ("noethersolve.gauge_equivalence", None),
    "list_parallels": ("noethersolve.gauge_equivalence", None),
    "simple_unify": ("noethersolve.gauge_equivalence", None),
    "GaugeEquivalenceReport": ("noethersolve.gauge_equivalence", None),
    "UnificationResult": ("noethersolve.gauge_equivalence", None),
    "RedundantDOF": ("noethersolve.gauge_equivalence", None),
    "GaugeDomain": ("noethersolve.gauge_equivalence", "Domain"),
    "KNOWN_PARALLELS": ("noethersolve.gauge_equivalence", None),

    # tool_graph.py
    "calculator": ("noethersolve.tool_graph", None),
    "get_registry": ("noethersolve.tool_graph", None),
    "find_tool_chain": ("noethersolve.tool_graph", None),
    "execute_chain": ("noethersolve.tool_graph", None),
    "ToolRegistry": ("noethersolve.tool_graph", None),
    "CalculatorMeta": ("noethersolve.tool_graph", None),

    # meta_router.py
    "MetaRouter": ("noethersolve.meta_router", None),
    "MetaRouterConfig": ("noethersolve.meta_router", None),
    "OutcomeRecord": ("noethersolve.meta_router", None),
    "FactEmbedder": ("noethersolve.meta_router", None),

    # stage_discovery.py
    "StageDiscoverer": ("noethersolve.stage_discovery", None),
    "DiscoveryConfig": ("noethersolve.stage_discovery", None),
    "EvalResult": ("noethersolve.stage_discovery", None),
    "StageSequence": ("noethersolve.stage_discovery", None),

    # outcome_logger.py
    "OutcomeLogger": ("noethersolve.outcome_logger", None),
    "get_outcome_logger": ("noethersolve.outcome_logger", "get_logger"),
    "log_outcome": ("noethersolve.outcome_logger", None),
    "log_batch": ("noethersolve.outcome_logger", None),

    # hooks.py
    "SessionState": ("noethersolve.hooks", None),
    "pre_tool_use": ("noethersolve.hooks", None),
    "post_tool_use": ("noethersolve.hooks", None),
    "session_end": ("noethersolve.hooks", None),
    "get_session_stats": ("noethersolve.hooks", None),
    "get_usage_stats": ("noethersolve.hooks", None),
    "get_tool_registry": ("noethersolve.hooks", None),
    "update_tool_registry": ("noethersolve.hooks", None),
    "categorize_tools": ("noethersolve.hooks", None),
    "is_local_tool": ("noethersolve.hooks", None),
    "is_verification_tool": ("noethersolve.hooks", None),
    "get_tool_domain": ("noethersolve.hooks", None),
}

# Cache for loaded modules and attributes
_loaded_attrs: dict[str, Any] = {}


def __getattr__(name: str) -> Any:
    """Lazy load attributes on first access."""
    # Check cache first
    if name in _loaded_attrs:
        return _loaded_attrs[name]

    # Check if this is a lazy import
    if name in _LAZY_IMPORTS:
        module_name, attr_name = _LAZY_IMPORTS[name]
        attr_name = attr_name or name

        try:
            module = importlib.import_module(module_name)
            attr = getattr(module, attr_name)
            _loaded_attrs[name] = attr
            return attr
        except ImportError as e:
            # Some modules are optional (e.g., reaction_engine needs RDKit)
            raise ImportError(
                f"Cannot import '{name}' from '{module_name}': {e}. "
                f"This may require optional dependencies."
            ) from e

    raise AttributeError(f"module 'noethersolve' has no attribute '{name}'")


def __dir__() -> list[str]:
    """List all available attributes including lazy ones."""
    return list(_LAZY_IMPORTS.keys()) + ["__version__"]


# For explicit imports like `from noethersolve import *`
__all__ = list(_LAZY_IMPORTS.keys()) + ["__version__"]
