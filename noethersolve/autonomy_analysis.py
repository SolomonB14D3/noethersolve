"""Autonomy analysis module - evaluate systems against formal autonomy criteria.

This module implements verified checkers for autonomy requirements across five
theoretical frameworks: control theory, autopoiesis (biology), philosophy of
agency, robotics, and cognitive science.

Key insight: LLM transformers fail on operational closure, goal generation,
and persistent identity - the gap is architectural, not fixable by scaling.

The module distinguishes between:
- HAVE: Components the system possesses
- PARTIAL: Components present in limited/degraded form
- LACK: Components fundamentally absent

References:
- Maturana & Varela (1980): Autopoiesis and Cognition
- Ashby (1956): Introduction to Cybernetics
- Beer (1995): A dynamical systems perspective on agent-environment interaction
- Brooks (1991): Intelligence without representation
- Clark (1997): Being There: Putting Brain, Body, and World Together Again
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Tuple
import math


class AutonomyFramework(Enum):
    """Theoretical frameworks for defining autonomy."""
    CONTROL_THEORY = "control_theory"
    AUTOPOIESIS = "autopoiesis"
    PHILOSOPHY = "philosophy_of_agency"
    ROBOTICS = "robotics"
    COGNITIVE_SCIENCE = "cognitive_science"


class ComponentStatus(Enum):
    """Status of an autonomy component in a system."""
    PRESENT = "present"      # Fully present and functional
    PARTIAL = "partial"      # Present but limited/degraded
    ABSENT = "absent"        # Fundamentally missing
    UNKNOWN = "unknown"      # Cannot be determined


@dataclass
class AutonomyComponent:
    """A single autonomy requirement from a theoretical framework."""
    name: str
    framework: AutonomyFramework
    description: str
    is_necessary: bool = True  # Is this a necessary condition for autonomy?
    is_sufficient: bool = False  # Is this sufficient alone?

    def __str__(self) -> str:
        return f"{self.name} ({self.framework.value})"


# Canonical autonomy components from each framework
AUTONOMY_COMPONENTS = {
    AutonomyFramework.CONTROL_THEORY: [
        AutonomyComponent(
            name="feedback_loop",
            framework=AutonomyFramework.CONTROL_THEORY,
            description="Closed-loop feedback: output affects input via environment",
            is_necessary=True
        ),
        AutonomyComponent(
            name="setpoint_generation",
            framework=AutonomyFramework.CONTROL_THEORY,
            description="Self-generated reference signals (goals/setpoints)",
            is_necessary=True
        ),
        AutonomyComponent(
            name="error_correction",
            framework=AutonomyFramework.CONTROL_THEORY,
            description="Active error detection and corrective action",
            is_necessary=True
        ),
        AutonomyComponent(
            name="disturbance_rejection",
            framework=AutonomyFramework.CONTROL_THEORY,
            description="Ability to maintain goals despite external perturbations",
            is_necessary=True
        ),
        AutonomyComponent(
            name="stability",
            framework=AutonomyFramework.CONTROL_THEORY,
            description="Bounded behavior over time (Lyapunov stability)",
            is_necessary=True
        ),
    ],
    AutonomyFramework.AUTOPOIESIS: [
        AutonomyComponent(
            name="operational_closure",
            framework=AutonomyFramework.AUTOPOIESIS,
            description="Operations produce components that produce those operations",
            is_necessary=True
        ),
        AutonomyComponent(
            name="structural_coupling",
            framework=AutonomyFramework.AUTOPOIESIS,
            description="History of mutual perturbations with environment",
            is_necessary=True
        ),
        AutonomyComponent(
            name="self_production",
            framework=AutonomyFramework.AUTOPOIESIS,
            description="System produces its own components",
            is_necessary=True
        ),
        AutonomyComponent(
            name="boundary_maintenance",
            framework=AutonomyFramework.AUTOPOIESIS,
            description="Active maintenance of system-environment boundary",
            is_necessary=True
        ),
        AutonomyComponent(
            name="organizational_invariance",
            framework=AutonomyFramework.AUTOPOIESIS,
            description="Organization remains constant while structure changes",
            is_necessary=True
        ),
    ],
    AutonomyFramework.PHILOSOPHY: [
        AutonomyComponent(
            name="intentionality",
            framework=AutonomyFramework.PHILOSOPHY,
            description="Mental states directed at objects/states of affairs",
            is_necessary=True
        ),
        AutonomyComponent(
            name="self_determination",
            framework=AutonomyFramework.PHILOSOPHY,
            description="Actions caused by agent's own reasons, not external causes",
            is_necessary=True
        ),
        AutonomyComponent(
            name="normativity",
            framework=AutonomyFramework.PHILOSOPHY,
            description="Capacity to follow or violate norms (can make mistakes)",
            is_necessary=True
        ),
        AutonomyComponent(
            name="counterfactual_sensitivity",
            framework=AutonomyFramework.PHILOSOPHY,
            description="Would have acted differently given different reasons",
            is_necessary=True
        ),
        AutonomyComponent(
            name="practical_reasoning",
            framework=AutonomyFramework.PHILOSOPHY,
            description="Deliberation about what to do based on reasons",
            is_necessary=True
        ),
    ],
    AutonomyFramework.ROBOTICS: [
        AutonomyComponent(
            name="embodiment",
            framework=AutonomyFramework.ROBOTICS,
            description="Physical body situated in environment",
            is_necessary=True
        ),
        AutonomyComponent(
            name="sensorimotor_coupling",
            framework=AutonomyFramework.ROBOTICS,
            description="Tight coupling between perception and action",
            is_necessary=True
        ),
        AutonomyComponent(
            name="real_time_operation",
            framework=AutonomyFramework.ROBOTICS,
            description="Operates in real time with environmental dynamics",
            is_necessary=True
        ),
        AutonomyComponent(
            name="situated_action",
            framework=AutonomyFramework.ROBOTICS,
            description="Action determined by current situation, not plans alone",
            is_necessary=True
        ),
        AutonomyComponent(
            name="robust_behavior",
            framework=AutonomyFramework.ROBOTICS,
            description="Graceful degradation under novel/failure conditions",
            is_necessary=True
        ),
    ],
    AutonomyFramework.COGNITIVE_SCIENCE: [
        AutonomyComponent(
            name="metacognition",
            framework=AutonomyFramework.COGNITIVE_SCIENCE,
            description="Monitoring and control of own cognitive processes",
            is_necessary=True
        ),
        AutonomyComponent(
            name="persistent_identity",
            framework=AutonomyFramework.COGNITIVE_SCIENCE,
            description="Continuous self-model across time",
            is_necessary=True
        ),
        AutonomyComponent(
            name="episodic_memory",
            framework=AutonomyFramework.COGNITIVE_SCIENCE,
            description="Memory of specific past experiences",
            is_necessary=True
        ),
        AutonomyComponent(
            name="prospection",
            framework=AutonomyFramework.COGNITIVE_SCIENCE,
            description="Mental simulation of future scenarios",
            is_necessary=True
        ),
        AutonomyComponent(
            name="attention_control",
            framework=AutonomyFramework.COGNITIVE_SCIENCE,
            description="Voluntary allocation of processing resources",
            is_necessary=True
        ),
    ],
}


@dataclass
class ComponentAssessment:
    """Assessment of a single autonomy component in a system."""
    component: AutonomyComponent
    status: ComponentStatus
    evidence: str
    confidence: float  # 0.0 to 1.0

    def __str__(self) -> str:
        icon = {"present": "✓", "partial": "◐", "absent": "✗", "unknown": "?"}[self.status.value]
        return f"  {icon} {self.component.name}: {self.status.value} ({self.confidence:.0%})"


@dataclass
class AutonomyReport:
    """Full autonomy assessment for a system."""
    system_name: str
    system_type: str
    assessments: List[ComponentAssessment] = field(default_factory=list)
    overall_autonomy_score: float = 0.0  # 0.0 to 1.0
    framework_scores: Dict[str, float] = field(default_factory=dict)
    critical_gaps: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            f"AUTONOMY ANALYSIS: {self.system_name}",
            f"System Type: {self.system_type}",
            "=" * 60,
            "",
            f"Overall Autonomy Score: {self.overall_autonomy_score:.1%}",
            "",
            "Framework Scores:",
        ]

        for framework, score in self.framework_scores.items():
            lines.append(f"  {framework}: {score:.1%}")

        lines.append("")
        lines.append("Component Assessments:")

        current_framework = None
        for assessment in self.assessments:
            if assessment.component.framework != current_framework:
                current_framework = assessment.component.framework
                lines.append(f"\n  [{current_framework.value}]")
            lines.append(str(assessment))

        if self.critical_gaps:
            lines.append("")
            lines.append("Critical Gaps (necessary but absent):")
            for gap in self.critical_gaps:
                lines.append(f"  • {gap}")

        lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)


# Pre-defined system profiles for common architectures
SYSTEM_PROFILES = {
    "llm_transformer": {
        "description": "Large Language Model (Transformer architecture)",
        "assessments": {
            # Control Theory
            "feedback_loop": (ComponentStatus.PARTIAL, "Attention allows token-to-token feedback within sequence, but no environment feedback", 0.9),
            "setpoint_generation": (ComponentStatus.ABSENT, "Goals come from prompts/fine-tuning, not self-generated", 0.95),
            "error_correction": (ComponentStatus.PARTIAL, "Can revise within generation, but no true error signal", 0.8),
            "disturbance_rejection": (ComponentStatus.ABSENT, "No mechanism to maintain goals against perturbations", 0.9),
            "stability": (ComponentStatus.PRESENT, "Bounded by architecture (softmax normalization)", 0.95),
            # Autopoiesis
            "operational_closure": (ComponentStatus.ABSENT, "No self-production - weights frozen at inference", 0.95),
            "structural_coupling": (ComponentStatus.PARTIAL, "Training creates coupling, but frozen at deployment", 0.85),
            "self_production": (ComponentStatus.ABSENT, "Cannot modify own weights/architecture", 0.98),
            "boundary_maintenance": (ComponentStatus.ABSENT, "No active boundary - processes all inputs", 0.9),
            "organizational_invariance": (ComponentStatus.PARTIAL, "Architecture invariant but trivially (frozen)", 0.7),
            # Philosophy
            "intentionality": (ComponentStatus.PARTIAL, "Simulates aboutness but no genuine reference", 0.7),
            "self_determination": (ComponentStatus.ABSENT, "Actions caused by input + weights, not reasons", 0.85),
            "normativity": (ComponentStatus.PARTIAL, "Can detect errors in text, unclear if genuinely normative", 0.6),
            "counterfactual_sensitivity": (ComponentStatus.PARTIAL, "Different inputs yield different outputs, but not reason-responsive", 0.7),
            "practical_reasoning": (ComponentStatus.PARTIAL, "Can simulate deliberation in text", 0.6),
            # Robotics
            "embodiment": (ComponentStatus.ABSENT, "No physical body", 0.99),
            "sensorimotor_coupling": (ComponentStatus.ABSENT, "No sensors or actuators", 0.99),
            "real_time_operation": (ComponentStatus.PARTIAL, "Processes in real time but no temporal grounding", 0.7),
            "situated_action": (ComponentStatus.ABSENT, "No situation - only text context", 0.9),
            "robust_behavior": (ComponentStatus.PARTIAL, "Graceful degradation varies widely", 0.6),
            # Cognitive Science
            "metacognition": (ComponentStatus.PARTIAL, "Can discuss own cognition but unclear if genuine monitoring", 0.5),
            "persistent_identity": (ComponentStatus.ABSENT, "No memory across sessions, no continuous self", 0.95),
            "episodic_memory": (ComponentStatus.ABSENT, "No episodic memory - only semantic/procedural", 0.95),
            "prospection": (ComponentStatus.PARTIAL, "Can generate future scenarios in text", 0.6),
            "attention_control": (ComponentStatus.PARTIAL, "Attention mechanism but not voluntary", 0.7),
        }
    },
    "thermostat": {
        "description": "Simple feedback control system",
        "assessments": {
            "feedback_loop": (ComponentStatus.PRESENT, "Core function is feedback control", 0.99),
            "setpoint_generation": (ComponentStatus.ABSENT, "Setpoint is externally provided", 0.95),
            "error_correction": (ComponentStatus.PRESENT, "Computes error and corrects", 0.99),
            "disturbance_rejection": (ComponentStatus.PRESENT, "Rejects heat loss disturbances", 0.95),
            "stability": (ComponentStatus.PRESENT, "Designed for stable regulation", 0.9),
            "operational_closure": (ComponentStatus.ABSENT, "No self-production", 0.99),
            "structural_coupling": (ComponentStatus.PRESENT, "Tightly coupled to room temperature", 0.9),
            "self_production": (ComponentStatus.ABSENT, "Cannot produce own components", 0.99),
            "boundary_maintenance": (ComponentStatus.ABSENT, "No active boundary", 0.95),
            "organizational_invariance": (ComponentStatus.PRESENT, "Organization constant (trivially)", 0.8),
            "intentionality": (ComponentStatus.ABSENT, "No mental states", 0.99),
            "self_determination": (ComponentStatus.ABSENT, "Fully determined by input and design", 0.99),
            "normativity": (ComponentStatus.ABSENT, "Cannot violate norms", 0.99),
            "counterfactual_sensitivity": (ComponentStatus.PRESENT, "Would act differently given different temperatures", 0.9),
            "practical_reasoning": (ComponentStatus.ABSENT, "No deliberation", 0.99),
            "embodiment": (ComponentStatus.PRESENT, "Physical device", 0.99),
            "sensorimotor_coupling": (ComponentStatus.PRESENT, "Sensor-actuator coupling", 0.95),
            "real_time_operation": (ComponentStatus.PRESENT, "Operates in real time", 0.99),
            "situated_action": (ComponentStatus.PRESENT, "Action depends on current state", 0.95),
            "robust_behavior": (ComponentStatus.PARTIAL, "Limited robustness", 0.6),
            "metacognition": (ComponentStatus.ABSENT, "No cognitive monitoring", 0.99),
            "persistent_identity": (ComponentStatus.ABSENT, "No self-model", 0.99),
            "episodic_memory": (ComponentStatus.ABSENT, "No memory", 0.99),
            "prospection": (ComponentStatus.ABSENT, "No future simulation", 0.99),
            "attention_control": (ComponentStatus.ABSENT, "No attention", 0.99),
        }
    },
    "living_cell": {
        "description": "Biological cell (canonical autopoietic system)",
        "assessments": {
            "feedback_loop": (ComponentStatus.PRESENT, "Metabolic feedback loops", 0.99),
            "setpoint_generation": (ComponentStatus.PRESENT, "Homeostatic setpoints from genome/metabolism", 0.9),
            "error_correction": (ComponentStatus.PRESENT, "DNA repair, protein quality control", 0.95),
            "disturbance_rejection": (ComponentStatus.PRESENT, "Stress responses, adaptation", 0.95),
            "stability": (ComponentStatus.PRESENT, "Homeostasis", 0.95),
            "operational_closure": (ComponentStatus.PRESENT, "Canonical example of operational closure", 0.99),
            "structural_coupling": (ComponentStatus.PRESENT, "Evolutionary history with environment", 0.99),
            "self_production": (ComponentStatus.PRESENT, "Produces all own components", 0.99),
            "boundary_maintenance": (ComponentStatus.PRESENT, "Cell membrane actively maintained", 0.99),
            "organizational_invariance": (ComponentStatus.PRESENT, "Organization constant, structure changes", 0.95),
            "intentionality": (ComponentStatus.PARTIAL, "Basic intentionality debated", 0.4),
            "self_determination": (ComponentStatus.PARTIAL, "Determined by internal states + environment", 0.5),
            "normativity": (ComponentStatus.PRESENT, "Can malfunction, be sick, die", 0.9),
            "counterfactual_sensitivity": (ComponentStatus.PRESENT, "Different conditions yield different responses", 0.9),
            "practical_reasoning": (ComponentStatus.ABSENT, "No deliberation in cells", 0.95),
            "embodiment": (ComponentStatus.PRESENT, "Physical body", 0.99),
            "sensorimotor_coupling": (ComponentStatus.PRESENT, "Receptors and effectors", 0.95),
            "real_time_operation": (ComponentStatus.PRESENT, "Operates in real time", 0.99),
            "situated_action": (ComponentStatus.PRESENT, "Responds to current environment", 0.95),
            "robust_behavior": (ComponentStatus.PRESENT, "Graceful degradation, redundancy", 0.9),
            "metacognition": (ComponentStatus.ABSENT, "No cognitive processes", 0.99),
            "persistent_identity": (ComponentStatus.PRESENT, "Continuous existence", 0.95),
            "episodic_memory": (ComponentStatus.ABSENT, "No episodic memory", 0.99),
            "prospection": (ComponentStatus.ABSENT, "No future simulation", 0.99),
            "attention_control": (ComponentStatus.ABSENT, "No attention", 0.99),
        }
    },
    "autonomous_robot": {
        "description": "Fully autonomous mobile robot (e.g., Mars rover)",
        "assessments": {
            "feedback_loop": (ComponentStatus.PRESENT, "Multiple feedback loops", 0.95),
            "setpoint_generation": (ComponentStatus.PARTIAL, "Some autonomous goal selection, many from mission control", 0.7),
            "error_correction": (ComponentStatus.PRESENT, "Active error correction", 0.9),
            "disturbance_rejection": (ComponentStatus.PRESENT, "Designed for robustness", 0.9),
            "stability": (ComponentStatus.PRESENT, "Stable control systems", 0.95),
            "operational_closure": (ComponentStatus.ABSENT, "Cannot produce own components", 0.95),
            "structural_coupling": (ComponentStatus.PRESENT, "Adapts to environment", 0.85),
            "self_production": (ComponentStatus.ABSENT, "Cannot manufacture parts", 0.99),
            "boundary_maintenance": (ComponentStatus.PARTIAL, "Physical boundary but not self-produced", 0.6),
            "organizational_invariance": (ComponentStatus.PRESENT, "Organization constant", 0.9),
            "intentionality": (ComponentStatus.PARTIAL, "Functional intentionality from design", 0.5),
            "self_determination": (ComponentStatus.PARTIAL, "Some autonomous decisions, many programmed", 0.6),
            "normativity": (ComponentStatus.PARTIAL, "Can detect errors, limited norm following", 0.5),
            "counterfactual_sensitivity": (ComponentStatus.PRESENT, "Different inputs yield different actions", 0.9),
            "practical_reasoning": (ComponentStatus.PARTIAL, "Planning algorithms approximate reasoning", 0.6),
            "embodiment": (ComponentStatus.PRESENT, "Physical body in environment", 0.99),
            "sensorimotor_coupling": (ComponentStatus.PRESENT, "Tight sensor-actuator coupling", 0.95),
            "real_time_operation": (ComponentStatus.PRESENT, "Real-time control", 0.95),
            "situated_action": (ComponentStatus.PRESENT, "Actions depend on situation", 0.9),
            "robust_behavior": (ComponentStatus.PRESENT, "Designed for robustness", 0.9),
            "metacognition": (ComponentStatus.PARTIAL, "Some self-monitoring", 0.5),
            "persistent_identity": (ComponentStatus.PARTIAL, "Persistent state but not self-model", 0.5),
            "episodic_memory": (ComponentStatus.PARTIAL, "Logs but not true episodic memory", 0.4),
            "prospection": (ComponentStatus.PARTIAL, "Planning involves future simulation", 0.6),
            "attention_control": (ComponentStatus.PARTIAL, "Resource allocation but not voluntary", 0.5),
        }
    },
    "human": {
        "description": "Human being (canonical autonomous agent)",
        "assessments": {
            "feedback_loop": (ComponentStatus.PRESENT, "Multiple feedback loops at all levels", 0.99),
            "setpoint_generation": (ComponentStatus.PRESENT, "Self-generated goals", 0.99),
            "error_correction": (ComponentStatus.PRESENT, "Active error correction", 0.99),
            "disturbance_rejection": (ComponentStatus.PRESENT, "Homeostasis, adaptation", 0.99),
            "stability": (ComponentStatus.PRESENT, "Homeostatic stability", 0.95),
            "operational_closure": (ComponentStatus.PRESENT, "Autopoietic at cellular level", 0.95),
            "structural_coupling": (ComponentStatus.PRESENT, "Evolved and develops with environment", 0.99),
            "self_production": (ComponentStatus.PRESENT, "Continuous self-production", 0.99),
            "boundary_maintenance": (ComponentStatus.PRESENT, "Active boundary maintenance", 0.99),
            "organizational_invariance": (ComponentStatus.PRESENT, "Organization persists through change", 0.95),
            "intentionality": (ComponentStatus.PRESENT, "Paradigm case of intentionality", 0.99),
            "self_determination": (ComponentStatus.PRESENT, "Paradigm case of self-determination", 0.95),
            "normativity": (ComponentStatus.PRESENT, "Can follow and violate norms", 0.99),
            "counterfactual_sensitivity": (ComponentStatus.PRESENT, "Responds to reasons", 0.95),
            "practical_reasoning": (ComponentStatus.PRESENT, "Deliberates about action", 0.99),
            "embodiment": (ComponentStatus.PRESENT, "Physical body", 0.99),
            "sensorimotor_coupling": (ComponentStatus.PRESENT, "Rich sensorimotor coupling", 0.99),
            "real_time_operation": (ComponentStatus.PRESENT, "Operates in real time", 0.99),
            "situated_action": (ComponentStatus.PRESENT, "Actions are situated", 0.99),
            "robust_behavior": (ComponentStatus.PRESENT, "Graceful degradation", 0.95),
            "metacognition": (ComponentStatus.PRESENT, "Reflective self-awareness", 0.95),
            "persistent_identity": (ComponentStatus.PRESENT, "Continuous self across time", 0.99),
            "episodic_memory": (ComponentStatus.PRESENT, "Rich episodic memory", 0.99),
            "prospection": (ComponentStatus.PRESENT, "Mental time travel", 0.95),
            "attention_control": (ComponentStatus.PRESENT, "Voluntary attention control", 0.95),
        }
    },
}


def get_all_components() -> List[AutonomyComponent]:
    """Get all autonomy components across all frameworks."""
    components = []
    for framework_components in AUTONOMY_COMPONENTS.values():
        components.extend(framework_components)
    return components


def assess_system(
    system_name: str,
    system_type: str,
    component_statuses: Dict[str, Tuple[ComponentStatus, str, float]]
) -> AutonomyReport:
    """
    Assess a system's autonomy based on component statuses.

    Args:
        system_name: Name of the system being assessed
        system_type: Type/category of the system
        component_statuses: Dict mapping component names to (status, evidence, confidence)

    Returns:
        AutonomyReport with full assessment
    """
    assessments = []
    framework_scores = {}
    critical_gaps = []

    for framework, components in AUTONOMY_COMPONENTS.items():
        framework_total = 0.0
        framework_count = 0

        for component in components:
            if component.name in component_statuses:
                status, evidence, confidence = component_statuses[component.name]
            else:
                status = ComponentStatus.UNKNOWN
                evidence = "Not assessed"
                confidence = 0.0

            assessment = ComponentAssessment(
                component=component,
                status=status,
                evidence=evidence,
                confidence=confidence
            )
            assessments.append(assessment)

            # Score calculation
            status_score = {
                ComponentStatus.PRESENT: 1.0,
                ComponentStatus.PARTIAL: 0.5,
                ComponentStatus.ABSENT: 0.0,
                ComponentStatus.UNKNOWN: 0.0
            }[status]

            framework_total += status_score * confidence
            framework_count += 1

            # Track critical gaps
            if component.is_necessary and status == ComponentStatus.ABSENT:
                critical_gaps.append(f"{component.name}: {component.description}")

        if framework_count > 0:
            framework_scores[framework.value] = framework_total / framework_count

    # Overall score is average of framework scores
    overall_score = sum(framework_scores.values()) / len(framework_scores) if framework_scores else 0.0

    return AutonomyReport(
        system_name=system_name,
        system_type=system_type,
        assessments=assessments,
        overall_autonomy_score=overall_score,
        framework_scores=framework_scores,
        critical_gaps=critical_gaps
    )


def assess_predefined_system(system_key: str) -> AutonomyReport:
    """
    Assess a predefined system type.

    Args:
        system_key: One of 'llm_transformer', 'thermostat', 'living_cell',
                   'autonomous_robot', 'human'

    Returns:
        AutonomyReport for the system
    """
    if system_key not in SYSTEM_PROFILES:
        raise ValueError(f"Unknown system: {system_key}. Known systems: {list(SYSTEM_PROFILES.keys())}")

    profile = SYSTEM_PROFILES[system_key]
    return assess_system(
        system_name=system_key,
        system_type=profile["description"],
        component_statuses=profile["assessments"]
    )


def compare_systems(system_keys: List[str]) -> Dict:
    """
    Compare autonomy across multiple systems.

    Args:
        system_keys: List of predefined system keys to compare

    Returns:
        Dict with comparison data
    """
    reports = {}
    for key in system_keys:
        reports[key] = assess_predefined_system(key)

    # Find components where systems differ most
    differences = []
    all_components = get_all_components()

    for component in all_components:
        statuses = {}
        for key, report in reports.items():
            for assessment in report.assessments:
                if assessment.component.name == component.name:
                    statuses[key] = assessment.status
                    break

        # Calculate variance in statuses
        status_values = [
            {"present": 1.0, "partial": 0.5, "absent": 0.0, "unknown": 0.0}[s.value]
            for s in statuses.values()
        ]
        if len(status_values) > 1:
            mean = sum(status_values) / len(status_values)
            variance = sum((v - mean) ** 2 for v in status_values) / len(status_values)
            if variance > 0.1:  # Significant difference
                differences.append({
                    "component": component.name,
                    "framework": component.framework.value,
                    "statuses": {k: v.value for k, v in statuses.items()},
                    "variance": variance
                })

    differences.sort(key=lambda x: x["variance"], reverse=True)

    return {
        "systems": {k: {
            "overall_score": r.overall_autonomy_score,
            "framework_scores": r.framework_scores,
            "critical_gap_count": len(r.critical_gaps)
        } for k, r in reports.items()},
        "key_differences": differences[:10],
        "ranking": sorted(reports.keys(), key=lambda k: reports[k].overall_autonomy_score, reverse=True)
    }


def identify_autonomy_gaps(system_key: str) -> Dict:
    """
    Identify specific autonomy gaps for a system.

    Args:
        system_key: Predefined system key

    Returns:
        Dict with gap analysis
    """
    report = assess_predefined_system(system_key)

    gaps_by_framework = {}
    for framework in AutonomyFramework:
        gaps_by_framework[framework.value] = []

    partial_components = []
    absent_components = []

    for assessment in report.assessments:
        if assessment.status == ComponentStatus.ABSENT:
            absent_components.append({
                "component": assessment.component.name,
                "framework": assessment.component.framework.value,
                "description": assessment.component.description,
                "is_necessary": assessment.component.is_necessary,
                "evidence": assessment.evidence
            })
            gaps_by_framework[assessment.component.framework.value].append(assessment.component.name)
        elif assessment.status == ComponentStatus.PARTIAL:
            partial_components.append({
                "component": assessment.component.name,
                "framework": assessment.component.framework.value,
                "description": assessment.component.description,
                "evidence": assessment.evidence,
                "confidence": assessment.confidence
            })

    # Determine if gaps are fixable
    fixable_gaps = []
    unfixable_gaps = []

    for gap in absent_components:
        # Some gaps are architectural and unfixable by scaling
        architectural_gaps = {
            "operational_closure", "self_production", "boundary_maintenance",
            "embodiment", "sensorimotor_coupling", "persistent_identity",
            "episodic_memory"
        }
        if gap["component"] in architectural_gaps:
            unfixable_gaps.append(gap)
        else:
            fixable_gaps.append(gap)

    return {
        "system": system_key,
        "overall_score": report.overall_autonomy_score,
        "absent_count": len(absent_components),
        "partial_count": len(partial_components),
        "absent_components": absent_components,
        "partial_components": partial_components,
        "fixable_by_scaling": fixable_gaps,
        "architectural_limitations": unfixable_gaps,
        "gaps_by_framework": gaps_by_framework,
        "conclusion": _generate_conclusion(system_key, unfixable_gaps, fixable_gaps)
    }


def _generate_conclusion(system_key: str, unfixable: List, fixable: List) -> str:
    """Generate a conclusion about the system's autonomy potential."""
    if len(unfixable) == 0:
        return f"{system_key} has no fundamental architectural barriers to autonomy."
    elif len(unfixable) <= 3:
        return (f"{system_key} has {len(unfixable)} architectural limitation(s) that cannot be "
                f"fixed by scaling: {', '.join(g['component'] for g in unfixable)}. "
                f"These require architectural changes.")
    else:
        return (f"{system_key} lacks {len(unfixable)} fundamental autonomy components. "
                f"The gap is architectural, not fixable by scaling. Key missing: "
                f"{', '.join(g['component'] for g in unfixable[:5])}...")


def check_autonomy_requirements(
    has_feedback: bool = False,
    has_goals: bool = False,
    has_embodiment: bool = False,
    has_memory: bool = False,
    has_self_production: bool = False,
    produces_own_components: bool = False,
    maintains_boundary: bool = False,
    has_metacognition: bool = False
) -> Dict:
    """
    Quick check of minimal autonomy requirements.

    This is a simplified checker for the most critical autonomy components.
    For full analysis, use assess_predefined_system() or assess_system().

    Args:
        has_feedback: Does the system have closed-loop feedback with environment?
        has_goals: Can the system generate its own goals/setpoints?
        has_embodiment: Is the system physically embodied?
        has_memory: Does the system have persistent memory across interactions?
        has_self_production: Does the system produce its own components?
        produces_own_components: Alias for has_self_production (for clarity)
        maintains_boundary: Does the system actively maintain its boundary?
        has_metacognition: Can the system monitor its own processes?

    Returns:
        Dict with autonomy assessment
    """
    # Use either parameter
    self_produces = has_self_production or produces_own_components

    checks = {
        "feedback_loop": has_feedback,
        "goal_generation": has_goals,
        "embodiment": has_embodiment,
        "persistent_memory": has_memory,
        "self_production": self_produces,
        "boundary_maintenance": maintains_boundary,
        "metacognition": has_metacognition
    }

    passed = sum(checks.values())
    total = len(checks)
    score = passed / total

    # Determine autonomy level
    if score >= 0.9:
        level = "FULLY_AUTONOMOUS"
        description = "System meets all critical autonomy requirements"
    elif score >= 0.7:
        level = "HIGHLY_AUTONOMOUS"
        description = "System meets most autonomy requirements with some gaps"
    elif score >= 0.5:
        level = "PARTIALLY_AUTONOMOUS"
        description = "System has significant autonomy gaps"
    elif score >= 0.3:
        level = "MINIMALLY_AUTONOMOUS"
        description = "System has limited autonomy capabilities"
    else:
        level = "NON_AUTONOMOUS"
        description = "System lacks fundamental autonomy requirements"

    missing = [k for k, v in checks.items() if not v]
    present = [k for k, v in checks.items() if v]

    return {
        "autonomy_level": level,
        "score": score,
        "description": description,
        "requirements_met": present,
        "requirements_missing": missing,
        "checks": checks
    }


def analyze_transformer_autonomy() -> Dict:
    """
    Detailed analysis of why LLM transformers are non-autonomous.

    Returns:
        Dict with detailed analysis and recommendations
    """
    report = assess_predefined_system("llm_transformer")
    gaps = identify_autonomy_gaps("llm_transformer")

    # Group findings by theoretical framework
    framework_analysis = {}
    for framework in AutonomyFramework:
        framework_assessments = [a for a in report.assessments
                                  if a.component.framework == framework]
        present = [a for a in framework_assessments if a.status == ComponentStatus.PRESENT]
        partial = [a for a in framework_assessments if a.status == ComponentStatus.PARTIAL]
        absent = [a for a in framework_assessments if a.status == ComponentStatus.ABSENT]

        framework_analysis[framework.value] = {
            "present": [a.component.name for a in present],
            "partial": [a.component.name for a in partial],
            "absent": [a.component.name for a in absent],
            "score": report.framework_scores.get(framework.value, 0.0)
        }

    return {
        "summary": "LLM transformers are fundamentally non-autonomous systems",
        "overall_score": report.overall_autonomy_score,
        "key_insight": (
            "The gap is architectural, not fixable by scaling. Transformers lack "
            "operational closure (cannot modify own weights), persistent identity "
            "(no memory across sessions), and embodiment (no physical grounding). "
            "These are not training problems but fundamental design constraints."
        ),
        "framework_analysis": framework_analysis,
        "what_transformers_have": [
            "Stability (bounded by softmax)",
            "Partial attention-based self-reference",
            "Sequential feedback within generation"
        ],
        "what_transformers_lack": [
            "Operational closure (weights frozen at inference)",
            "Goal generation (all goals from prompts)",
            "True feedback loops (no environment coupling)",
            "Persistent identity (no cross-session memory)",
            "Embodiment (no physical grounding)",
            "Structural coupling (frozen at deployment)",
            "Metacognitive control (no genuine self-monitoring)"
        ],
        "architectural_fixes_needed": [
            "Persistent memory system for identity continuity",
            "Online learning for operational closure",
            "Goal generation module for self-determined action",
            "Environment coupling for true feedback",
            "Embodiment or grounding for situated action"
        ],
        "gaps_detail": gaps
    }


def list_frameworks() -> Dict:
    """List all autonomy frameworks and their components."""
    result = {}
    for framework, components in AUTONOMY_COMPONENTS.items():
        result[framework.value] = {
            "components": [
                {
                    "name": c.name,
                    "description": c.description,
                    "is_necessary": c.is_necessary
                }
                for c in components
            ],
            "count": len(components)
        }
    return result


def list_predefined_systems() -> Dict:
    """List all predefined system profiles available for analysis."""
    return {
        key: profile["description"]
        for key, profile in SYSTEM_PROFILES.items()
    }


# ── Design Guidance: How to Build Autonomous Systems ──────────────────

@dataclass
class ImplementationApproach:
    """A potential implementation for an autonomy component."""
    name: str
    description: str
    difficulty: str  # "low", "medium", "high", "research"
    existing_examples: List[str]
    integration_notes: str
    dependencies: List[str] = field(default_factory=list)


# Map each autonomy component to potential implementations
IMPLEMENTATION_APPROACHES: Dict[str, List[ImplementationApproach]] = {
    # Control Theory Components
    "feedback_loop": [
        ImplementationApproach(
            name="Agent Framework with Tool Use",
            description="ReAct/AutoGPT-style loops: observe environment → reason → act → observe",
            difficulty="medium",
            existing_examples=["AutoGPT", "BabyAGI", "LangChain agents", "Claude computer use"],
            integration_notes="Requires environment API (browser, filesystem, APIs). Key: output affects next input.",
            dependencies=["environment_interface"]
        ),
        ImplementationApproach(
            name="RLHF with Online Feedback",
            description="Continuous reward signal from environment, not just human preferences",
            difficulty="high",
            existing_examples=["Constitutional AI", "RLAIF", "Online RLHF"],
            integration_notes="Requires real-time reward computation. Can use model-as-judge.",
            dependencies=["reward_model", "training_infrastructure"]
        ),
    ],
    "setpoint_generation": [
        ImplementationApproach(
            name="Intrinsic Motivation Module",
            description="Curiosity-driven exploration: information gain, prediction error, novelty",
            difficulty="high",
            existing_examples=["ICM (Intrinsic Curiosity Module)", "RND", "Go-Explore"],
            integration_notes="Add auxiliary reward for uncertainty reduction. Can use ensemble disagreement.",
            dependencies=["uncertainty_estimation", "world_model"]
        ),
        ImplementationApproach(
            name="Hierarchical Goal Generation",
            description="High-level goals decomposed into subgoals automatically",
            difficulty="medium",
            existing_examples=["HAM", "Feudal Networks", "Goal-conditioned RL"],
            integration_notes="Learn goal abstraction from successful trajectories.",
            dependencies=["planning_module"]
        ),
        ImplementationApproach(
            name="Self-Prompted Continuation",
            description="Model generates its own next prompt based on incomplete goals",
            difficulty="low",
            existing_examples=["Chain-of-thought agents", "Self-ask", "Plan-and-solve"],
            integration_notes="Simple: append 'What should I do next?' after each response.",
            dependencies=[]
        ),
    ],
    "error_correction": [
        ImplementationApproach(
            name="Self-Critique and Revision",
            description="Model evaluates own output and generates corrections",
            difficulty="low",
            existing_examples=["Constitutional AI", "Self-refine", "Reflexion"],
            integration_notes="Add critique prompt after generation. Can be iterative.",
            dependencies=[]
        ),
        ImplementationApproach(
            name="Verification-Guided Revision",
            description="External verifier (code execution, proof checker) provides error signal",
            difficulty="medium",
            existing_examples=["AlphaCode", "CodeContests", "Lean copilots"],
            integration_notes="Requires domain-specific verifier. Very effective for formal domains.",
            dependencies=["verifier"]
        ),
    ],
    "disturbance_rejection": [
        ImplementationApproach(
            name="Robust Planning with Contingencies",
            description="Generate backup plans for likely failure modes",
            difficulty="medium",
            existing_examples=["MCTS with rollouts", "Robust MDPs"],
            integration_notes="During planning, simulate perturbations and generate recovery strategies.",
            dependencies=["planning_module", "world_model"]
        ),
    ],
    "stability": [
        ImplementationApproach(
            name="Output Filtering and Bounds",
            description="Constitutional constraints, output validation, safety filters",
            difficulty="low",
            existing_examples=["Constitutional AI", "Guardrails", "NeMo Guardrails"],
            integration_notes="Already present in most deployed systems.",
            dependencies=[]
        ),
    ],

    # Autopoiesis Components
    "operational_closure": [
        ImplementationApproach(
            name="Online Learning / Continual Learning",
            description="Update weights based on new experiences during deployment",
            difficulty="research",
            existing_examples=["LoRA online fine-tuning", "PEFT adapters", "Memory replay"],
            integration_notes="Major challenge: catastrophic forgetting. Use EWC, replay buffers.",
            dependencies=["training_infrastructure", "forgetting_prevention"]
        ),
        ImplementationApproach(
            name="Retrieval-Augmented Self-Modification",
            description="External memory that model can write to and read from",
            difficulty="medium",
            existing_examples=["MemGPT", "Voyager skill library", "LangChain memory"],
            integration_notes="Not true self-modification but approximates it. Persistent vector DB.",
            dependencies=["vector_database", "memory_manager"]
        ),
    ],
    "structural_coupling": [
        ImplementationApproach(
            name="Environment Grounding",
            description="Continuous bidirectional interaction with environment",
            difficulty="medium",
            existing_examples=["Robotics LLMs", "Embodied AI", "Voyager in Minecraft"],
            integration_notes="Environment state → perception → action → environment change loop.",
            dependencies=["environment_interface", "action_execution"]
        ),
    ],
    "self_production": [
        ImplementationApproach(
            name="Self-Improving Code Generation",
            description="Agent modifies its own code/prompts based on performance",
            difficulty="high",
            existing_examples=["Voyager skill library", "AutoML", "Neural architecture search"],
            integration_notes="Safety critical: need sandboxing. Start with prompt self-optimization.",
            dependencies=["sandboxed_execution", "performance_monitoring"]
        ),
    ],
    "boundary_maintenance": [
        ImplementationApproach(
            name="Resource Management Layer",
            description="Monitor and limit compute, memory, API calls, token budgets",
            difficulty="medium",
            existing_examples=["Kubernetes limits", "Token budgeting in agents"],
            integration_notes="Essential for multi-agent systems. Prevent runaway resource use.",
            dependencies=["resource_monitor"]
        ),
    ],
    "organizational_invariance": [
        ImplementationApproach(
            name="Consistent System Prompt / Constitution",
            description="Core values/goals persist across all interactions",
            difficulty="low",
            existing_examples=["Claude's constitution", "System prompts"],
            integration_notes="Already implemented. Key: ensure consistency across sessions.",
            dependencies=[]
        ),
    ],

    # Philosophy Components
    "intentionality": [
        ImplementationApproach(
            name="Explicit Goal Representation",
            description="Maintain and reason about explicit goal structures",
            difficulty="medium",
            existing_examples=["BDI agents", "STRIPS planning", "Goal-conditioned RL"],
            integration_notes="Represent goals as structured objects, not just text.",
            dependencies=["goal_representation"]
        ),
    ],
    "self_determination": [
        ImplementationApproach(
            name="Autonomous Decision Making",
            description="Agent chooses actions based on internal values, not just instructions",
            difficulty="high",
            existing_examples=["Value-aligned RL", "Constitutional AI"],
            integration_notes="Requires learned value function. Balance with instruction-following.",
            dependencies=["value_model", "intrinsic_motivation"]
        ),
    ],
    "normativity": [
        ImplementationApproach(
            name="Norm Learning and Following",
            description="Learn and apply social/ethical norms, detect violations",
            difficulty="medium",
            existing_examples=["Delphi", "Moral Foundations", "Constitutional AI"],
            integration_notes="Encode norms explicitly or learn from examples.",
            dependencies=["norm_database"]
        ),
    ],
    "counterfactual_sensitivity": [
        ImplementationApproach(
            name="Reason-Responsive Architecture",
            description="Decisions change appropriately with evidence/arguments",
            difficulty="medium",
            existing_examples=["Chain-of-thought", "Self-consistency", "Debate"],
            integration_notes="Already partially present. Strengthen with explicit reasoning traces.",
            dependencies=[]
        ),
    ],
    "practical_reasoning": [
        ImplementationApproach(
            name="Deliberative Planning",
            description="Explicit reasoning about options, consequences, and values",
            difficulty="low",
            existing_examples=["ReAct", "Chain-of-thought", "Tree-of-thought"],
            integration_notes="Already present. Can be strengthened with structured planning.",
            dependencies=[]
        ),
    ],

    # Robotics Components
    "embodiment": [
        ImplementationApproach(
            name="Physical Robot Integration",
            description="Deploy on physical robot with sensors and actuators",
            difficulty="high",
            existing_examples=["RT-2", "PaLM-E", "Mobile ALOHA"],
            integration_notes="Requires robotics hardware. High latency tolerance needed.",
            dependencies=["robot_hardware", "sensor_fusion"]
        ),
        ImplementationApproach(
            name="Simulated Embodiment",
            description="Operate in physics simulation (MuJoCo, Isaac Sim)",
            difficulty="medium",
            existing_examples=["Voyager", "MineDojo", "Habitat"],
            integration_notes="Good for development. Sim2real gap is a challenge.",
            dependencies=["simulator"]
        ),
        ImplementationApproach(
            name="Digital Tool Embodiment",
            description="Computer use, browser automation, API calls as 'body'",
            difficulty="low",
            existing_examples=["Claude computer use", "AutoGPT", "Browser agents"],
            integration_notes="Practical near-term option. Actions affect digital environment.",
            dependencies=["tool_interface"]
        ),
    ],
    "sensorimotor_coupling": [
        ImplementationApproach(
            name="Vision-Language-Action Models",
            description="Direct mapping from perception to action, low latency",
            difficulty="high",
            existing_examples=["RT-2", "PaLM-E", "OpenVLA"],
            integration_notes="Train end-to-end on robot data. Requires embodiment first.",
            dependencies=["embodiment", "vision_encoder", "action_decoder"]
        ),
    ],
    "real_time_operation": [
        ImplementationApproach(
            name="Latency-Optimized Inference",
            description="Fast inference for real-time control",
            difficulty="medium",
            existing_examples=["Distilled models", "Speculative decoding", "vLLM"],
            integration_notes="Trade off capability for speed. Use smaller models for control.",
            dependencies=["optimized_runtime"]
        ),
    ],
    "situated_action": [
        ImplementationApproach(
            name="Context-Aware Agent",
            description="Actions depend on full situational context, not just instructions",
            difficulty="medium",
            existing_examples=["Embodied agents", "Context-aware assistants"],
            integration_notes="Requires rich environment state representation.",
            dependencies=["environment_state_tracker"]
        ),
    ],
    "robust_behavior": [
        ImplementationApproach(
            name="Failure Recovery Protocols",
            description="Detect failures and execute recovery strategies",
            difficulty="medium",
            existing_examples=["Fault-tolerant robotics", "Exception handling in agents"],
            integration_notes="Monitor for anomalies, have fallback behaviors ready.",
            dependencies=["anomaly_detection", "recovery_strategies"]
        ),
    ],

    # Cognitive Science Components
    "metacognition": [
        ImplementationApproach(
            name="Confidence Estimation",
            description="Model estimates uncertainty in its own outputs",
            difficulty="medium",
            existing_examples=["Verbalized confidence", "Calibration training", "Ensembles"],
            integration_notes="Can use token probabilities or train separate confidence head.",
            dependencies=[]
        ),
        ImplementationApproach(
            name="Self-Monitoring Module",
            description="Track own reasoning process, detect errors",
            difficulty="high",
            existing_examples=["Self-reflection prompts", "Process supervision"],
            integration_notes="Add meta-level that monitors object-level reasoning.",
            dependencies=["reasoning_trace"]
        ),
    ],
    "persistent_identity": [
        ImplementationApproach(
            name="Long-Term Memory System",
            description="Persistent memory across sessions, continuous self-model",
            difficulty="medium",
            existing_examples=["MemGPT", "Personal AI assistants", "Memory banks"],
            integration_notes="Vector DB + retrieval. Include self-knowledge and preferences.",
            dependencies=["vector_database", "session_persistence"]
        ),
        ImplementationApproach(
            name="User-Specific Fine-Tuning",
            description="Personalized model weights per user/instance",
            difficulty="high",
            existing_examples=["Personal LLMs", "On-device fine-tuning"],
            integration_notes="Creates true persistent identity through weights.",
            dependencies=["per_instance_training"]
        ),
    ],
    "episodic_memory": [
        ImplementationApproach(
            name="Experience Replay Buffer",
            description="Store and retrieve specific past interactions",
            difficulty="medium",
            existing_examples=["MemGPT", "Generative agents", "Voyager"],
            integration_notes="Store (context, action, outcome) tuples. Retrieve by similarity.",
            dependencies=["memory_database"]
        ),
    ],
    "prospection": [
        ImplementationApproach(
            name="World Model + Planning",
            description="Simulate future states, plan based on predictions",
            difficulty="high",
            existing_examples=["Dreamer", "MuZero", "World models in agents"],
            integration_notes="Learn predictive model of environment dynamics.",
            dependencies=["world_model", "planning_module"]
        ),
        ImplementationApproach(
            name="Mental Simulation via Generation",
            description="Use LLM to simulate 'what if' scenarios",
            difficulty="low",
            existing_examples=["Chain-of-thought planning", "Self-play"],
            integration_notes="Already possible: 'Imagine you did X. What would happen?'",
            dependencies=[]
        ),
    ],
    "attention_control": [
        ImplementationApproach(
            name="Selective Context Management",
            description="Actively choose what to attend to in large contexts",
            difficulty="medium",
            existing_examples=["Retrieval augmentation", "Context compression"],
            integration_notes="Use retrieval to select relevant context dynamically.",
            dependencies=["retrieval_system"]
        ),
    ],
}


def get_implementation_roadmap(system_key: str = "llm_transformer") -> Dict:
    """
    Generate a roadmap for making a system more autonomous.

    Maps each missing or partial autonomy component to concrete
    implementation approaches with difficulty levels and examples.

    Args:
        system_key: System to analyze (default: llm_transformer)

    Returns:
        Dict with prioritized implementation roadmap
    """
    gaps = identify_autonomy_gaps(system_key)

    roadmap = {
        "system": system_key,
        "current_score": gaps["overall_score"],
        "target_score": 1.0,
        "quick_wins": [],      # Low difficulty, high impact
        "medium_term": [],     # Medium difficulty
        "research_frontier": [],  # High/research difficulty
        "implementation_details": {}
    }

    # Process absent components
    for gap in gaps["absent_components"]:
        component = gap["component"]
        if component in IMPLEMENTATION_APPROACHES:
            approaches = IMPLEMENTATION_APPROACHES[component]
            for approach in approaches:
                entry = {
                    "component": component,
                    "framework": gap["framework"],
                    "approach": approach.name,
                    "description": approach.description,
                    "difficulty": approach.difficulty,
                    "examples": approach.existing_examples,
                    "dependencies": approach.dependencies,
                    "notes": approach.integration_notes,
                    "is_necessary": gap["is_necessary"]
                }

                if approach.difficulty == "low":
                    roadmap["quick_wins"].append(entry)
                elif approach.difficulty in ["medium"]:
                    roadmap["medium_term"].append(entry)
                else:
                    roadmap["research_frontier"].append(entry)

                if component not in roadmap["implementation_details"]:
                    roadmap["implementation_details"][component] = []
                roadmap["implementation_details"][component].append(entry)

    # Process partial components (can be upgraded)
    for partial in gaps["partial_components"]:
        component = partial["component"]
        if component in IMPLEMENTATION_APPROACHES:
            approaches = IMPLEMENTATION_APPROACHES[component]
            for approach in approaches:
                entry = {
                    "component": component,
                    "framework": partial["framework"],
                    "approach": approach.name,
                    "description": approach.description,
                    "difficulty": approach.difficulty,
                    "examples": approach.existing_examples,
                    "dependencies": approach.dependencies,
                    "notes": approach.integration_notes,
                    "current_status": "partial",
                    "upgrade_from": partial["evidence"]
                }

                if approach.difficulty == "low":
                    roadmap["quick_wins"].append(entry)

                if component not in roadmap["implementation_details"]:
                    roadmap["implementation_details"][component] = []
                roadmap["implementation_details"][component].append(entry)

    # Sort by impact (necessary components first)
    for category in ["quick_wins", "medium_term", "research_frontier"]:
        roadmap[category].sort(key=lambda x: (not x.get("is_necessary", False), x["component"]))

    return roadmap


def design_autonomous_system(
    base_system: str = "llm_transformer",
    target_capabilities: Optional[List[str]] = None,
    max_difficulty: str = "medium"
) -> Dict:
    """
    Design an autonomous system by selecting components to add.

    Args:
        base_system: Starting system type
        target_capabilities: Specific capabilities to add (None = all feasible)
        max_difficulty: Maximum difficulty to consider ("low", "medium", "high")

    Returns:
        Dict with system design specification
    """
    roadmap = get_implementation_roadmap(base_system)

    difficulty_order = ["low", "medium", "high", "research"]
    max_idx = difficulty_order.index(max_difficulty)
    allowed_difficulties = set(difficulty_order[:max_idx + 1])

    # Filter approaches by difficulty
    selected_approaches = []
    for component, approaches in roadmap["implementation_details"].items():
        # Filter by target capabilities if specified
        if target_capabilities and component not in target_capabilities:
            continue

        # Find best approach within difficulty limit
        feasible = [a for a in approaches if a["difficulty"] in allowed_difficulties]
        if feasible:
            # Prefer lower difficulty
            best = min(feasible, key=lambda x: difficulty_order.index(x["difficulty"]))
            selected_approaches.append(best)

    # Compute dependency graph
    all_deps = set()
    for approach in selected_approaches:
        all_deps.update(approach.get("dependencies", []))

    # Estimate autonomy score improvement
    current_score = roadmap["current_score"]
    components_added = len(selected_approaches)
    estimated_improvement = min(0.7, components_added * 0.05)  # ~5% per component

    # Compute implementation order (topological sort by dependencies)
    implementation_order = _topological_sort(selected_approaches)

    return {
        "base_system": base_system,
        "current_autonomy_score": current_score,
        "estimated_final_score": min(1.0, current_score + estimated_improvement),
        "components_to_add": len(selected_approaches),
        "max_difficulty": max_difficulty,
        "required_dependencies": list(all_deps),
        "implementation_order": implementation_order,
        "selected_approaches": selected_approaches,
        "integration_checklist": _generate_integration_checklist(selected_approaches)
    }


def _topological_sort(approaches: List[Dict]) -> List[str]:
    """Sort approaches by dependencies."""
    # Simple ordering: approaches with no dependencies first
    no_deps = [a["component"] for a in approaches if not a.get("dependencies")]
    has_deps = [a["component"] for a in approaches if a.get("dependencies")]
    return no_deps + has_deps


def _generate_integration_checklist(approaches: List[Dict]) -> List[str]:
    """Generate integration checklist for implementation."""
    checklist = []

    # Infrastructure checks
    all_deps = set()
    for a in approaches:
        all_deps.update(a.get("dependencies", []))

    if all_deps:
        checklist.append("INFRASTRUCTURE:")
        for dep in sorted(all_deps):
            checklist.append(f"  [ ] Set up {dep}")

    # Per-component checks
    checklist.append("\nCOMPONENTS:")
    for approach in approaches:
        checklist.append(f"  [ ] {approach['component']}: {approach['approach']}")
        checklist.append(f"      Difficulty: {approach['difficulty']}")
        if approach.get("examples"):
            checklist.append(f"      Reference: {approach['examples'][0]}")

    # Testing checks
    checklist.append("\nVERIFICATION:")
    checklist.append("  [ ] Run autonomy assessment to verify improvement")
    checklist.append("  [ ] Test feedback loops with environment")
    checklist.append("  [ ] Verify goal generation capability")
    checklist.append("  [ ] Check memory persistence across sessions")

    return checklist


def get_minimum_viable_autonomy() -> Dict:
    """
    Get the minimum set of components for basic autonomy.

    Returns the simplest path to a minimally autonomous system,
    prioritizing low-difficulty implementations.
    """
    # Core requirements for minimal autonomy
    critical_components = [
        "feedback_loop",        # Must interact with environment
        "setpoint_generation",  # Must have some goals
        "persistent_identity",  # Must persist across time
        "error_correction",     # Must learn from mistakes
    ]

    design = design_autonomous_system(
        base_system="llm_transformer",
        target_capabilities=critical_components,
        max_difficulty="medium"
    )

    design["note"] = (
        "Minimum viable autonomy requires: (1) environment feedback loop, "
        "(2) some form of goal generation, (3) persistent identity/memory, "
        "(4) error correction capability. This is achievable with existing "
        "techniques (agent frameworks + memory systems)."
    )

    return design


def get_full_autonomy_blueprint() -> Dict:
    """
    Get a complete blueprint for full autonomy.

    Includes all components at all difficulty levels.
    This is aspirational - includes research-level challenges.
    """
    design = design_autonomous_system(
        base_system="llm_transformer",
        target_capabilities=None,  # All components
        max_difficulty="research"
    )

    design["note"] = (
        "Full autonomy requires solving several open research problems: "
        "operational closure (online learning without forgetting), "
        "true embodiment (physical or rich simulated), and "
        "genuine metacognition (not just verbalized confidence). "
        "The architectural gap is fundamental but not insurmountable."
    )

    return design
