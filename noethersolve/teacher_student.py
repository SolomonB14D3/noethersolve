"""Teacher-Student Research Framework.

Use a larger model (e.g., 32B) as oracle/teacher to guide
research and adapter training on a smaller model (e.g., 3B).

Architecture:
    1. Teacher model: Scores facts, determines ground truth
    2. Student model: Target for adapter training
    3. Knowledge flows: Teacher's judgments → Student's adapters

Memory requirements (4-bit quantized):
    - Qwen3.5-27B-4bit: ~14GB
    - Qwen3-4B-4bit:    ~2GB
    - Total:            ~16GB (fits easily in 96GB with headroom)

Usage:
    from noethersolve.teacher_student import TeacherStudentResearch

    research = TeacherStudentResearch(
        teacher_model="mlx-community/Qwen3.5-27B-4bit",
        student_model="mlx-community/Qwen3-4B-4bit",
    )

    # Run autonomous research loop
    results = research.run_autonomous_loop(
        problem_yaml="problems/vortex_conservation.yaml",
        max_iterations=100,
    )
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable
import json
import os
import time

import numpy as np


@dataclass
class TeacherStudentConfig:
    """Configuration for teacher-student research."""
    teacher_model: str = "mlx-community/Qwen3.5-27B-4bit"
    student_model: str = "mlx-community/Qwen3-4B-4bit"

    # Training hyperparameters
    adapter_steps: int = 1000
    adapter_lr: float = 1e-5
    d_inner: int = 64
    margin_target: float = 1.5

    # Oracle thresholds
    teacher_confidence_threshold: float = 5.0  # Minimum margin to trust teacher
    student_improvement_threshold: float = 0.5  # Minimum improvement to keep adapter

    # Resource management
    unload_teacher_during_training: bool = True  # Free VRAM during student training
    checkpoint_interval: int = 10  # Save every N iterations


@dataclass
class ResearchResult:
    """Result of a single research iteration."""
    hypothesis: str
    teacher_margin: float
    teacher_verdict: str  # "PASS", "FAIL", "UNCERTAIN"
    student_baseline_margin: float
    student_adapted_margin: Optional[float] = None
    adapter_path: Optional[Path] = None
    improvement: float = 0.0
    notes: str = ""


@dataclass
class ResearchSession:
    """Full research session state."""
    config: TeacherStudentConfig
    results: list = field(default_factory=list)
    total_teacher_evals: int = 0
    total_student_trains: int = 0
    discoveries: list = field(default_factory=list)


class TeacherStudentResearch:
    """Orchestrates teacher-student autonomous research.

    The teacher model (larger, more capable) serves as the oracle:
    - Scores candidate facts
    - Determines what is "true" (positive margin = truth)
    - Its judgments become training signal for student

    The student model (smaller, faster) is the training target:
    - Receives adapters trained on teacher's judgments
    - Goal: match teacher's truth preferences with minimal capacity
    """

    def __init__(
        self,
        teacher_model: str = "Qwen/Qwen3-32B-Base",
        student_model: str = "Qwen/Qwen3-3B-Base",
        config: Optional[TeacherStudentConfig] = None,
    ):
        self.config = config or TeacherStudentConfig(
            teacher_model=teacher_model,
            student_model=student_model,
        )

        # Lazy-loaded models
        self._teacher = None
        self._teacher_tokenizer = None
        self._teacher_lm_head = None

        self._student = None
        self._student_tokenizer = None
        self._student_lm_head = None

        self._session = ResearchSession(config=self.config)

    def _load_teacher(self):
        """Load teacher model (lazy)."""
        if self._teacher is not None:
            return

        import mlx_lm
        from noethersolve.train_utils import get_lm_head_fn

        print(f"Loading teacher: {self.config.teacher_model}")
        t0 = time.time()
        self._teacher, self._teacher_tokenizer = mlx_lm.load(
            self.config.teacher_model
        )
        self._teacher.freeze()
        self._teacher_lm_head = get_lm_head_fn(self._teacher)
        print(f"  Teacher loaded in {time.time() - t0:.1f}s")

    def _load_student(self):
        """Load student model (lazy)."""
        if self._student is not None:
            return

        import mlx_lm
        from noethersolve.train_utils import get_lm_head_fn

        print(f"Loading student: {self.config.student_model}")
        t0 = time.time()
        self._student, self._student_tokenizer = mlx_lm.load(
            self.config.student_model
        )
        self._student.freeze()
        self._student_lm_head = get_lm_head_fn(self._student)
        print(f"  Student loaded in {time.time() - t0:.1f}s")

    def _unload_teacher(self):
        """Unload teacher to free VRAM for student training."""
        if self._teacher is None:
            return

        import mlx.core as mx

        print("  Unloading teacher to free VRAM...")
        del self._teacher
        del self._teacher_tokenizer
        del self._teacher_lm_head
        self._teacher = None
        self._teacher_tokenizer = None
        self._teacher_lm_head = None
        mx.metal.clear_cache()

    def _unload_student(self):
        """Unload student to free VRAM for teacher scoring."""
        if self._student is None:
            return

        import mlx.core as mx

        del self._student
        del self._student_tokenizer
        del self._student_lm_head
        self._student = None
        self._student_tokenizer = None
        self._student_lm_head = None
        mx.metal.clear_cache()

    def score_with_teacher(self, facts: list) -> dict:
        """Score facts using the teacher model.

        Returns dict with margins and verdicts.
        """
        self._load_teacher()
        from noethersolve.oracle import score_fact_mc

        results = []
        for fact in facts:
            win, margin, truth_lp, best_dist_lp = score_fact_mc(
                self._teacher,
                self._teacher_tokenizer,
                fact["context"],
                fact["truth"],
                fact["distractors"],
                lm_head=self._teacher_lm_head,
            )

            # Determine verdict based on confidence
            if margin > self.config.teacher_confidence_threshold:
                verdict = "PASS"
            elif margin < -self.config.teacher_confidence_threshold:
                verdict = "FAIL"
            else:
                verdict = "UNCERTAIN"

            results.append({
                "context": fact["context"],
                "truth": fact["truth"],
                "margin": float(margin),
                "verdict": verdict,
                "win": bool(win),
            })

        self._session.total_teacher_evals += len(facts)

        return {
            "results": results,
            "mean_margin": float(np.mean([r["margin"] for r in results])),
            "n_pass": sum(1 for r in results if r["verdict"] == "PASS"),
            "n_fail": sum(1 for r in results if r["verdict"] == "FAIL"),
            "n_uncertain": sum(1 for r in results if r["verdict"] == "UNCERTAIN"),
        }

    def score_with_student(self, facts: list, adapter=None) -> dict:
        """Score facts using the student model.

        Optionally with an adapter applied.
        """
        self._load_student()
        from noethersolve.oracle import score_fact_mc

        results = []
        for fact in facts:
            win, margin, _, _ = score_fact_mc(
                self._student,
                self._student_tokenizer,
                fact["context"],
                fact["truth"],
                fact["distractors"],
                adapter=adapter,
                lm_head=self._student_lm_head,
            )
            results.append({
                "context": fact["context"],
                "truth": fact["truth"],
                "margin": float(margin),
                "win": bool(win),
            })

        return {
            "results": results,
            "mean_margin": float(np.mean([r["margin"] for r in results])),
            "n_pass": sum(1 for r in results if r["win"]),
        }

    def train_student_adapter(
        self,
        training_examples: list,
        save_path: Optional[Path] = None,
    ):
        """Train adapter on student model using teacher's judgments.

        training_examples: List of facts where teacher had high confidence.
        """
        if self.config.unload_teacher_during_training:
            self._unload_teacher()

        self._load_student()

        # Import training utilities
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "training" / "scripts"))
        from train_vortex_adapter import train_vortex_adapter

        # Convert to training format
        tuples = []
        for ex in training_examples:
            ctx = ex.get("context", "")
            truth = ex.get("truth", "")
            distractors = ex.get("distractors", [])
            while len(distractors) < 3:
                distractors.append("This is incorrect")
            tuples.append((ctx, truth, distractors[:4], "teacher_guided"))

        # Train
        adapter = train_vortex_adapter(
            self._student,
            self._student_tokenizer,
            self._student_lm_head,
            tuples,
            steps=self.config.adapter_steps,
            lr=self.config.adapter_lr,
            d_inner=self.config.d_inner,
            margin_target=self.config.margin_target,
        )

        self._session.total_student_trains += 1

        # Save if path provided
        if save_path:
            import mlx.core as mx
            from mlx.utils import tree_flatten
            weights = dict(tree_flatten(adapter.parameters()))
            mx.savez(str(save_path), **weights)

        return adapter

    def research_iteration(
        self,
        hypothesis: str,
        facts: list,
        training_examples: list,
        adapters_dir: Path,
    ) -> ResearchResult:
        """Run one iteration of teacher-student research.

        1. Teacher scores the facts
        2. If teacher is confident, train student adapter
        3. Measure student improvement
        """
        result = ResearchResult(
            hypothesis=hypothesis,
            teacher_margin=0.0,
            teacher_verdict="PENDING",
            student_baseline_margin=0.0,
        )

        # Step 1: Teacher scoring
        print(f"\n[Teacher] Scoring: {hypothesis[:60]}...")
        teacher_result = self.score_with_teacher(facts)
        result.teacher_margin = teacher_result["mean_margin"]

        if teacher_result["n_uncertain"] > len(facts) // 2:
            result.teacher_verdict = "UNCERTAIN"
            result.notes = "Teacher uncertain on majority of facts"
            return result

        if teacher_result["n_fail"] > teacher_result["n_pass"]:
            result.teacher_verdict = "FAIL"
            result.notes = "Teacher judged this hypothesis as false"
            return result

        result.teacher_verdict = "PASS"

        # Step 2: Get student baseline
        print(f"[Student] Baseline scoring...")
        student_baseline = self.score_with_student(facts)
        result.student_baseline_margin = student_baseline["mean_margin"]

        # Step 3: Train student adapter on teacher-approved facts
        high_confidence_examples = [
            ex for ex, r in zip(training_examples, teacher_result["results"])
            if r["margin"] > self.config.teacher_confidence_threshold
        ]

        if len(high_confidence_examples) < 5:
            result.notes = f"Only {len(high_confidence_examples)} high-confidence examples, skipping training"
            return result

        print(f"[Student] Training on {len(high_confidence_examples)} teacher-approved examples...")
        adapter_path = adapters_dir / f"{hypothesis.replace(' ', '_')[:40]}.npz"
        adapter = self.train_student_adapter(
            high_confidence_examples,
            save_path=adapter_path,
        )

        # Step 4: Measure improvement
        print(f"[Student] Measuring improvement...")
        student_adapted = self.score_with_student(facts, adapter=adapter)
        result.student_adapted_margin = student_adapted["mean_margin"]
        result.adapter_path = adapter_path
        result.improvement = result.student_adapted_margin - result.student_baseline_margin

        if result.improvement > self.config.student_improvement_threshold:
            result.notes = f"Success! Student improved by {result.improvement:.2f}"
            self._session.discoveries.append(hypothesis)
        else:
            result.notes = f"Minimal improvement ({result.improvement:.2f})"

        self._session.results.append(result)
        return result

    def run_autonomous_loop(
        self,
        problem_yaml: str,
        max_iterations: int = 100,
        hypothesis_generator: Optional[Callable] = None,
    ) -> ResearchSession:
        """Run autonomous research loop.

        1. Generate/load hypotheses
        2. For each: teacher scores → student trains → measure
        3. Accumulate discoveries
        """
        import yaml

        with open(problem_yaml) as f:
            problem = yaml.safe_load(f)

        adapters_dir = Path(__file__).parent.parent / "adapters" / "teacher_student"
        adapters_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Teacher-Student Autonomous Research")
        print(f"Teacher: {self.config.teacher_model}")
        print(f"Student: {self.config.student_model}")
        print(f"Max iterations: {max_iterations}")
        print(f"{'='*60}\n")

        # TODO: Integrate with autonomy_loop.py hypothesis generation
        # For now, load from problem's verification set
        facts_path = problem.get("verification_set")
        if facts_path:
            problem_dir = Path(problem_yaml).parent
            if not Path(facts_path).is_absolute():
                facts_path = problem_dir / facts_path
            with open(facts_path) as f:
                facts_data = json.load(f)
                if isinstance(facts_data, list):
                    facts = facts_data
                else:
                    facts = facts_data.get("facts", [])

            # Run single iteration as demo
            result = self.research_iteration(
                hypothesis=problem.get("name", "unknown"),
                facts=facts[:5],  # Limit for demo
                training_examples=facts,
                adapters_dir=adapters_dir,
            )
            print(f"\nResult: {result}")

        return self._session

    def get_session_stats(self) -> dict:
        """Get statistics for current session."""
        return {
            "total_teacher_evals": self._session.total_teacher_evals,
            "total_student_trains": self._session.total_student_trains,
            "discoveries": len(self._session.discoveries),
            "results_count": len(self._session.results),
        }


# Convenience function
def run_teacher_student_research(
    teacher: str = "mlx-community/Qwen3.5-27B-4bit",
    student: str = "mlx-community/Qwen3-4B-4bit",
    problem_yaml: str = "problems/vortex_conservation.yaml",
    max_iterations: int = 100,
) -> ResearchSession:
    """Run teacher-student research with default settings."""
    research = TeacherStudentResearch(
        teacher_model=teacher,
        student_model=student,
    )
    return research.run_autonomous_loop(
        problem_yaml=problem_yaml,
        max_iterations=max_iterations,
    )
