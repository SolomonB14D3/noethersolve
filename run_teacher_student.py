#!/usr/bin/env python3
"""Run teacher-student autonomous research with large cached models."""

import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Set HF_HOME to use the 4TB drive cache
os.environ["HF_HOME"] = "/Volumes/4TB SD/ml_cache/huggingface"

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

def main():
    print(f"\n{'='*60}")
    print(f"Teacher-Student Research Starting")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"HF_HOME: {os.environ.get('HF_HOME')}")
    print(f"{'='*60}\n")
    
    # Use 7B teacher with 0.5B student (MLX 4-bit, confirmed working)
    teacher_model = "mlx-community/Qwen2.5-7B-Instruct-4bit"
    student_model = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
    
    # Check available problem files and find one with many facts
    problems_dir = Path("problems")
    yamls = list(problems_dir.glob("*.yaml"))
    print(f"Found {len(yamls)} problem files")
    
    # Find problem with most facts
    best_problem = None
    max_facts = 0
    import yaml
    import json
    
    for yaml_path in yamls:
        try:
            with open(yaml_path) as f:
                prob = yaml.safe_load(f)
            vs = prob.get("verification_set")
            if vs:
                if not Path(vs).is_absolute():
                    vs = yaml_path.parent / vs
                if Path(vs).exists():
                    with open(vs) as f:
                        data = json.load(f)
                    count = len(data) if isinstance(data, list) else len(data.get("facts", []))
                    if count > max_facts:
                        max_facts = count
                        best_problem = str(yaml_path)
        except:
            continue
    
    problem_file = best_problem or str(yamls[0]) if yamls else None
    
    if not problem_file:
        print("ERROR: No problem YAML files found!")
        return
        
    print(f"Problem file: {problem_file}")
    print(f"Facts count: {max_facts}")
    print(f"Teacher model: {teacher_model}")
    print(f"Student model: {student_model}")
    print(f"Loading teacher-student framework...")
    
    from noethersolve.teacher_student import TeacherStudentResearch, TeacherStudentConfig
    
    config = TeacherStudentConfig(
        teacher_model=teacher_model,
        student_model=student_model,
        adapter_steps=500,
        teacher_confidence_threshold=2.0,  # Lower threshold for more training data
        student_improvement_threshold=0.3,
        unload_teacher_during_training=True,
    )
    
    research = TeacherStudentResearch(config=config)
    
    print(f"Config: adapter_steps={config.adapter_steps}")
    print(f"Config: confidence_threshold={config.teacher_confidence_threshold}")
    print()
    
    try:
        session = research.run_autonomous_loop(
            problem_yaml=problem_file,
            max_iterations=100,
        )
        
        print(f"\n{'='*60}")
        print(f"Session Complete")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        
        stats = research.get_session_stats()
        print(f"Teacher evaluations: {stats['total_teacher_evals']}")
        print(f"Student trainings: {stats['total_student_trains']}")
        print(f"Discoveries: {stats['discoveries']}")
        print(f"Results: {stats['results_count']}")
        
        if session.discoveries:
            print("\nDiscoveries:")
            for d in session.discoveries:
                print(f"  - {d}")
                
    except Exception as e:
        print(f"\nERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        
    print(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
