"""Tool Graph — Automatic calculator chaining via metadata tags.

This module enables:
1. Declarative tool metadata (inputs, outputs, prerequisites, chains)
2. Automatic chain discovery via input/output type matching
3. Embedding-based routing for sequence prediction
4. Online learning of successful chains

Usage:
    @calculator(
        domain="mechanics",
        inputs=["elastic_modulus_E", "poisson_ratio_nu"],
        outputs=["bulk_modulus_K", "shear_modulus_G"],
        prerequisites=[],
        chains_to=["calc_seismic_velocity"],
        tags=["modulus_conversion", "foundational"]
    )
    def convert_elastic_moduli(E: float, nu: float) -> dict:
        ...
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple, Any
from functools import wraps
import json
import numpy as np
from pathlib import Path


@dataclass
class CalculatorMeta:
    """Metadata for a registered calculator."""
    name: str
    func: Callable
    domain: str
    inputs: List[str]  # Input type names
    outputs: List[str]  # Output type names
    prerequisites: List[str]  # Tools that often precede this one
    chains_to: List[str]  # Tools that often follow this one
    tags: List[str]  # Semantic tags for embedding
    description: str = ""

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name


class ToolRegistry:
    """Registry of all calculators with their metadata."""

    def __init__(self):
        self._tools: Dict[str, CalculatorMeta] = {}
        self._type_producers: Dict[str, Set[str]] = {}  # type -> tools that produce it
        self._type_consumers: Dict[str, Set[str]] = {}  # type -> tools that consume it
        self._embeddings: Optional[np.ndarray] = None
        self._embedding_index: Dict[str, int] = {}

    def register(self, meta: CalculatorMeta):
        """Register a calculator with its metadata."""
        self._tools[meta.name] = meta

        # Index by output types (producers)
        for out_type in meta.outputs:
            if out_type not in self._type_producers:
                self._type_producers[out_type] = set()
            self._type_producers[out_type].add(meta.name)

        # Index by input types (consumers)
        for in_type in meta.inputs:
            if in_type not in self._type_consumers:
                self._type_consumers[in_type] = set()
            self._type_consumers[in_type].add(meta.name)

    def get(self, name: str) -> Optional[CalculatorMeta]:
        return self._tools.get(name)

    def all_tools(self) -> List[CalculatorMeta]:
        return list(self._tools.values())

    def find_chains(self, start: str, max_depth: int = 3) -> List[List[str]]:
        """Find all valid chains starting from a tool."""
        chains = []
        self._find_chains_recursive(start, [], chains, max_depth, set())
        return chains

    def _find_chains_recursive(
        self,
        current: str,
        path: List[str],
        chains: List[List[str]],
        max_depth: int,
        visited: Set[str]
    ):
        if len(path) >= max_depth:
            return
        if current in visited:
            return

        tool = self._tools.get(current)
        if not tool:
            return

        new_path = path + [current]
        new_visited = visited | {current}

        # Record this chain if it has more than one tool
        if len(new_path) > 1:
            chains.append(new_path)

        # Find tools that can consume our outputs
        next_tools = set()
        for out_type in tool.outputs:
            consumers = self._type_consumers.get(out_type, set())
            next_tools.update(consumers)

        # Also include explicit chains_to
        next_tools.update(tool.chains_to)

        # Continue to next tools
        for next_tool in next_tools - new_visited:
            self._find_chains_recursive(next_tool, new_path, chains, max_depth, new_visited)

    def find_path(self, input_types: List[str], output_types: List[str]) -> Optional[List[str]]:
        """Find a chain of tools that transforms input_types to output_types."""
        # BFS to find shortest path
        from collections import deque

        input_set = set(input_types)
        output_set = set(output_types)

        # Find starting tools (all inputs satisfied by initial input_types)
        starts = []
        for name, tool in self._tools.items():
            if set(tool.inputs) <= input_set:
                starts.append(name)

        # Find ending tools (produce at least one output type)
        ends = set()
        for out_type in output_types:
            ends.update(self._type_producers.get(out_type, set()))

        if not starts or not ends:
            return None

        # BFS with available types tracking
        queue = deque([(tool, [tool], input_set) for tool in starts])
        visited_states = set()  # (tool, frozenset(available)) to handle different paths

        while queue:
            current, path, available_types = queue.popleft()

            state_key = (current, frozenset(available_types))
            if state_key in visited_states:
                continue
            visited_states.add(state_key)

            tool = self._tools[current]

            # Verify all inputs are available before "executing" this tool
            if not set(tool.inputs) <= available_types:
                continue

            # Add our outputs to available types
            new_available = available_types | set(tool.outputs)

            # Check if we've reached an end and can produce all required outputs
            if current in ends and output_set <= new_available:
                return path

            # Find next tools whose inputs can now be satisfied
            next_tools = set()
            for out_type in tool.outputs:
                next_tools.update(self._type_consumers.get(out_type, set()))
            next_tools.update(tool.chains_to)

            for next_tool in next_tools:
                next_tool_meta = self._tools.get(next_tool)
                if next_tool_meta and set(next_tool_meta.inputs) <= new_available:
                    queue.append((next_tool, path + [next_tool], new_available))

        return None

    def compute_embeddings(self, embed_fn: Optional[Callable] = None):
        """Compute embeddings for all tools based on their metadata."""
        if embed_fn is None:
            # Simple bag-of-words embedding
            all_tags = set()
            for tool in self._tools.values():
                all_tags.update(tool.tags)
                all_tags.update(tool.inputs)
                all_tags.update(tool.outputs)
                all_tags.add(tool.domain)

            tag_to_idx = {tag: i for i, tag in enumerate(sorted(all_tags))}
            dim = len(tag_to_idx)

            embeddings = []
            for name in sorted(self._tools.keys()):
                tool = self._tools[name]
                vec = np.zeros(dim)
                for tag in tool.tags + tool.inputs + tool.outputs + [tool.domain]:
                    if tag in tag_to_idx:
                        vec[tag_to_idx[tag]] = 1.0
                # Normalize
                if np.linalg.norm(vec) > 0:
                    vec = vec / np.linalg.norm(vec)
                embeddings.append(vec)
                self._embedding_index[name] = len(embeddings) - 1

            self._embeddings = np.array(embeddings)
        else:
            # Use provided embedding function
            embeddings = []
            for name in sorted(self._tools.keys()):
                tool = self._tools[name]
                # Create text representation for embedding
                text = f"{tool.domain} {' '.join(tool.tags)} {' '.join(tool.inputs)} {' '.join(tool.outputs)}"
                vec = embed_fn(text)
                embeddings.append(vec)
                self._embedding_index[name] = len(embeddings) - 1
            self._embeddings = np.array(embeddings)

    def find_similar_tools(self, tool_name: str, k: int = 5) -> List[Tuple[str, float]]:
        """Find k most similar tools by embedding."""
        if self._embeddings is None:
            self.compute_embeddings()

        if tool_name not in self._embedding_index:
            return []

        idx = self._embedding_index[tool_name]
        query = self._embeddings[idx]

        # Cosine similarity
        sims = self._embeddings @ query

        # Get top k (excluding self)
        indices = np.argsort(-sims)
        results = []
        for i in indices:
            name = [n for n, j in self._embedding_index.items() if j == i][0]
            if name != tool_name:
                results.append((name, float(sims[i])))
            if len(results) >= k:
                break

        return results

    def to_graph_dict(self) -> dict:
        """Export registry as a graph structure for visualization."""
        nodes = []
        edges = []

        for name, tool in self._tools.items():
            nodes.append({
                "id": name,
                "domain": tool.domain,
                "tags": tool.tags,
                "inputs": tool.inputs,
                "outputs": tool.outputs,
            })

            # Edges from type matching
            for out_type in tool.outputs:
                for consumer in self._type_consumers.get(out_type, set()):
                    if consumer != name:
                        edges.append({
                            "source": name,
                            "target": consumer,
                            "type": out_type,
                            "kind": "type_match"
                        })

            # Edges from explicit chains
            for target in tool.chains_to:
                edges.append({
                    "source": name,
                    "target": target,
                    "kind": "explicit_chain"
                })

        return {"nodes": nodes, "edges": edges}

    def save(self, path: Path):
        """Save registry to JSON."""
        data = {
            "tools": {
                name: {
                    "domain": t.domain,
                    "inputs": t.inputs,
                    "outputs": t.outputs,
                    "prerequisites": t.prerequisites,
                    "chains_to": t.chains_to,
                    "tags": t.tags,
                    "description": t.description,
                }
                for name, t in self._tools.items()
            }
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> 'ToolRegistry':
        """Load registry from JSON (without functions)."""
        with open(path) as f:
            data = json.load(f)

        registry = cls()
        for name, meta in data["tools"].items():
            registry._tools[name] = CalculatorMeta(
                name=name,
                func=lambda: None,  # Placeholder
                **meta
            )

        # Rebuild indices
        for name, tool in registry._tools.items():
            for out_type in tool.outputs:
                if out_type not in registry._type_producers:
                    registry._type_producers[out_type] = set()
                registry._type_producers[out_type].add(name)
            for in_type in tool.inputs:
                if in_type not in registry._type_consumers:
                    registry._type_consumers[in_type] = set()
                registry._type_consumers[in_type].add(name)

        return registry


# Global registry
_registry = ToolRegistry()


def calculator(
    domain: str,
    inputs: List[str],
    outputs: List[str],
    prerequisites: Optional[List[str]] = None,
    chains_to: Optional[List[str]] = None,
    tags: Optional[List[str]] = None,
    description: str = "",
):
    """Decorator to register a calculator with metadata.

    Example:
        @calculator(
            domain="mechanics",
            inputs=["elastic_modulus_E", "poisson_ratio_nu"],
            outputs=["bulk_modulus_K", "shear_modulus_G"],
            chains_to=["calc_seismic_velocity"],
            tags=["modulus_conversion", "foundational"]
        )
        def convert_elastic_moduli(E: float, nu: float) -> dict:
            K = E / (3 * (1 - 2*nu))
            G = E / (2 * (1 + nu))
            return {"K": K, "G": G}
    """
    def decorator(func: Callable) -> Callable:
        meta = CalculatorMeta(
            name=func.__name__,
            func=func,
            domain=domain,
            inputs=inputs,
            outputs=outputs,
            prerequisites=prerequisites or [],
            chains_to=chains_to or [],
            tags=tags or [],
            description=description or func.__doc__ or "",
        )
        _registry.register(meta)

        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        wrapper._calculator_meta = meta
        return wrapper

    return decorator


def get_registry() -> ToolRegistry:
    """Get the global tool registry."""
    return _registry


def find_tool_chain(input_types: List[str], output_types: List[str]) -> Optional[List[str]]:
    """Find a chain of tools to transform inputs to outputs."""
    return _registry.find_path(input_types, output_types)


def execute_chain(
    chain: List[str],
    initial_inputs: Dict[str, Any],
    registry: Optional[ToolRegistry] = None
) -> Dict[str, Any]:
    """Execute a chain of calculators, passing outputs as inputs."""
    reg = registry or _registry
    context = dict(initial_inputs)

    for tool_name in chain:
        tool = reg.get(tool_name)
        if not tool:
            raise ValueError(f"Unknown tool: {tool_name}")

        # Gather inputs for this tool
        func_inputs = {}
        for in_type in tool.inputs:
            if in_type in context:
                func_inputs[in_type] = context[in_type]

        # Call the tool
        result = tool.func(**func_inputs)

        # Merge outputs into context
        if isinstance(result, dict):
            context.update(result)
        else:
            # Single output - use first output type name
            if tool.outputs:
                context[tool.outputs[0]] = result

    return context


# Example calculators for testing
@calculator(
    domain="mechanics",
    inputs=["elastic_modulus_E", "poisson_ratio_nu"],
    outputs=["bulk_modulus_K", "shear_modulus_G"],
    chains_to=["calc_seismic_velocity"],
    tags=["modulus_conversion", "foundational", "material_properties"]
)
def convert_elastic_moduli(elastic_modulus_E: float, poisson_ratio_nu: float) -> dict:
    """Convert Young's modulus and Poisson's ratio to bulk and shear moduli."""
    E, nu = elastic_modulus_E, poisson_ratio_nu
    K = E / (3 * (1 - 2*nu))
    G = E / (2 * (1 + nu))
    return {"bulk_modulus_K": K, "shear_modulus_G": G}


@calculator(
    domain="mechanics",
    inputs=["bulk_modulus_K", "shear_modulus_G", "density_rho"],
    outputs=["P_wave_velocity", "S_wave_velocity"],
    prerequisites=["convert_elastic_moduli"],
    tags=["velocity_calculation", "seismic", "wave_propagation"]
)
def calc_seismic_velocity(bulk_modulus_K: float, shear_modulus_G: float, density_rho: float) -> dict:
    """Calculate P-wave and S-wave velocities from elastic moduli and density."""
    import math
    Vp = math.sqrt((bulk_modulus_K + 4/3 * shear_modulus_G) / density_rho)
    Vs = math.sqrt(shear_modulus_G / density_rho)
    return {"P_wave_velocity": Vp, "S_wave_velocity": Vs}


@calculator(
    domain="thermodynamics",
    inputs=["temperature_K", "pressure_Pa"],
    outputs=["density_rho"],
    tags=["ideal_gas", "density", "foundational"]
)
def calc_ideal_gas_density(temperature_K: float, pressure_Pa: float, molar_mass: float = 0.029) -> dict:
    """Calculate density using ideal gas law. Default molar mass is air (0.029 kg/mol)."""
    R = 8.314  # J/(mol·K)
    rho = (pressure_Pa * molar_mass) / (R * temperature_K)
    return {"density_rho": rho}


if __name__ == "__main__":
    # Demo
    print("=== Tool Graph Demo ===\n")

    registry = get_registry()
    print(f"Registered tools: {[t.name for t in registry.all_tools()]}")

    # Find chains from elastic moduli
    chains = registry.find_chains("convert_elastic_moduli", max_depth=3)
    print(f"\nChains from convert_elastic_moduli: {chains}")

    # Find path from E, nu to seismic velocities
    path = registry.find_path(
        ["elastic_modulus_E", "poisson_ratio_nu", "density_rho"],
        ["P_wave_velocity", "S_wave_velocity"]
    )
    print(f"\nPath from E, nu, rho to Vp, Vs: {path}")

    # Execute the chain
    if path:
        result = execute_chain(path, {
            "elastic_modulus_E": 70e9,  # 70 GPa (aluminum)
            "poisson_ratio_nu": 0.33,
            "density_rho": 2700,  # kg/m³
        })
        print(f"\nExecution result: Vp = {result['P_wave_velocity']:.0f} m/s, Vs = {result['S_wave_velocity']:.0f} m/s")

    # Compute embeddings and find similar
    registry.compute_embeddings()
    similar = registry.find_similar_tools("convert_elastic_moduli", k=3)
    print(f"\nTools similar to convert_elastic_moduli: {similar}")
