"""Tests for the tool graph and calculator chaining framework."""

from pathlib import Path
import tempfile

from noethersolve.tool_graph import (
    ToolRegistry,
    CalculatorMeta,
    calculator,
    get_registry,
    execute_chain,
)


class TestCalculatorDecorator:
    """Test the @calculator decorator."""

    def test_decorator_registers_tool(self):
        ToolRegistry()

        @calculator(
            domain="test",
            inputs=["a"],
            outputs=["b"],
            tags=["test_tag"]
        )
        def test_func(a):
            return {"b": a * 2}

        # The global registry should have it
        global_reg = get_registry()
        assert "test_func" in [t.name for t in global_reg.all_tools()]

    def test_decorator_preserves_function(self):
        @calculator(
            domain="test",
            inputs=["x"],
            outputs=["y"],
        )
        def double(x):
            return {"y": x * 2}

        result = double(5)
        assert result == {"y": 10}

    def test_decorator_attaches_metadata(self):
        @calculator(
            domain="math",
            inputs=["x", "y"],
            outputs=["sum", "product"],
            chains_to=["downstream_tool"],
            tags=["arithmetic"]
        )
        def compute(x, y):
            return {"sum": x + y, "product": x * y}

        meta = compute._calculator_meta
        assert meta.domain == "math"
        assert meta.inputs == ["x", "y"]
        assert meta.outputs == ["sum", "product"]
        assert "downstream_tool" in meta.chains_to
        assert "arithmetic" in meta.tags


class TestToolRegistry:
    """Test the ToolRegistry class."""

    def test_register_and_get(self):
        registry = ToolRegistry()
        meta = CalculatorMeta(
            name="test_tool",
            func=lambda x: x,
            domain="test",
            inputs=["a"],
            outputs=["b"],
            prerequisites=[],
            chains_to=[],
            tags=["test"],
        )
        registry.register(meta)

        retrieved = registry.get("test_tool")
        assert retrieved is not None
        assert retrieved.name == "test_tool"

    def test_type_indexing(self):
        registry = ToolRegistry()

        # Tool that produces type_x
        meta1 = CalculatorMeta(
            name="producer",
            func=lambda: None,
            domain="test",
            inputs=[],
            outputs=["type_x"],
            prerequisites=[],
            chains_to=[],
            tags=[],
        )

        # Tool that consumes type_x
        meta2 = CalculatorMeta(
            name="consumer",
            func=lambda: None,
            domain="test",
            inputs=["type_x"],
            outputs=["type_y"],
            prerequisites=[],
            chains_to=[],
            tags=[],
        )

        registry.register(meta1)
        registry.register(meta2)

        assert "producer" in registry._type_producers["type_x"]
        assert "consumer" in registry._type_consumers["type_x"]

    def test_find_chains(self):
        registry = ToolRegistry()

        # A -> B -> C chain
        for name, inp, out in [("A", [], ["x"]), ("B", ["x"], ["y"]), ("C", ["y"], ["z"])]:
            registry.register(CalculatorMeta(
                name=name, func=lambda: None, domain="test",
                inputs=inp, outputs=out, prerequisites=[], chains_to=[], tags=[]
            ))

        chains = registry.find_chains("A", max_depth=4)
        # Should find A->B, A->B->C
        assert ["A", "B"] in chains
        assert ["A", "B", "C"] in chains

    def test_find_path(self):
        registry = ToolRegistry()

        # Set up: input_a -> tool1 -> intermediate -> tool2 -> output_b
        registry.register(CalculatorMeta(
            name="tool1", func=lambda: None, domain="test",
            inputs=["input_a"], outputs=["intermediate"],
            prerequisites=[], chains_to=[], tags=[]
        ))
        registry.register(CalculatorMeta(
            name="tool2", func=lambda: None, domain="test",
            inputs=["intermediate"], outputs=["output_b"],
            prerequisites=[], chains_to=[], tags=[]
        ))

        path = registry.find_path(["input_a"], ["output_b"])
        assert path == ["tool1", "tool2"]

    def test_find_path_no_solution(self):
        registry = ToolRegistry()
        registry.register(CalculatorMeta(
            name="isolated", func=lambda: None, domain="test",
            inputs=["x"], outputs=["y"],
            prerequisites=[], chains_to=[], tags=[]
        ))

        path = registry.find_path(["a"], ["z"])
        assert path is None


class TestEmbeddings:
    """Test embedding computation and similarity search."""

    def test_compute_embeddings(self):
        registry = ToolRegistry()
        for i in range(3):
            registry.register(CalculatorMeta(
                name=f"tool_{i}",
                func=lambda: None,
                domain="test",
                inputs=[f"in_{i}"],
                outputs=[f"out_{i}"],
                prerequisites=[],
                chains_to=[],
                tags=[f"tag_{i}"],
            ))

        registry.compute_embeddings()
        assert registry._embeddings is not None
        assert registry._embeddings.shape[0] == 3

    def test_similar_tools(self):
        registry = ToolRegistry()

        # Two similar tools (same domain, overlapping tags)
        registry.register(CalculatorMeta(
            name="tool_a", func=lambda: None, domain="physics",
            inputs=["x"], outputs=["y"],
            prerequisites=[], chains_to=[], tags=["mechanics", "velocity"]
        ))
        registry.register(CalculatorMeta(
            name="tool_b", func=lambda: None, domain="physics",
            inputs=["y"], outputs=["z"],
            prerequisites=[], chains_to=[], tags=["mechanics", "acceleration"]
        ))
        # One different tool
        registry.register(CalculatorMeta(
            name="tool_c", func=lambda: None, domain="chemistry",
            inputs=["a"], outputs=["b"],
            prerequisites=[], chains_to=[], tags=["reaction", "kinetics"]
        ))

        registry.compute_embeddings()
        similar = registry.find_similar_tools("tool_a", k=2)

        # tool_b should be more similar to tool_a than tool_c
        names = [name for name, _ in similar]
        assert "tool_b" in names


class TestChainExecution:
    """Test executing tool chains."""

    def test_execute_simple_chain(self):
        registry = ToolRegistry()

        def add_one(x):
            return {"y": x + 1}

        def double(y):
            return {"z": y * 2}

        registry.register(CalculatorMeta(
            name="add_one", func=add_one, domain="test",
            inputs=["x"], outputs=["y"],
            prerequisites=[], chains_to=[], tags=[]
        ))
        registry.register(CalculatorMeta(
            name="double", func=double, domain="test",
            inputs=["y"], outputs=["z"],
            prerequisites=[], chains_to=[], tags=[]
        ))

        result = execute_chain(["add_one", "double"], {"x": 5}, registry)
        assert result["y"] == 6
        assert result["z"] == 12

    def test_execute_chain_preserves_context(self):
        registry = ToolRegistry()

        def step1(a, b):
            return {"c": a + b}

        def step2(c, b):  # Uses both new output and original input
            return {"d": c * b}

        registry.register(CalculatorMeta(
            name="step1", func=step1, domain="test",
            inputs=["a", "b"], outputs=["c"],
            prerequisites=[], chains_to=[], tags=[]
        ))
        registry.register(CalculatorMeta(
            name="step2", func=step2, domain="test",
            inputs=["c", "b"], outputs=["d"],
            prerequisites=[], chains_to=[], tags=[]
        ))

        result = execute_chain(["step1", "step2"], {"a": 3, "b": 4}, registry)
        assert result["c"] == 7  # 3 + 4
        assert result["d"] == 28  # 7 * 4


class TestSerialization:
    """Test saving and loading registry."""

    def test_save_and_load(self):
        registry = ToolRegistry()
        registry.register(CalculatorMeta(
            name="test_tool",
            func=lambda: None,
            domain="test_domain",
            inputs=["in1", "in2"],
            outputs=["out1"],
            prerequisites=["prereq"],
            chains_to=["next"],
            tags=["tag1", "tag2"],
            description="Test description"
        ))

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            registry.save(path)
            loaded = ToolRegistry.load(path)

            tool = loaded.get("test_tool")
            assert tool is not None
            assert tool.domain == "test_domain"
            assert tool.inputs == ["in1", "in2"]
            assert tool.outputs == ["out1"]
            assert tool.tags == ["tag1", "tag2"]
        finally:
            path.unlink()


class TestGraphExport:
    """Test graph structure export."""

    def test_to_graph_dict(self):
        registry = ToolRegistry()

        registry.register(CalculatorMeta(
            name="A", func=lambda: None, domain="test",
            inputs=[], outputs=["x"],
            prerequisites=[], chains_to=["B"], tags=[]
        ))
        registry.register(CalculatorMeta(
            name="B", func=lambda: None, domain="test",
            inputs=["x"], outputs=["y"],
            prerequisites=[], chains_to=[], tags=[]
        ))

        graph = registry.to_graph_dict()

        assert len(graph["nodes"]) == 2
        assert any(n["id"] == "A" for n in graph["nodes"])
        assert any(n["id"] == "B" for n in graph["nodes"])

        # Should have edges: A->B (type_match on x) and A->B (explicit_chain)
        edges_ab = [e for e in graph["edges"] if e["source"] == "A" and e["target"] == "B"]
        assert len(edges_ab) >= 1


class TestIntegrationWithExamples:
    """Test with the example calculators defined in the module."""

    def test_seismic_velocity_chain(self):
        """Test the E, nu -> K, G -> Vp, Vs chain."""
        registry = get_registry()

        # Find the chain
        path = registry.find_path(
            ["elastic_modulus_E", "poisson_ratio_nu", "density_rho"],
            ["P_wave_velocity", "S_wave_velocity"]
        )

        assert path is not None
        assert "convert_elastic_moduli" in path
        assert "calc_seismic_velocity" in path

        # Execute it
        result = execute_chain(path, {
            "elastic_modulus_E": 70e9,  # 70 GPa (aluminum)
            "poisson_ratio_nu": 0.33,
            "density_rho": 2700,
        }, registry)

        # Check physically reasonable values
        assert 5000 < result["P_wave_velocity"] < 7000  # m/s for aluminum
        assert 2500 < result["S_wave_velocity"] < 4000  # m/s for aluminum
