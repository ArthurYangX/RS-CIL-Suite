"""Test that all modules import without error."""


def test_models_import():
    from benchmark.models import SimpleEncoder, build_backbone, list_backbones
    assert "simple_encoder" in list_backbones()


def test_config_import():
    from benchmark.config import load_config, flatten_config


def test_methods_import():
    from benchmark.methods import get_method_registry
    reg = get_method_registry()
    assert len(reg) >= 15


def test_protocols_import():
    from benchmark.protocols.cil import PROTOCOLS, get_protocol
    assert "B1" in PROTOCOLS


def test_utils_import():
    from benchmark.utils import build_optimizer, build_scheduler, remap_labels
    from benchmark.utils.exemplars import ExemplarMemory, list_strategies


def test_eval_import():
    from benchmark.eval.metrics import evaluate, BenchmarkResult
    from benchmark.eval.plots import plot_task_curves
    from benchmark.eval.colors import get_colormap


def test_compare_import():
    from benchmark.compare import load_results, print_table, print_latex, print_markdown


def test_run_import():
    from benchmark.run import run, main_cli, _build_parser


def test_infer_import():
    from benchmark.infer import main
