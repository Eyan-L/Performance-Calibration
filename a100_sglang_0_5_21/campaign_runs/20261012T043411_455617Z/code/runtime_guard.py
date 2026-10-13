"""Verify the imported SGLang version and source before running calibration code."""
from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
import runpy
import sys
import tempfile
from pathlib import Path


def check_runtime() -> dict:
    config = json.loads(Path(__file__).with_name('runtime_config.json').read_text())
    expected_version = config['sglang_version']
    expected_root = Path(config['sglang_source_root']).resolve()
    cache_root = Path(tempfile.gettempdir()) / f'sglang_0521_calibration_{os.getuid()}'
    cache_paths = {}
    for variable, directory in (
        ('XDG_CACHE_HOME', 'xdg'), ('TRITON_CACHE_DIR', 'triton'),
        ('TORCH_EXTENSIONS_DIR', 'torch'), ('FLASHINFER_WORKSPACE_BASE', 'flashinfer'),
        ('TVM_FFI_CACHE_DIR', 'tvm'), ('TORCHINDUCTOR_CACHE_DIR', 'inductor'),
    ):
        path = cache_root / directory
        path.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(path)
        cache_paths[variable] = str(path)

    import sglang
    import sglang.version
    import torch

    installed_version = importlib.metadata.version('sglang')
    source_version = sglang.version.__version__
    if installed_version != expected_version or source_version != expected_version:
        raise RuntimeError(
            f'Expected SGLang {expected_version}; distribution={installed_version}, '
            f'imported source={source_version}'
        )
    source_file = Path(sglang.__file__).resolve()
    if not source_file.is_relative_to(expected_root):
        raise RuntimeError(f'Unexpected SGLang source: {source_file}')
    benchmark = importlib.util.find_spec(config['native_benchmark_module'])
    if benchmark is None or benchmark.origin is None:
        raise RuntimeError('Official SGLang one_batch benchmark is missing')
    benchmark_file = Path(benchmark.origin).resolve()
    if not benchmark_file.is_relative_to(expected_root):
        raise RuntimeError(f'Unexpected benchmark source: {benchmark_file}')
    reference_src = (Path(config['reference_kit']) / 'src').resolve()
    if any(Path(p).resolve().is_relative_to(reference_src) for p in sys.path):
        raise RuntimeError('Reference kit source leaked into the runtime import path')

    return {
        'sglang_distribution_version': installed_version,
        'sglang_source_version': source_version,
        'sglang_import_path': str(source_file),
        'native_benchmark_path': str(benchmark_file),
        'torch': torch.__version__,
        'torch_cuda': torch.version.cuda,
        'transformers': importlib.metadata.version('transformers'),
        'python': sys.version,
        'image': config['image'],
        'cache_paths': cache_paths,
        'reference_source_on_import_path': False,
        'gpu_execution_tested': False,
    }


def main() -> None:
    report = check_runtime()
    args = sys.argv[1:]
    if not args or args == ['--check']:
        print(json.dumps(report, indent=2))
        return
    print(json.dumps({'runtime_verified': report}), file=sys.stderr, flush=True)
    if args[0] == '--module':
        if len(args) < 2:
            raise SystemExit('--module requires a module name')
        sys.argv = args[1:]
        runpy.run_module(args[1], run_name='__main__', alter_sys=True)
    else:
        script = Path(args[0]).resolve()
        reference_src = (Path(json.loads(Path(__file__).with_name('runtime_config.json').read_text())['reference_kit']) / 'src').resolve()
        if script.is_relative_to(reference_src):
            raise RuntimeError('Run an external adapter rather than the reference SGLang source')
        sys.path.insert(0, str(script.parent))
        sys.argv = [str(script), *args[1:]]
        runpy.run_path(str(script), run_name='__main__')


if __name__ == '__main__':
    main()
