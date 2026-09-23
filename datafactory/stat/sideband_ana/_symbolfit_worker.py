"""SymbolFit 子进程 worker（独立脚本，不经 datafactory 包导入）。

在干净进程中运行 PySR/SymbolFit 候选式搜索，避免 Julia 的 LLVM 与父进程
ROOT/libCling 的 LLVM 发生 ``cl::cl::opt`` 重复注册冲突（进程直接 abort）。
因此本脚本绝对不能 import datafactory（包 ``__init__`` 会加载 ROOT）——
``_symbolfit.py`` 通过文件路径以 importlib 直接加载。由 ``fit.py`` 通过

    python <this script> payload.npz result.json

调用，不作为公共 API。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


def _load_symbolfit_module():
    """按文件路径加载叶子模块 ``_symbolfit.py``（不触发包导入）。"""
    module_path = Path(__file__).resolve().parent / "_symbolfit.py"
    spec = importlib.util.spec_from_file_location(
        "_sideband_ana_symbolfit", module_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    if len(argv) != 3:
        print(
            f"用法: python {Path(__file__).name} payload.npz result.json",
            file=sys.stderr,
        )
        return 2
    payload_path, result_path = Path(argv[1]), Path(argv[2])

    data = np.load(payload_path)
    config = json.loads(str(data["config_json"]))
    config["centers"] = data["centers"]
    config["density"] = data["density"]
    config["density_errors"] = data["density_errors"]
    config["training_mask"] = data["training_mask"]

    symbolfit = _load_symbolfit_module()
    try:
        result = symbolfit._run_symbolfit_selection(config)
        result["ok"] = True
    except Exception as error:  # noqa: BLE001 - 错误信息回传父进程
        result = {
            "ok": False,
            "error": f"{type(error).__name__}: {error}",
        }
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
