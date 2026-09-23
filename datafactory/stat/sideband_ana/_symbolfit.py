"""SymbolFit 本底候选式搜索与白名单表达式模型（叶子模块）。

本模块只依赖 numpy（+ 标准库），绝对不能引入 datafactory 包的其他部分或
任何拟合栈：``_symbolfit_worker.py`` 通过文件路径直接加载本模块，在干净子
进程中运行，避免 Julia 的 LLVM 与父进程 ROOT/libCling 的 LLVM 符号冲突。
"""

from __future__ import annotations

import ast
import os
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np

__all__ = ["BackgroundModel", "_compile_expression", "_run_symbolfit_selection"]


# ---------------------------------------------------------------------------
# 表达式白名单 AST 转换器
# ---------------------------------------------------------------------------
#
# 支持的语法：数字常量、x0、参数符号 a1/a2/...、+ - *、**(非负整数次幂)、
# exp()、square()。每个 ** 的指数必须是非负整数，且底数/指数内部不得再嵌
# 套 **；exp() 的参数必须是 x0 的次数 <= 6 的多项式。其余节点立即报错。


def _validate_powers(tree: ast.AST, formula: str) -> None:
    """检查所有 ** 节点：非负整数指数、无嵌套幂。"""
    for node in ast.walk(tree):
        if not (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow)):
            continue
        for descendant in (*ast.walk(node.left), *ast.walk(node.right)):
            if isinstance(descendant, ast.BinOp) and isinstance(
                descendant.op, ast.Pow
            ):
                raise ValueError(f"表达式含嵌套幂运算，不支持: {formula}")
        exponent = node.right
        if (
            not isinstance(exponent, ast.Constant)
            or isinstance(exponent.value, bool)
            or not isinstance(exponent.value, int)
            or exponent.value < 0
        ):
            raise ValueError(
                f"表达式的幂指数必须是非负整数常量，不支持: {formula}"
            )


def _polynomial_degree_in_mass(node: ast.AST) -> int | None:
    """返回子树作为 x0 多项式的次数；不是多项式时返回 None。"""
    if isinstance(node, ast.Expression):
        return _polynomial_degree_in_mass(node.body)
    if isinstance(node, ast.Constant):
        return 0
    if isinstance(node, ast.Name):
        return 1 if node.id == "x0" else 0
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        return _polynomial_degree_in_mass(node.operand)
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
        left = _polynomial_degree_in_mass(node.left)
        right = _polynomial_degree_in_mass(node.right)
        if left is None or right is None:
            return None
        return max(left, right)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        left = _polynomial_degree_in_mass(node.left)
        right = _polynomial_degree_in_mass(node.right)
        if left is None or right is None:
            return None
        return left + right
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
        base = _polynomial_degree_in_mass(node.left)
        if base is None:
            return None
        return base * node.right.value
    if isinstance(node, ast.Call):
        degree = _polynomial_degree_in_mass(node.args[0])
        if degree is None:
            return None
        return 0 if degree == 0 else None
    return None


def _compile_expression(
    formula: str,
    parameter_names: list[str],
    exp: Callable,
    square: Callable,
) -> Callable:
    """把白名单表达式文本编译为 ``f(mass, params) -> value``。

    ``params`` 是参数名到数值（numpy 或 TensorFlow/zfit 参数均可）的映射；
    ``exp`` / ``square`` 决定后端。遇到不支持的节点立即抛出 ``ValueError``。
    """
    known_parameters = set(parameter_names)

    try:
        tree = ast.parse(formula, mode="eval")
    except SyntaxError as error:
        raise ValueError(f"表达式语法错误: {formula}") from error
    _validate_powers(tree, formula)

    def build(node: ast.AST):
        if isinstance(node, ast.Expression):
            return build(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(
                node.value, (int, float)
            ):
                raise ValueError(f"表达式含非常量操作数: {formula}")
            # 常数叶子必须随 mass 形状广播: 纯常数式 (如 "3.0e7") 或不含 x0 的
            # 参数式会退化为标量, 在二维拟合的 tf.reshape((bins, n_quad)) 处
            # 直接崩溃。value + mass*0 对 numpy 与 TensorFlow 同构且不改变
            # dtype (整数字面量 0 不触发 float32 提升), 标量输入仍返回标量。
            value = node.value
            return lambda mass, params: value + mass * 0
        if isinstance(node, ast.Name):
            if node.id == "x0":
                return lambda mass, params: mass
            if node.id in known_parameters:
                name = node.id
                # 参数叶子同样广播 (见上: 不含 x0 的表达式整体为标量).
                return lambda mass, params: params[name] + mass * 0
            raise ValueError(
                f"表达式含未知符号 {node.id}（既不是 x0 也不是拟合参数）: {formula}"
            )
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            operand = build(node.operand)
            return lambda mass, params: -operand(mass, params)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
            operand = build(node.operand)
            return lambda mass, params: operand(mass, params)
        if isinstance(node, ast.BinOp):
            left = build(node.left)
            right = build(node.right)
            if isinstance(node.op, ast.Add):
                return lambda mass, params: left(mass, params) + right(mass, params)
            if isinstance(node.op, ast.Sub):
                return lambda mass, params: left(mass, params) - right(mass, params)
            if isinstance(node.op, ast.Mult):
                return lambda mass, params: left(mass, params) * right(mass, params)
            if isinstance(node.op, ast.Pow):
                exponent = node.right.value
                return lambda mass, params: left(mass, params) ** exponent
            raise ValueError(
                f"表达式含不支持的二元运算符 {type(node.op).__name__}: {formula}"
            )
        if isinstance(node, ast.Call):
            if (
                not isinstance(node.func, ast.Name)
                or node.func.id not in {"exp", "square"}
            ):
                raise ValueError(f"表达式含不支持的函数调用: {formula}")
            if len(node.args) != 1 or node.keywords:
                raise ValueError(f"exp/square 只接受一个位置参数: {formula}")
            argument = build(node.args[0])
            if node.func.id == "exp":
                degree = _polynomial_degree_in_mass(node.args[0])
                if degree is None or degree > 6:
                    raise ValueError(
                        f"exp() 的参数必须是 x0 的次数 <= 6 的多项式: {formula}"
                    )
                return lambda mass, params: exp(argument(mass, params))
            return lambda mass, params: square(argument(mass, params))
        raise ValueError(
            f"表达式含不支持的 AST 节点 {type(node).__name__}: {formula}"
        )

    return build(tree)


class BackgroundModel:
    """SymbolFit 选出的非负本底解析式 b(m; θ)（密度单位：counts / GeV）。

    ``formula`` 是含参数符号的参数化表达式文本（例如
    ``a1*(x0 - a2)**2 + a3``）；``symbolfit_initial_values`` 记录 SymbolFit
    选出的参数值，作为后续 zfit profiling 的初值。
    """

    def __init__(
        self,
        formula: str,
        parameter_names: list[str],
        symbolfit_initial_values: Mapping[str, float] | None = None,
    ):
        self.formula = str(formula)
        self.parameter_names = [str(name) for name in parameter_names]
        self.symbolfit_initial_values = (
            dict(symbolfit_initial_values) if symbolfit_initial_values else {}
        )
        self._numpy_evaluator = _compile_expression(
            self.formula, self.parameter_names, np.exp, np.square
        )

    def evaluate_density(
        self, mass, parameter_values: Mapping[str, float]
    ) -> np.ndarray:
        """在给定参数下求本底密度 b(m; θ)。"""
        mass_array = np.asarray(mass, dtype=float)
        values = {name: parameter_values[name] for name in self.parameter_names}
        result = np.asarray(
            self._numpy_evaluator(mass_array, values), dtype=float
        )
        if result.shape != mass_array.shape:
            result = np.broadcast_to(result, mass_array.shape).copy()
        return result

    def integrate(
        self,
        mass_lo: float,
        mass_hi: float,
        parameter_values: Mapping[str, float],
        n_points: int = 64,
    ) -> float:
        """Gauss-Legendre 积分 ∫ b(m; θ) dm。"""
        nodes, weights = np.polynomial.legendre.leggauss(n_points)
        points = 0.5 * (mass_lo + mass_hi) + 0.5 * (mass_hi - mass_lo) * nodes
        mapped_weights = 0.5 * (mass_hi - mass_lo) * weights
        values = self.evaluate_density(points, parameter_values)
        return float(np.dot(mapped_weights, values))


# ---------------------------------------------------------------------------
# SymbolFit 候选式搜索（在干净子进程中运行）
# ---------------------------------------------------------------------------


def _run_symbolfit_selection(config: dict) -> dict:
    """运行 SymbolFit 并按 training chi2 + 2*complexity 选择最优本底式。

    在子进程中执行（父进程可能已加载 ROOT/libCling，与 Julia 的 LLVM 冲突）。
    ``config`` 的 key：centers、density、density_errors、training_mask（以上为
    ndarray）、mass_lo、mass_hi、random_seed、output_dir（str | None）。
    返回可 JSON 序列化的 dict（formula / parameter_names / initial_values /
    covariance）。
    """
    from pysr import PySRRegressor
    from symbolfit.symbolfit import SymbolFit

    centers = np.asarray(config["centers"], dtype=float)
    density = np.asarray(config["density"], dtype=float)
    density_errors = np.asarray(config["density_errors"], dtype=float)
    training_mask = np.asarray(config["training_mask"], dtype=bool)
    mass_lo = float(config["mass_lo"])
    mass_hi = float(config["mass_hi"])
    random_seed = int(config["random_seed"])
    output_dir = config.get("output_dir")
    output_dir = Path(output_dir) if output_dir else None

    if np.any(density_errors[training_mask] <= 0.0):
        invalid = centers[training_mask][density_errors[training_mask] <= 0.0]
        raise ValueError(
            f"SymbolFit 训练数据在质量 {invalid.tolist()} 处误差非正"
        )

    x_train = centers[training_mask].reshape(-1, 1)
    y_train = density[training_mask].reshape(-1, 1)
    yerr_train = density_errors[training_mask].reshape(-1, 1)
    pysr_config = PySRRegressor(
        model_selection="accuracy",
        niterations=100,
        maxsize=15,
        binary_operators=["+", "-", "*"],
        unary_operators=["exp", "square(x) = x*x"],
        constraints={"exp": 5},
        nested_constraints={
            "exp": {"exp": 0, "square": 0},
            "square": {"square": 0},
        },
        extra_sympy_mappings={"square": lambda value: value**2},
        elementwise_loss=(
            "loss(y, y_pred, weights) = (y - y_pred)^2 * weights"
        ),
    )
    model = SymbolFit(
        x=x_train,
        y=y_train,
        y_up=yerr_train,
        y_down=yerr_train,
        pysr_config=pysr_config,
        max_complexity=15,
        input_rescale=True,
        scale_y_by="mean",
        max_stderr=20,
        fit_y_unc=True,
        random_seed=random_seed,
    )

    # SymbolFit 会在当前目录写中间文件，切换到工作目录再切回。
    if output_dir is not None:
        work_dir = Path(output_dir) / "symbolfit_work"
        work_dir.mkdir(parents=True, exist_ok=True)
    else:
        work_dir = Path(tempfile.mkdtemp(prefix="symbolfit_"))
    original_dir = os.getcwd()
    try:
        os.chdir(work_dir)
        model.fit()
    finally:
        os.chdir(original_dir)

    if model.func_candidates.empty:
        raise RuntimeError("SymbolFit 没有返回任何本底候选式")

    dense_masses = np.linspace(mass_lo, mass_hi, 2000)
    selection_scores = []
    best = None  # (score, formula, names, initial_values, covariance)
    for _, candidate in model.func_candidates.iterrows():
        formula = candidate["Parameterized equation, unscaled"]
        parameters = candidate["Parameters: (best-fit, +1, -1)"]
        parameter_names = list(parameters)
        best_values = {
            name: float(parameters[name][0]) for name in parameter_names
        }
        training_chi2 = float("nan")
        score = float("inf")
        try:
            background = BackgroundModel(formula, parameter_names, best_values)
            center_prediction = background.evaluate_density(centers, best_values)
            dense_prediction = background.evaluate_density(
                dense_masses, best_values
            )
            if (
                np.all(np.isfinite(center_prediction))
                and np.all(np.isfinite(dense_prediction))
                and np.all(dense_prediction >= 0.0)
            ):
                pulls = (
                    density[training_mask] - center_prediction[training_mask]
                ) / density_errors[training_mask]
                training_chi2 = float(np.sum(np.square(pulls)))
                score = training_chi2 + 2.0 * float(candidate["Complexity"])
        except ValueError:
            score = float("inf")
        selection_scores.append(score)

        if score < float("inf") and (best is None or score < best[0]):
            covariance = np.zeros((len(parameter_names), len(parameter_names)))
            for index, name in enumerate(parameter_names):
                upper_error = abs(float(parameters[name][1]))
                lower_error = abs(float(parameters[name][2]))
                covariance[index, index] = np.square(
                    0.5 * (upper_error + lower_error)
                )
            for name_pair, covariance_value in candidate["Covariance"].items():
                first_name, second_name = (
                    name.strip() for name in name_pair.split(",")
                )
                first_index = parameter_names.index(first_name)
                second_index = parameter_names.index(second_name)
                covariance[first_index, second_index] = covariance_value
                covariance[second_index, first_index] = covariance_value
            best = (score, formula, parameter_names, best_values, covariance)

    if best is None:
        raise RuntimeError(
            "SymbolFit 没有任何有限且非负的本底候选式（白名单校验全部失败）"
        )

    if output_dir is not None:
        model.func_candidates["Background selection score"] = selection_scores
        model.save_to_csv(output_dir=str(output_dir))

    _, formula, parameter_names, best_values, covariance = best
    return {
        "formula": formula,
        "parameter_names": parameter_names,
        "initial_values": best_values,
        "covariance": covariance.tolist(),
    }
