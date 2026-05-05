#!/usr/bin/env python3
"""
Pre-commit demo and acceleration checker for microringlib.

Default behavior:
- Run unit-speed demos.
- Check fast public APIs.
- Skip optional FDE/FDTD demos unless backends are installed and --include-heavy is passed.
- Skip known slow demos unless --run-slow is passed.

Usage:
    python tools/check_acceleration_and_demos.py
    python tools/check_acceleration_and_demos.py --jobs 2 --timeout 300
    python tools/check_acceleration_and_demos.py --run-slow --timeout 900 --jobs 1
    python tools/check_acceleration_and_demos.py --include-heavy --run-slow --timeout 1200 --jobs 1
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import pathlib
import subprocess
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]


SLOW_DEMO_NAMES = {
    "demo1.py",
    "demo3.py",
    "demo4.py",
    "demo5.py",
    "demo6.py",
    "demo_ai_inverse_design_random.py",
    "demo_material_backends.py",
    "demo_ring_modulator_eye.py",
}


@dataclass
class CheckResult:
    name: str
    status: str
    detail: str = ""


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def run_python_file(path: pathlib.Path, timeout: int) -> CheckResult:
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["MRL_FAST_DEMO"] = "1"

    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(PROJECT_ROOT)
        if not existing_pythonpath
        else str(PROJECT_ROOT) + os.pathsep + existing_pythonpath
    )

    cmd = [sys.executable, str(path)]

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )

        rel_name = str(path.relative_to(PROJECT_ROOT))

        if proc.returncode == 0:
            return CheckResult(rel_name, "PASS")

        detail = "\n".join(
            [
                "STDOUT:",
                proc.stdout[-3000:],
                "STDERR:",
                proc.stderr[-3000:],
            ]
        )
        return CheckResult(rel_name, "FAIL", detail)

    except subprocess.TimeoutExpired:
        return CheckResult(
            str(path.relative_to(PROJECT_ROOT)),
            "TIMEOUT",
            f"Timed out after {timeout} seconds.",
        )

    except Exception as exc:
        return CheckResult(
            str(path.relative_to(PROJECT_ROOT)),
            "FAIL",
            f"{type(exc).__name__}: {exc}",
        )


def run_python_files_parallel(
    paths: list[pathlib.Path],
    *,
    timeout: int,
    jobs: int,
) -> list[CheckResult]:
    if not paths:
        return []

    if jobs <= 1:
        return [run_python_file(path, timeout=timeout) for path in paths]

    results: list[CheckResult] = []

    with ThreadPoolExecutor(max_workers=jobs) as executor:
        future_to_path = {
            executor.submit(run_python_file, path, timeout): path
            for path in paths
        }

        for future in as_completed(future_to_path):
            results.append(future.result())

    return sorted(results, key=lambda r: r.name)


def check_import() -> CheckResult:
    try:
        import microringlib as mrl

        path = pathlib.Path(mrl.__file__).resolve()
        return CheckResult("import microringlib", "PASS", str(path))
    except Exception:
        return CheckResult("import microringlib", "FAIL", traceback.format_exc())


def check_fast_apis() -> list[CheckResult]:
    results: list[CheckResult] = []

    try:
        import microringlib as mrl

        required_fast_names = [
            "single_mrr_thru_fast_batch",
            "sfwm_pair_rate_relative_fast",
            "resonance_metrics_fast",
            "compute_resonance_metrics_fast",
            "monte_carlo_resonance_formula_fast",
        ]

        missing = [name for name in required_fast_names if not hasattr(mrl, name)]

        if missing:
            return [
                CheckResult(
                    "fast public API",
                    "FAIL",
                    "Missing: " + ", ".join(missing),
                )
            ]

        wl = np.linspace(1540e-9, 1560e-9, 101)

        fields, powers, t, k = mrl.single_mrr_thru_fast_batch(
            wl,
            10e-6,
            3.476,
            2.0,
            [0.01, 0.02],
        )

        if fields.shape != (2, 101):
            results.append(
                CheckResult(
                    "single_mrr_thru_fast_batch shape",
                    "FAIL",
                    f"fields.shape={fields.shape}, expected (2, 101)",
                )
            )
        else:
            results.append(CheckResult("single_mrr_thru_fast_batch shape", "PASS"))

        if powers.shape != (2, 101):
            results.append(
                CheckResult(
                    "single_mrr_thru_fast_batch power shape",
                    "FAIL",
                    f"powers.shape={powers.shape}, expected (2, 101)",
                )
            )
        elif np.all(np.isfinite(powers)) and np.all(powers <= 1.0 + 1e-8):
            results.append(CheckResult("single_mrr_thru_fast_batch passivity", "PASS"))
        else:
            results.append(
                CheckResult(
                    "single_mrr_thru_fast_batch passivity",
                    "FAIL",
                    f"min={np.nanmin(powers)}, max={np.nanmax(powers)}",
                )
            )

        metrics = mrl.resonance_metrics_fast(
            wl,
            powers[0],
            target_wavelength=1550e-9,
            kind="dips",
        )

        required_metric_keys = [
            "resonance_wavelength",
            "fwhm",
            "fsr",
            "loaded_Q",
            "extinction_ratio_db",
        ]

        if all(key in metrics for key in required_metric_keys):
            results.append(CheckResult("resonance_metrics_fast keys", "PASS"))
        else:
            results.append(
                CheckResult(
                    "resonance_metrics_fast keys",
                    "FAIL",
                    f"keys={sorted(metrics.keys())}",
                )
            )

        P = np.linspace(0, 1e-3, 10)

        r = mrl.sfwm_pair_rate_relative_fast(
            P,
            gamma=2.0,
            loaded_Q=10000,
            ring_radius=10e-6,
        )

        if r.shape != P.shape:
            results.append(
                CheckResult(
                    "sfwm_pair_rate_relative_fast shape",
                    "FAIL",
                    f"shape={r.shape}, expected {P.shape}",
                )
            )
        elif (
            np.all(np.isfinite(r))
            and np.min(r) >= -1e-14
            and np.max(r) <= 1.0 + 1e-12
        ):
            results.append(CheckResult("sfwm_pair_rate_relative_fast normalized", "PASS"))
        else:
            results.append(
                CheckResult(
                    "sfwm_pair_rate_relative_fast normalized",
                    "FAIL",
                    f"min={np.nanmin(r)}, max={np.nanmax(r)}",
                )
            )

        rng = np.random.default_rng(1)
        n_samples = 3.476 + rng.normal(0.0, 1e-4, 1000)
        radius_samples = 10e-6 + rng.normal(0.0, 5e-9, 1000)

        out = mrl.monte_carlo_resonance_formula_fast(
            n_eff_samples=n_samples,
            radius_samples=radius_samples,
            target_wavelength=1550e-9,
            n_g=4.2,
            loaded_Q_nominal=5.0e4,
            extinction_ratio_db_nominal=4.5,
        )

        if (
            "resonance_wavelength_nm" in out
            and "loaded_Q" in out
            and "extinction_ratio_db" in out
            and out["resonance_wavelength_nm"].shape == n_samples.shape
        ):
            results.append(CheckResult("monte_carlo_resonance_formula_fast", "PASS"))
        else:
            results.append(
                CheckResult(
                    "monte_carlo_resonance_formula_fast",
                    "FAIL",
                    f"keys={sorted(out.keys())}",
                )
            )

    except Exception:
        results.append(CheckResult("fast API smoke", "FAIL", traceback.format_exc()))

    return results


def check_optional_solver_imports(
    *,
    modesolverpy_available: bool,
    meep_available: bool,
) -> list[CheckResult]:
    results: list[CheckResult] = []

    if modesolverpy_available:
        results.append(CheckResult("optional FDE backend modesolverpy", "AVAILABLE"))
    else:
        results.append(
            CheckResult(
                "optional FDE backend modesolverpy",
                "SKIP",
                "Install with: pip install modesolverpy",
            )
        )

    if meep_available:
        results.append(CheckResult("optional FDTD backend meep", "AVAILABLE"))
    else:
        results.append(
            CheckResult(
                "optional FDTD backend meep",
                "SKIP",
                "Install with: conda install -c conda-forge pymeep",
            )
        )

    return results


def is_optional_fde_demo(path: pathlib.Path) -> bool:
    lower = path.name.lower()
    return "fde" in lower or "modesolver" in lower


def is_optional_fdtd_demo(path: pathlib.Path) -> bool:
    lower = path.name.lower()
    return "fdtd" in lower or "meep" in lower


def is_slow_demo(path: pathlib.Path) -> bool:
    return path.name in SLOW_DEMO_NAMES


def discover_all_demo_files() -> list[pathlib.Path]:
    """
    Discover demos, preferring examples/demo*.py over duplicated root demo*.py.

    If both exist:
        demo2.py
        examples/demo2.py

    only examples/demo2.py is used.
    """
    root_demos = {
        path.name: path.resolve()
        for path in PROJECT_ROOT.glob("demo*.py")
    }

    example_demos = {
        path.name: path.resolve()
        for path in PROJECT_ROOT.glob("examples/demo*.py")
    }

    merged = dict(root_demos)
    merged.update(example_demos)

    return sorted(set(merged.values()))


def split_demo_files() -> tuple[
    list[pathlib.Path],
    list[pathlib.Path],
    list[pathlib.Path],
    list[pathlib.Path],
]:
    all_demos = discover_all_demo_files()

    fde_demos = [path for path in all_demos if is_optional_fde_demo(path)]
    fdtd_demos = [path for path in all_demos if is_optional_fdtd_demo(path)]
    slow_demos = [path for path in all_demos if is_slow_demo(path)]

    excluded = set(fde_demos) | set(fdtd_demos) | set(slow_demos)

    base_demos = [
        path
        for path in all_demos
        if path not in excluded
    ]

    return base_demos, slow_demos, fde_demos, fdtd_demos


def print_results(results: list[CheckResult]) -> bool:
    hard_fail = False

    for r in results:
        marker = {
            "PASS": "✅",
            "AVAILABLE": "✅",
            "SKIP": "⏭️ ",
            "FAIL": "❌",
            "TIMEOUT": "⏱️ ",
        }.get(r.status, "•")

        print(f"{marker} {r.status:9s} {r.name}")

        if r.detail and r.status in {"FAIL", "TIMEOUT"}:
            print("    " + r.detail.replace("\n", "\n    "))

        if r.status in {"FAIL", "TIMEOUT"}:
            hard_fail = True

    return not hard_fail


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument(
        "--include-heavy",
        action="store_true",
        help="Run optional heavy FDE/FDTD demos when their backends are installed.",
    )
    parser.add_argument(
        "--run-slow",
        action="store_true",
        help="Run known slow demos. By default they are skipped for pre-commit.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=max(1, min(2, (os.cpu_count() or 2) // 2)),
        help="Number of demo subprocesses to run in parallel.",
    )

    args = parser.parse_args()

    modesolverpy_available = has_module("modesolverpy")
    meep_available = has_module("meep")

    results: list[CheckResult] = []

    results.append(check_import())
    results.extend(check_fast_apis())
    results.extend(
        check_optional_solver_imports(
            modesolverpy_available=modesolverpy_available,
            meep_available=meep_available,
        )
    )

    base_demos, slow_demos, fde_demos, fdtd_demos = split_demo_files()

    print("\n=== Base analytical and accelerated demo files ===")
    if not base_demos:
        results.append(CheckResult("base demos", "SKIP", "No base demo*.py files found."))
    else:
        results.extend(
            run_python_files_parallel(
                base_demos,
                timeout=args.timeout,
                jobs=args.jobs,
            )
        )

    print("\n=== Slow demos ===")
    if not slow_demos:
        results.append(CheckResult("slow demos", "SKIP", "No known slow demos found."))
    elif not args.run_slow:
        for demo in slow_demos:
            results.append(
                CheckResult(
                    str(demo.relative_to(PROJECT_ROOT)),
                    "SKIP",
                    "Known slow demo. Use --run-slow to execute.",
                )
            )
    else:
        results.extend(
            run_python_files_parallel(
                slow_demos,
                timeout=args.timeout,
                jobs=1,
            )
        )

    print("\n=== Optional FDE demos ===")
    if not fde_demos:
        results.append(CheckResult("FDE demos", "SKIP", "No FDE/modesolver demo files found."))
    elif not modesolverpy_available:
        for demo in fde_demos:
            results.append(
                CheckResult(
                    str(demo.relative_to(PROJECT_ROOT)),
                    "SKIP",
                    "modesolverpy is not installed.",
                )
            )
    elif not args.include_heavy:
        for demo in fde_demos:
            results.append(
                CheckResult(
                    str(demo.relative_to(PROJECT_ROOT)),
                    "SKIP",
                    "Use --include-heavy to run optional FDE demos.",
                )
            )
    else:
        results.extend(
            run_python_files_parallel(
                fde_demos,
                timeout=args.timeout,
                jobs=max(1, min(args.jobs, 2)),
            )
        )

    print("\n=== Optional FDTD demos ===")
    if not fdtd_demos:
        results.append(CheckResult("FDTD demos", "SKIP", "No FDTD/MEEP demo files found."))
    elif not meep_available:
        for demo in fdtd_demos:
            results.append(
                CheckResult(
                    str(demo.relative_to(PROJECT_ROOT)),
                    "SKIP",
                    "meep is not installed.",
                )
            )
    elif not args.include_heavy:
        for demo in fdtd_demos:
            results.append(
                CheckResult(
                    str(demo.relative_to(PROJECT_ROOT)),
                    "SKIP",
                    "Use --include-heavy to run optional FDTD demos.",
                )
            )
    else:
        results.extend(
            run_python_files_parallel(
                fdtd_demos,
                timeout=args.timeout,
                jobs=1,
            )
        )

    print("\n=== Results ===")
    ok = print_results(results)

    if ok:
        print("\nAll required checks passed. Optional unavailable/heavy/slow demos were skipped cleanly.")
        return 0

    print("\nSome required checks failed. Fix failures before committing.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())