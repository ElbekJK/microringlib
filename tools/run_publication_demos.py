#!/usr/bin/env python3
"""
Run recommended microringlib publication demos, collect figures, and save logs.

Default publication set:
    1. Critical-coupling metrics
    2. 8-channel WDM filter bank with explicit spacing
    3. FDE modesolverpy -> microringlib core
    4. FDTD MEEP -> microringlib core comparison
    5. Kerr bistability
    6. SiC SFWM photon-pair source

Usage:
    python tools/run_publication_demos.py --clean

Optional:
    python tools/run_publication_demos.py --include-supplement
    python tools/run_publication_demos.py --include-all
    python tools/run_publication_demos.py --skip-heavy
    python tools/run_publication_demos.py --clean --include-all

Outputs:
    publication_figures/
        collected figures, prefixed by demo name

    publication_logs/
        one .log file per demo
        run_summary.txt
        run_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import os
import pathlib
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
EXAMPLES_DIR = PROJECT_ROOT / "examples"
PUBFIG_DIR = PROJECT_ROOT / "publication_figures"
PUBLOG_DIR = PROJECT_ROOT / "publication_logs"


# ---------------------------------------------------------------------
# Recommended main publication set
# ---------------------------------------------------------------------

MAIN_PUBLICATION_DEMOS = [
    # 1. Core physics / metrics
    "demo2_critical_coupling_metrics.py",

    # 2. System-level WDM design
    "demo_wdm_8ch_filter_bank_with_spacing.py",

    # 3. FDE-calibrated analytical model
    "demo_fde_modesolverpy_microring_core.py",

    # 4. FDTD validation/comparison
    "demo_fdtd_meep_vs_microring_core.py",

    # 5. Nonlinear Kerr physics
    "demo_kerr_bistability.py",

    # 6. Quantum/nonlinear SFWM use case
    "demo_sic_sfwm_photon_pairs.py",
]


# ---------------------------------------------------------------------
# Supplementary demos
# ---------------------------------------------------------------------

SUPPLEMENTARY_DEMOS = [
    "demo_monte_carlo_tolerance.py",
    "demo_ai_inverse_design_random.py",
    "demo_ring_modulator_eye.py",
    "demo_material_backends.py",
    "demo_frequency_comb_toy.py",
    "demo2.py",
    "demo_wdm_8ch_filter_bank.py",
]


# ---------------------------------------------------------------------
# All remaining legacy/slow examples
# ---------------------------------------------------------------------

EXTRA_DEMOS = [
    "demo1.py",
    "demo3.py",
    "demo4.py",
    "demo5.py",
    "demo6.py",
    "demo_fde_modesolverpy_waveguide.py",
    "demo_fdtd_meep_straight_waveguide.py",
    "demo_fdtd_meep_ring.py",
]


HEAVY_DEMOS = {
    "demo_fde_modesolverpy_microring_core.py",
    "demo_fdtd_meep_vs_microring_core.py",
    "demo_fde_modesolverpy_waveguide.py",
    "demo_fdtd_meep_straight_waveguide.py",
    "demo_fdtd_meep_ring.py",
}

SLOW_DEMOS = {
    "demo1.py",
    "demo3.py",
    "demo4.py",
    "demo5.py",
    "demo6.py",
    "demo_ai_inverse_design_random.py",
    "demo_ring_modulator_eye.py",
    "demo_material_backends.py",
}


FIG_EXTENSIONS = {
    ".png",
    ".pdf",
    ".svg",
    ".jpg",
    ".jpeg",
}


@dataclass
class DemoResult:
    demo: str
    status: str
    returncode: int
    stdout: str
    stderr: str
    elapsed_s: float
    log_path: pathlib.Path | None = None


def safe_stem(name: str) -> str:
    return pathlib.Path(name).stem.replace(" ", "_")


def now_iso() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def demo_timeout(demo_name: str, args: argparse.Namespace) -> int:
    if demo_name in HEAVY_DEMOS:
        return args.timeout_heavy
    if demo_name in SLOW_DEMOS:
        return args.timeout_slow
    return args.timeout_fast


def build_demo_plan(args: argparse.Namespace) -> list[str]:
    demos: list[str] = []

    demos.extend(MAIN_PUBLICATION_DEMOS)

    if args.include_supplement or args.include_all:
        demos.extend(SUPPLEMENTARY_DEMOS)

    if args.include_all:
        demos.extend(EXTRA_DEMOS)

    if args.skip_heavy:
        demos = [demo for demo in demos if demo not in HEAVY_DEMOS]

    # Preserve order while removing duplicates.
    seen = set()
    unique_demos = []
    for demo in demos:
        if demo not in seen:
            unique_demos.append(demo)
            seen.add(demo)

    return unique_demos


def snapshot_figures() -> dict[pathlib.Path, float]:
    files: dict[pathlib.Path, float] = {}

    search_dirs = [
        PROJECT_ROOT,
        PROJECT_ROOT / "figures",
        EXAMPLES_DIR,
    ]

    excluded_dirs = {
        PUBFIG_DIR.resolve(),
        PUBLOG_DIR.resolve(),
    }

    for root in search_dirs:
        if not root.exists():
            continue

        for path in root.rglob("*"):
            if not path.is_file():
                continue

            resolved = path.resolve()

            if any(str(resolved).startswith(str(excluded)) for excluded in excluded_dirs):
                continue

            if path.suffix.lower() in FIG_EXTENSIONS:
                files[resolved] = path.stat().st_mtime

    return files


def collect_new_figures(before: dict[pathlib.Path, float], demo_name: str) -> list[pathlib.Path]:
    after = snapshot_figures()

    changed_files: list[pathlib.Path] = []

    for path, mtime_after in after.items():
        mtime_before = before.get(path)

        # Collect if new or modified.
        if mtime_before is None or mtime_after > mtime_before + 1e-9:
            changed_files.append(path)

    new_files = sorted(changed_files)

    PUBFIG_DIR.mkdir(exist_ok=True)

    collected: list[pathlib.Path] = []
    stem = safe_stem(demo_name)

    for src in new_files:
        dst = PUBFIG_DIR / f"{stem}__{src.name}"

        if dst.exists():
            base = dst.with_suffix("")
            suffix = dst.suffix
            i = 2
            while True:
                candidate = pathlib.Path(f"{base}_{i}{suffix}")
                if not candidate.exists():
                    dst = candidate
                    break
                i += 1

        shutil.copy2(src, dst)
        collected.append(dst)

    return collected


def write_demo_log(
    *,
    result: DemoResult,
    demo_path: pathlib.Path,
    timeout: int,
    figures: list[pathlib.Path],
    command: list[str],
) -> pathlib.Path:
    PUBLOG_DIR.mkdir(exist_ok=True)

    log_path = PUBLOG_DIR / f"{safe_stem(result.demo)}.log"

    with log_path.open("w", encoding="utf-8") as f:
        f.write("=" * 88 + "\n")
        f.write(f"Demo: {result.demo}\n")
        f.write(f"Timestamp: {now_iso()}\n")
        f.write(f"Status: {result.status}\n")
        f.write(f"Return code: {result.returncode}\n")
        f.write(f"Elapsed seconds: {result.elapsed_s:.3f}\n")
        f.write(f"Timeout seconds: {timeout}\n")
        f.write(f"Project root: {PROJECT_ROOT}\n")
        f.write(f"Demo path: {demo_path}\n")
        f.write(f"Command: {' '.join(command)}\n")
        f.write(f"Python executable: {sys.executable}\n")
        f.write("=" * 88 + "\n\n")

        f.write("Collected figures:\n")
        if figures:
            for fig in figures:
                try:
                    f.write(f"  - {fig.relative_to(PROJECT_ROOT)}\n")
                except ValueError:
                    f.write(f"  - {fig}\n")
        else:
            f.write("  - none\n")

        f.write("\n" + "=" * 88 + "\n")
        f.write("STDOUT\n")
        f.write("=" * 88 + "\n")
        f.write(result.stdout or "")
        if result.stdout and not result.stdout.endswith("\n"):
            f.write("\n")

        f.write("\n" + "=" * 88 + "\n")
        f.write("STDERR\n")
        f.write("=" * 88 + "\n")
        f.write(result.stderr or "")
        if result.stderr and not result.stderr.endswith("\n"):
            f.write("\n")

    return log_path


def run_demo(demo_name: str, timeout: int) -> tuple[DemoResult, list[pathlib.Path]]:
    demo_path = EXAMPLES_DIR / demo_name

    command = [sys.executable, str(demo_path)]

    if not demo_path.exists():
        result = DemoResult(
            demo=demo_name,
            status="MISSING",
            returncode=127,
            stdout="",
            stderr=f"{demo_path} does not exist",
            elapsed_s=0.0,
        )
        log_path = write_demo_log(
            result=result,
            demo_path=demo_path,
            timeout=timeout,
            figures=[],
            command=command,
        )
        result.log_path = log_path
        return result, []

    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["MRL_FAST_DEMO"] = "1"

    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(PROJECT_ROOT)
        if not existing_pythonpath
        else str(PROJECT_ROOT) + os.pathsep + existing_pythonpath
    )

    before = snapshot_figures()
    t0 = time.perf_counter()

    try:
        proc = subprocess.run(
            command,
            cwd=str(PROJECT_ROOT),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )

        elapsed_s = time.perf_counter() - t0
        status = "PASS" if proc.returncode == 0 else "FAIL"

        result = DemoResult(
            demo=demo_name,
            status=status,
            returncode=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            elapsed_s=elapsed_s,
        )

    except subprocess.TimeoutExpired as exc:
        elapsed_s = time.perf_counter() - t0

        stdout = exc.stdout or ""
        stderr = exc.stderr or ""

        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")

        stderr = stderr + f"\nTimed out after {timeout} seconds."

        result = DemoResult(
            demo=demo_name,
            status="TIMEOUT",
            returncode=124,
            stdout=stdout,
            stderr=stderr,
            elapsed_s=elapsed_s,
        )

    collected = collect_new_figures(before, demo_name)

    log_path = write_demo_log(
        result=result,
        demo_path=demo_path,
        timeout=timeout,
        figures=collected,
        command=command,
    )
    result.log_path = log_path

    return result, collected


def print_result(result: DemoResult, figures: list[pathlib.Path]) -> None:
    marker = {
        "PASS": "✅",
        "FAIL": "❌",
        "TIMEOUT": "⏱️ ",
        "MISSING": "⚠️ ",
    }.get(result.status, "•")

    print(f"{marker} {result.status:8s} {result.demo} ({result.elapsed_s:.2f} s)")

    if result.log_path is not None:
        print(f"    log: {result.log_path.relative_to(PROJECT_ROOT)}")

    if figures:
        print(f"    collected {len(figures)} figure(s):")
        for fig in figures:
            print(f"      - {fig.relative_to(PROJECT_ROOT)}")

    if result.status != "PASS":
        if result.stdout:
            print("    STDOUT tail:")
            print("    " + result.stdout[-1500:].replace("\n", "\n    "))
        if result.stderr:
            print("    STDERR tail:")
            print("    " + result.stderr[-1500:].replace("\n", "\n    "))


def write_run_summary(
    *,
    results: list[DemoResult],
    figure_map: dict[str, list[pathlib.Path]],
    args: argparse.Namespace,
    demo_plan: list[str],
) -> None:
    PUBLOG_DIR.mkdir(exist_ok=True)

    txt_path = PUBLOG_DIR / "run_summary.txt"
    csv_path = PUBLOG_DIR / "run_summary.csv"

    with txt_path.open("w", encoding="utf-8") as f:
        f.write("=" * 88 + "\n")
        f.write("microringlib publication demo run summary\n")
        f.write("=" * 88 + "\n")
        f.write(f"Timestamp: {now_iso()}\n")
        f.write(f"Project root: {PROJECT_ROOT}\n")
        f.write(f"Publication figure folder: {PUBFIG_DIR}\n")
        f.write(f"Publication log folder: {PUBLOG_DIR}\n")
        f.write(f"Python executable: {sys.executable}\n")
        f.write(f"include_supplement: {args.include_supplement}\n")
        f.write(f"include_all: {args.include_all}\n")
        f.write(f"skip_heavy: {args.skip_heavy}\n")
        f.write(f"timeout_fast: {args.timeout_fast}\n")
        f.write(f"timeout_slow: {args.timeout_slow}\n")
        f.write(f"timeout_heavy: {args.timeout_heavy}\n")
        f.write("\nSelected demos:\n")
        for demo in demo_plan:
            f.write(f"  - {demo}\n")
        f.write("\n")

        for result in results:
            figs = figure_map.get(result.demo, [])
            f.write("-" * 88 + "\n")
            f.write(f"Demo: {result.demo}\n")
            f.write(f"Status: {result.status}\n")
            f.write(f"Return code: {result.returncode}\n")
            f.write(f"Elapsed seconds: {result.elapsed_s:.3f}\n")
            if result.log_path is not None:
                f.write(f"Log: {result.log_path.relative_to(PROJECT_ROOT)}\n")
            f.write(f"Figures collected: {len(figs)}\n")
            for fig in figs:
                f.write(f"  - {fig.relative_to(PROJECT_ROOT)}\n")

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "demo",
                "status",
                "returncode",
                "elapsed_s",
                "log_path",
                "n_figures",
                "figures",
            ]
        )

        for result in results:
            figs = figure_map.get(result.demo, [])
            writer.writerow(
                [
                    result.demo,
                    result.status,
                    result.returncode,
                    f"{result.elapsed_s:.3f}",
                    str(result.log_path.relative_to(PROJECT_ROOT)) if result.log_path else "",
                    len(figs),
                    ";".join(str(fig.relative_to(PROJECT_ROOT)) for fig in figs),
                ]
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--include-supplement",
        action="store_true",
        help="Also run selected supplementary demos.",
    )
    parser.add_argument(
        "--include-all",
        action="store_true",
        help="Run main, supplementary, and extra demos.",
    )
    parser.add_argument(
        "--skip-heavy",
        action="store_true",
        help="Skip FDE/FDTD heavy demos even if they are in the selected set.",
    )
    parser.add_argument("--timeout-fast", type=int, default=300)
    parser.add_argument("--timeout-slow", type=int, default=1800)
    parser.add_argument("--timeout-heavy", type=int, default=2400)
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete publication_figures and publication_logs before running.",
    )

    args = parser.parse_args()

    if args.clean:
        if PUBFIG_DIR.exists():
            shutil.rmtree(PUBFIG_DIR)
        if PUBLOG_DIR.exists():
            shutil.rmtree(PUBLOG_DIR)

    PUBFIG_DIR.mkdir(exist_ok=True)
    PUBLOG_DIR.mkdir(exist_ok=True)

    demo_plan = build_demo_plan(args)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Publication figure folder: {PUBFIG_DIR}")
    print(f"Publication log folder: {PUBLOG_DIR}")
    print(f"Total demos planned: {len(demo_plan)}")
    print("Selected demos:")
    for demo in demo_plan:
        print(f"  - {demo}")
    print()

    failures = 0
    results: list[DemoResult] = []
    figure_map: dict[str, list[pathlib.Path]] = {}

    for demo in demo_plan:
        timeout = demo_timeout(demo, args)

        result, figures = run_demo(demo, timeout=timeout)

        results.append(result)
        figure_map[demo] = figures

        print_result(result, figures)

        if result.status not in {"PASS"}:
            failures += 1

        print()

    all_figs = sorted(PUBFIG_DIR.glob("*"))
    all_logs = sorted(PUBLOG_DIR.glob("*.log"))

    write_run_summary(
        results=results,
        figure_map=figure_map,
        args=args,
        demo_plan=demo_plan,
    )

    print("=" * 88)
    print(f"Collected publication figures: {len(all_figs)}")
    for fig in all_figs:
        print(f"  - {fig.relative_to(PROJECT_ROOT)}")

    print()
    print(f"Saved demo logs: {len(all_logs)}")
    for log in all_logs:
        print(f"  - {log.relative_to(PROJECT_ROOT)}")

    print()
    print("Summary files:")
    print(f"  - {(PUBLOG_DIR / 'run_summary.txt').relative_to(PROJECT_ROOT)}")
    print(f"  - {(PUBLOG_DIR / 'run_summary.csv').relative_to(PROJECT_ROOT)}")

    if failures:
        print(f"\nCompleted with {failures} failed/timed-out/missing demo(s).")
        return 1

    print("\nAll selected publication demos completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())