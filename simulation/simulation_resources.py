#!/usr/bin/env python3
import os
import sys
import time
import re
import subprocess
from pathlib import Path
from typing import Optional

# --- repo root on sys.path ---
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

# ============================
# Error classification helpers
# ============================

RECOVERABLE_SUBSTR = (
    "too many resources requested",
    "out of memory",
    "resource allocation failed",
    "illegal address",
    "invalid configuration",
    "worker failed to finish",
    ".vxr",
    "cudaerrormemoryallocation",
)

FATAL_SUBSTR = (
    "no cuda-capable device is detected",
    "error: no gpu found",
    "device lost",
    "driver shutting down",
    "device has fallen off the bus",
    "xid",
    "cuinit(0) failed",
    "unspecified launch failure",
)

def classify_text(s: str) -> str:
    s = (s or "").lower()
    if any(x in s for x in FATAL_SUBSTR):
        return "fatal"
    if any(x in s for x in RECOVERABLE_SUBSTR):
        return "recoverable"
    return "ok"

def write_fatal_flag_atomic(path: Path, msg: str) -> None:
    tmp = str(path) + ".tmp"
    Path(tmp).write_text(msg + "\n", encoding="utf-8")
    os.replace(tmp, path)

# ============================
# Report helpers (FORGIVING)
# ============================

def find_report(expected_report_file: Path) -> Optional[Path]:
    """
    Super forgiving:
      - If expected exists, use it
      - Else any *.xml in the folder
      - Else any file containing 'report' in its name
    """
    if expected_report_file.exists():
        return expected_report_file

    parent = expected_report_file.parent
    xmls = list(parent.glob("*.xml"))
    if xmls:
        return xmls[0]

    reportish = [p for p in parent.iterdir() if p.is_file() and "report" in p.name.lower()]
    return reportish[0] if reportish else None

def parse_fitness_from_report(report_file: Path) -> float:
    """
    Very forgiving extraction:
      1) Try a few common fitness tag variants (case-insensitive)
      2) Fallback: first float-like number anywhere in the file
    Only fails if it can't find ANY number at all.
    """
    text = report_file.read_text(encoding="utf-8", errors="ignore")
    if not text.strip():
        raise ValueError("empty report")

    num = r"([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
    patterns = [
        rf"<\s*fitness_score\s*>\s*{num}\s*<\s*/\s*fitness_score\s*>",
        rf"<\s*fitness[_-]score\s*>\s*{num}\s*<\s*/\s*fitness[_-]score\s*>",
        rf"<\s*fitness\s*>\s*{num}\s*<\s*/\s*fitness\s*>",
        rf"<\s*fitnessScore\s*>\s*{num}\s*<\s*/\s*fitnessScore\s*>",
        rf"<\s*score\s*>\s*{num}\s*<\s*/\s*score\s*>",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return float(m.group(1))

    # Nuclear option: first number anywhere
    m = re.search(num, text)
    if m:
        return float(m.group(1))

    raise ValueError("no numeric value found in report")

# ============================
# VoxCraft batch runner (SIMPLE)
# ============================

def simulate_voxcraft_batch(population, args):
    """
    Simple, conservative runner:
      - Serial only (no parallelism)
      - No NVML/admission control
      - Multiple retries per robot
      - Success = report exists AND we can extract a numeric fitness
      - Fatal GPU/runtime errors abort the whole run with a FATAL sentinel
    """
    sim_bin = Path(args.docker_path) / "voxcraft-sim" / "build" / "voxcraft-sim"
    worker_bin = Path(args.docker_path) / "voxcraft-sim" / "build" / "vx3_node_worker"

    if not sim_bin.exists():
        raise FileNotFoundError(f"Simulator binary not found at {sim_bin}")
    if not worker_bin.exists():
        raise FileNotFoundError(f"Worker binary not found at {worker_bin}")

    # ---- Tunables (favor success + correctness) ----
    SIM_TIMEOUT_SEC     = 60        # give robots slack
    MAX_ATTEMPTS        = 2          # retry a bunch
    BACKOFF_BASE_SEC    = 5.0        # attempt 1-> sleep 2s, attempt2->4s, etc.
    CLEAN_BEFORE_RETRY  = True       # delete stale outputs before retrying

    out_path_hist = (
        Path(args.out_path)
        / args.study_name
        / args.experiment_name
        / f"run_{args.run}"
        / "simulations"
    )
    os.makedirs(out_path_hist, exist_ok=True)

    def robot_dir_for(ind):
        return (
            Path(args.out_path)
            / args.study_name
            / args.experiment_name
            / f"run_{args.run}"
            / "robots"
            / f"robot{ind.id}"
        )

    def cleanup_outputs(ind_id: int, history_file: Path, expected_report_file: Path) -> None:
        # Remove per-robot history
        try:
            if history_file.exists():
                history_file.unlink()
        except Exception:
            pass

        # Remove expected report file
        try:
            if expected_report_file.exists():
                expected_report_file.unlink()
        except Exception:
            pass

        # Remove any other xml that looks like it belongs to this robot
        try:
            for p in expected_report_file.parent.glob("*.xml"):
                if p.name.startswith(f"{ind_id}_") or p.name == expected_report_file.name:
                    try:
                        p.unlink()
                    except Exception:
                        pass
        except Exception:
            pass

    def run_one(ind) -> bool:
        """
        Returns True if success (fitness extracted), else False after retries.
        May sys.exit(2) on fatal GPU/driver/runtime death.
        """
        if not getattr(ind, "valid", True):
            return True

        robot_dir = robot_dir_for(ind)
        if not robot_dir.exists():
            print(f"[SIM-SKIP] {ind.id}: robot dir missing: {robot_dir}")
            return True

        base_vxa = robot_dir / "base.vxa"
        vxd_file = robot_dir / f"{ind.id}.vxd"
        if not base_vxa.exists():
            print(f"[SIM-SKIP] {ind.id}: base.vxa missing")
            return True
        if not vxd_file.exists():
            print(f"[SIM-SKIP] {ind.id}: VXD missing")
            return True

        history_file = out_path_hist / f"{ind.id}.history"
        report_file  = out_path_hist / f"{ind.id}_report.xml"

        cmd = [
            str(sim_bin),
            "-l",
            "-w", str(worker_bin),
            "-i", str(robot_dir),
            "-o", str(report_file),
            "-f",
        ]

        for attempt in range(1, MAX_ATTEMPTS + 1):
            if attempt > 1 and CLEAN_BEFORE_RETRY:
                cleanup_outputs(ind.id, history_file, report_file)

            # Run sim, capture stdout to history, stderr in memory
            attempt_t0 = time.perf_counter()

            with open(history_file, "w", encoding="utf-8") as out_f:
                p = subprocess.Popen(
                    cmd,
                    stdout=out_f,
                    stderr=subprocess.PIPE,
                    text=True,
                )

                timed_out = False
                try:
                    stderr = p.communicate(timeout=SIM_TIMEOUT_SEC)[1]
                except subprocess.TimeoutExpired:
                    timed_out = True
                    print(f"[TIMEOUT] {ind.id} attempt {attempt}/{MAX_ATTEMPTS} > {SIM_TIMEOUT_SEC}s, killing.")
                    p.kill()
                    try:
                        stderr = p.communicate(timeout=15)[1]
                    except subprocess.TimeoutExpired:
                        stderr = "<no stderr after kill>"

            # Read history for classification
            try:
                hist_text = history_file.read_text(encoding="utf-8", errors="ignore") if history_file.exists() else ""
            except Exception as e:
                hist_text = f"<history read error: {e}>"

            status_hist = classify_text(hist_text)
            status_err  = classify_text(stderr)

            # Fatal: abort whole run
            if "fatal" in (status_hist, status_err):
                fatal_flag = Path(args.out_path) / args.study_name / "FATAL"
                write_fatal_flag_atomic(fatal_flag, f"CUDA FATAL on robot {ind.id}. See {history_file}")
                tail = "\n".join(hist_text.splitlines()[-120:]) if hist_text else "<no history>"
                print(
                    f"\n[SIM-CRITICAL] {ind.id}: fatal GPU/driver/runtime error.\n"
                    f"----- history tail -----\n{tail}\n----- end tail -----\n",
                    file=sys.stderr, flush=True
                )
                sys.exit(2)

            # Success only if a report exists and we can extract a number
            actual_report = find_report(report_file)
            if actual_report is not None and not timed_out:
                try:
                    ind.displacement = parse_fitness_from_report(actual_report)
                    attempt_dt = time.perf_counter() - attempt_t0
                    print(f"[SIM-OK] {ind.id} attempt {attempt}/{MAX_ATTEMPTS}: {attempt_dt:.2f}s")

                    return True
                except Exception as e:
                    parse_err = f"parse failed: {e}"
            else:
                parse_err = "missing report" if actual_report is None else "timed out"

            # Failed attempt -> retry with backoff (unless last attempt)
            reason_bits = []
            if timed_out:
                reason_bits.append("timeout")
            if p.returncode != 0:
                reason_bits.append(f"rc={p.returncode}")
            if status_err != "ok":
                reason_bits.append(f"stderr={status_err}")
            if status_hist != "ok":
                reason_bits.append(f"hist={status_hist}")
            reason_bits.append(parse_err)

            reason = ", ".join(reason_bits)

            if attempt < MAX_ATTEMPTS:
                sleep_s = BACKOFF_BASE_SEC * attempt
                print(f"[SIM-RETRY] {ind.id} attempt {attempt}/{MAX_ATTEMPTS} failed: {reason} -> sleep {sleep_s:.1f}s")
                time.sleep(sleep_s)
            else:
                print(f"[SIM-FAIL]  {ind.id} failed after {MAX_ATTEMPTS} attempts: {reason}. See {history_file}")
                return False

        return False

    # ---- Serial loop ----
    total = 0
    ok = 0
    failed = 0

    for ind in population:
        if not getattr(ind, "valid", True):
            continue
        total += 1
        if run_one(ind):
            ok += 1
        else:
            failed += 1

    print(f"[SIM-DONE] total={total} ok={ok} failed={failed}")
