
#!/usr/bin/env python3
import os
import sys
import time
import subprocess
from pathlib import Path
from typing import Optional

# --- repo root on sys.path ---
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

# ============================
# Error classification helpers
# ============================

# Per-robot issues to SKIP (continue the run)
RECOVERABLE_SUBSTR = (
    "too many resources requested",
    "out of memory",
    "resource allocation failed",
    "illegal address",
    "invalid configuration",
    "worker failed to finish",
    ".vxr",                        # missing worker output file
    "cudaerrormemoryallocation",   # sometimes appears tokenized
)

# True GPU/driver/runtime death (ABORT run, write sentinel)
FATAL_SUBSTR = (
    "no cuda-capable device is detected",
    "error: no gpu found",
    "device lost",
    "driver shutting down",
    "device has fallen off the bus",
    "xid",                         # NVIDIA driver Xid errors
    "cuinit(0) failed",
    # keep this last & be cautious; rely on your logs context if needed
    "unspecified launch failure",
)

def classify_text(s: str) -> str:
    """Return 'fatal', 'recoverable', or 'ok' based on log text."""
    s = (s or "").lower()
    if any(x in s for x in FATAL_SUBSTR):
        return "fatal"
    if any(x in s for x in RECOVERABLE_SUBSTR):
        return "recoverable"
    return "ok"

def write_fatal_flag_atomic(path: Path, msg: str) -> None:
    """Atomically write the FATAL sentinel so bash never sees a half-written file."""
    tmp = str(path) + ".tmp"
    Path(tmp).write_text(msg + "\n", encoding="utf-8")
    os.replace(tmp, path)

# ============================
# NVML (admission) helpers
# ============================

_NVML_READY = False
def _nvml_init_once():
    global _NVML_READY
    if _NVML_READY:
        return True
    try:
        import pynvml  # type: ignore
        pynvml.nvmlInit()
        _NVML_READY = True
    except Exception:
        _NVML_READY = False
    return _NVML_READY

def get_free_vram_bytes(gpu_index: int = 0) -> Optional[int]:
    """
    Return free VRAM bytes via NVML, or None if NVML unavailable.
    """
    if not _nvml_init_once():
        return None
    try:
        import pynvml  # type: ignore
        h = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        info = pynvml.nvmlDeviceGetMemoryInfo(h)
        return int(info.free)
    except Exception:
        return None

def admit_or_defer(min_free_bytes: int, trials: int, sleep_sec: float) -> bool:
    """
    Admission loop: return True if free VRAM >= threshold within N trials,
    else False (defer).
    If NVML is absent, always return True (don't block).
    """
    free0 = get_free_vram_bytes()
    if free0 is None:
        return True  # can't check; don't block launches
    for _ in range(trials):
        free_now = get_free_vram_bytes()
        if free_now is None or free_now >= min_free_bytes:
            return True
        time.sleep(sleep_sec)
    return False

# ============================
# Report helpers
# ============================

from glob import glob

def find_report(report_file: Path) -> Optional[Path]:
    """
    Return the expected report_file if it exists; else any *.xml in the same folder;
    else None.
    """
    if report_file.exists():
        return report_file
    cands = list(report_file.parent.glob("*.xml"))
    return cands[0] if cands else None

def parse_fitness_from_report(report_file: Path) -> float:
    """
    String-parse the first <fitness_score>...</fitness_score> from the xml-like file.
    """
    if not report_file.exists():
        raise FileNotFoundError(f"Report file not found: {report_file}")
    text = report_file.read_text(encoding="utf-8", errors="ignore")
    start_tag = "<fitness_score>"
    end_tag = "</fitness_score>"
    i = text.find(start_tag)
    if i == -1:
        raise ValueError(f"No {start_tag} in {report_file}")
    i += len(start_tag)
    j = text.find(end_tag, i)
    if j == -1:
        raise ValueError(f"No {end_tag} in {report_file}")
    return float(text[i:j].strip())

def explain_classification(ind_id, rc, timed_out, status_hist, status_err, report_exists, parse_ok):
    if timed_out:
        return "timeout"
    if status_err == "fatal" or status_hist == "fatal":
        return "fatal token in stderr/history"
    if status_err == "recoverable":
        return "recoverable token in stderr"
    if status_hist == "recoverable":
        return "recoverable token in history"
    if not report_exists:
        return "missing report"
    if not parse_ok:
        return "report parse failed"
    if rc != 0:
        return f"non-zero return code {rc}"
    return "unknown"

# ============================
# VoxCraft batch runner
# ============================

def simulate_voxcraft_batch(population, args):
    """
    Run voxcraft-sim with admission control + retries.

    Policy:
      * If report parses -> ACCEPT (even if rc!=0 or 'recoverable' noise).
      * Else if rc==0 and no error tokens -> ACCEPT (warn).
      * Recoverables (OOM/resources/worker/timeout/rc!=0 w/out report) -> skip / retry.
      * Fatal tokens (device lost / no GPU / runtime dead) -> write FATAL + exit(2).
    """
    sim_bin = Path(args.docker_path) / "voxcraft-sim" / "build" / "voxcraft-sim"
    worker_bin = Path(args.docker_path) / "voxcraft-sim" / "build" / "vx3_node_worker"

    if not sim_bin.exists():
        raise FileNotFoundError(f"Simulator binary not found at {sim_bin}")
    if not worker_bin.exists():
        raise FileNotFoundError(f"Worker binary not found at {worker_bin}")

    # ===== Tunables =====
    MAX_PARALLEL_FIRST   = 2      # first pass local concurrency
    MAX_PARALLEL_RETRY   = 1      # retry passes run serially by default
    SIM_TIMEOUT          = 180    # per-robot seconds (more forgiving)

    # Admission threshold & trials
    ADMIT_MIN_FREE_GB    = 3.0    # require at least this much free VRAM to start a robot
    ADMIT_TRIALS         = 5      # N tries per robot before deferring
    ADMIT_SLEEP_SEC      = 2.0    # wait between trials

    # Retry policy
    RETRY_ROUNDS         = 2      # how many retry passes after the first
    RETRY_RECOVERABLES   = True   # push recoverables into retry queue

    min_free_bytes = int(ADMIT_MIN_FREE_GB * (1024**3))

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

    def _run_pass(individuals, max_parallel, round_name):
        """
        Launch + wait pass with admission. Returns a list of individuals that should
        be retried (deferred or recoverable).
        """
        procs = []          # (ind, Popen, fh, hist_path, report_path, launch_note)
        retry_queue = []    # to be processed in later rounds
        errors = []

        # ---- Launch phase ----
        for ind in individuals:
            if not getattr(ind, "valid", True):
                continue

            robot_dir = robot_dir_for(ind)
            if not robot_dir.exists():
                print(f"[{round_name}][SIM-SKIP] {ind.id}: robot dir missing: {robot_dir}")
                continue

            base_vxa = robot_dir / "base.vxa"
            vxd_file = robot_dir / f"{ind.id}.vxd"
            if not base_vxa.exists():
                print(f"[{round_name}][SIM-SKIP] {ind.id}: base_vxa missing")
                continue
            if not vxd_file.exists():
                print(f"[{round_name}][SIM-SKIP] {ind.id}: VXD missing")
                continue

            history_file = out_path_hist / f"{ind.id}.history"
            report_file  = out_path_hist / f"{ind.id}_report.xml"

            # throttle to max_parallel
            while True:
                running = [p for _, p, _, _, _, _ in procs if p.poll() is None]
                if len(running) < max_parallel:
                    break
                time.sleep(0.05)

            # Admission control
            admitted = admit_or_defer(min_free_bytes, ADMIT_TRIALS, ADMIT_SLEEP_SEC)
            if not admitted:
                print(f"[{round_name}][ADMIT-DEFER] {ind.id}: low free VRAM; deferring to retry pass.")
                retry_queue.append(ind)
                continue

            cmd = [
                str(sim_bin),
                "-l",
                "-w", str(worker_bin),
                "-i", str(robot_dir),
                "-o", str(report_file),
                "-f",
            ]

            launch_free = get_free_vram_bytes()
            launch_note = f"{launch_free}B free VRAM at launch" if launch_free is not None else "NVML unavailable"

            out_f = open(history_file, "w")
            p = subprocess.Popen(
                cmd,
                stdout=out_f,
                stderr=subprocess.PIPE,
                text=True,
            )
            procs.append((ind, p, out_f, history_file, report_file, launch_note))

        # ---- Wait & collect ----
        for ind, p, out_f, history_file, report_file, launch_note in procs:
            timed_out = False
            try:
                stderr = p.communicate(timeout=SIM_TIMEOUT)[1]
            except subprocess.TimeoutExpired:
                timed_out = True
                print(f"[{round_name}][TIMEOUT] {ind.id} > {SIM_TIMEOUT}s, killing.")
                p.kill()
                try:
                    stderr = p.communicate(timeout=5)[1]
                except subprocess.TimeoutExpired:
                    stderr = "<no stderr after kill>"
            finally:
                try:
                    out_f.close()
                except Exception:
                    pass

            # read history for classification
            try:
                hist_text = history_file.read_text(encoding="utf-8", errors="ignore") if history_file.exists() else ""
            except Exception as e:
                hist_text = f"<history read error: {e}>"

            status_hist = classify_text(hist_text)
            status_err  = classify_text(stderr)

            # Fatal wins immediately
            if "fatal" in (status_hist, status_err):
                fatal_flag = Path(args.out_path) / args.study_name / "FATAL"
                write_fatal_flag_atomic(fatal_flag, f"CUDA FATAL on robot {ind.id}. See {history_file}")
                tail = "\n".join(hist_text.splitlines()[-120:]) if hist_text else "<no history>"
                print(
                    f"\n[{round_name}][SIM-CRITICAL] {ind.id}: fatal GPU/driver/runtime error.\n"
                    f"----- history tail -----\n{tail}\n----- end tail -----\n",
                    file=sys.stderr, flush=True
                )
                sys.exit(2)

            # Try to use the actual report (ACCEPT even with rc!=0 or recoverable noise)
            actual_report = find_report(report_file)
            report_exists = actual_report is not None
            parse_ok = False
            if report_exists and not timed_out:
                try:
                    displacement = parse_fitness_from_report(actual_report)
                    ind.displacement = displacement
                    parse_ok = True
                except Exception:
                    parse_ok = False

            if parse_ok:
                notes = []
                if p.returncode != 0: notes.append(f"rc={p.returncode}")
                if status_hist == "recoverable": notes.append("hist=recoverable")
                if status_err  == "recoverable": notes.append("stderr=recoverable")
                # if notes:
                #     print(f"[{round_name}][SIM-OK-WARN] {ind.id}: report parsed with warnings ({', '.join(notes)}); {launch_note}")
                # else:
                #     print(f"[{round_name}][SIM-OK] {ind.id}: report parsed; {launch_note}")
                # continue

            # If no report parse, ACCEPT when rc==0 and no error tokens
            if p.returncode == 0 and status_hist == "ok" and status_err == "ok" and not timed_out:
               # print(f"[{round_name}][SIM-OK-WARN] {ind.id}: no/invalid report but rc==0 & no error tokens; accepting; {launch_note}")
                # Optional: provide a default behavior if needed:
                # ind.displacement = 0.0
                continue

            # Otherwise treat as recoverable (give clear reason)
            reason = explain_classification(ind.id, p.returncode, timed_out, status_hist, status_err, report_exists, parse_ok)
            print(f"[{round_name}][SIM-RECOVERABLE] {ind.id}: {reason}. See {history_file} | {launch_note}")
            if RETRY_RECOVERABLES:
                retry_queue.append(ind)
            else:
                errors.append(f"[RECOVERABLE] {ind.id}: {reason}")

        return retry_queue

    # ===== First pass =====
    initial_list = [ind for ind in population if getattr(ind, "valid", True)]
    retry_queue = _run_pass(initial_list, MAX_PARALLEL_FIRST, "PASS1")

    # ===== Retry passes =====
    for r in range(1, RETRY_ROUNDS + 1):
        if not retry_queue:
            break
        print(f"[RETRY] Round {r} starting with {len(retry_queue)} robots...")
        retry_queue = _run_pass(retry_queue, MAX_PARALLEL_RETRY, f"RETRY{r}")

    if retry_queue:
        print(f"[RETRY] Unresolved after {RETRY_ROUNDS} rounds: {len(retry_queue)} robots (skipped).")
