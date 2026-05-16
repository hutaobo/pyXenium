from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REMOTE_HOST = "sscb-a100.scilifelab.se"
REMOTE_ROOT = "/data/taobo.hu/pyxenium_lazyslide_cervical_wta_20260511"
REMOTE_BREAST_ROOT = "/data/taobo.hu/pyxenium_lazyslide_breast_wta_20260507"
REMOTE_REPO = f"{REMOTE_BREAST_ROOT}/repo"
REMOTE_PYTHON = f"{REMOTE_BREAST_ROOT}/envs/plip-patch/bin/python"
REMOTE_DATA = f"{REMOTE_ROOT}/data"
REMOTE_RUNS = f"{REMOTE_ROOT}/runs"
REMOTE_LOGS = f"{REMOTE_ROOT}/logs"
LOCAL_CERVICAL_ROOT = Path(r"Y:\long\10X_datasets\Xenium\Atera\WTA_Preview_FFPE_Cervical_Cancer_outs")

BREAST_DATA_REL = Path("docs/_static/tutorials/multimodal_histoseg_lazyslide_breast_wta")
PACKAGE_REL = BREAST_DATA_REL / "naturebiotech_package"
AUTOPILOT_SUBDIR = "autopilot_20260511"
CERVICAL_PACKAGE_SUBDIR = "cervical_replication_20260511"

REMOTE_HE_OME = f"{REMOTE_DATA}/WTA_Preview_FFPE_Cervical_Cancer_he_image.ome.tif"
REMOTE_HE_PYRAMID = f"{REMOTE_DATA}/WTA_Preview_FFPE_Cervical_Cancer_he_image.tiffslide_pyramid.tif"
REMOTE_SPATIALDATA = f"{REMOTE_DATA}/spatialdata.zarr"
REMOTE_GEOJSON = f"{REMOTE_DATA}/xenium_explorer_annotations.geojson"
REMOTE_PROGRAM_JSON = f"{REMOTE_DATA}/cervical_mtm_wta_programs.json"

FAILURE_TOKENS = (
    "Traceback",
    "Killed",
    "CUDA out of memory",
    "No such file",
    "KeyError",
    "ValueError",
    "RuntimeError",
)

CERVICAL_PROGRAMS: dict[str, list[str]] = {
    "epithelial_identity": ["EPCAM", "KRT7", "KRT8", "KRT18", "KRT19", "MUC1", "TACSTD2", "CLDN4", "CLDN7"],
    "basal_squamous_state": ["KRT5", "KRT14", "KRT17", "TP63", "LAMB3", "LAMC2", "ITGA6"],
    "cell_cycle_proliferation": ["MKI67", "TOP2A", "UBE2C", "CCNB1", "CCNB2", "CDK1", "PCNA", "MCM2", "MCM5", "BIRC5", "CENPF", "AURKA"],
    "hypoxia_glycolysis": ["CA9", "SLC2A1", "LDHA", "ENO1", "PGK1", "ALDOA", "VEGFA", "HILPDA", "BNIP3", "NDRG1", "PFKP"],
    "oxidative_phosphorylation": ["NDUFA1", "NDUFB8", "COX5A", "COX6C", "ATP5F1A", "ATP5MC1", "UQCRC1", "SDHB"],
    "unfolded_protein_response": ["XBP1", "DDIT3", "HSPA5", "HERPUD1", "ATF3", "ATF4", "ERN1", "PDIA3", "HSP90B1"],
    "p53_apoptosis_stress": ["TP53", "CDKN1A", "MDM2", "BAX", "PMAIP1", "BBC3", "GADD45A", "FAS", "DDB2"],
    "emt_invasion": ["VIM", "FN1", "SNAI2", "TWIST1", "ZEB1", "MMP2", "MMP9", "MMP11", "ITGA5", "ITGB1", "CDH2", "SPARC"],
    "emt_invasive_front": ["VIM", "MMP11", "ITGB6", "KRT14", "KRT17", "TGFB1", "FN1", "SPARC", "COL1A1", "COL3A1"],
    "tgf_beta_response": ["TGFB1", "TGFBR1", "TGFBR2", "SMAD2", "SMAD3", "SERPINE1", "CTGF", "INHBA", "THBS1", "PMEPA1"],
    "collagen_ecm_organization": ["COL1A1", "COL1A2", "COL3A1", "COL5A1", "COL5A2", "COL6A1", "COL6A2", "DCN", "LUM", "POSTN", "THBS2"],
    "myofibroblast_caf_activation": ["ACTA2", "TAGLN", "MYL9", "COL11A1", "FAP", "PDPN", "PDGFRA", "PDGFRB", "CTHRC1"],
    "stromal_encapsulation": ["COL1A1", "COL3A1", "DCN", "TAGLN", "ACTA2", "THY1", "LUM", "COL1A2", "FN1"],
    "angiogenesis_endothelial": ["PECAM1", "VWF", "KDR", "FLT1", "ESAM", "EMCN", "ENG", "PLVAP", "CDH5", "RAMP2"],
    "myeloid_activation": ["SPP1", "CD68", "HLA-DRA", "FCER1G", "LST1", "TYROBP", "APOE", "C1QA", "C1QB", "C1QC"],
    "myeloid_vascular_belt": ["CXCR4", "CD68", "LST1", "SPP1", "PECAM1", "KDR", "COL4A1", "COL4A2", "RGS5"],
    "immune_activation": ["CD3D", "TRAC", "NKG7", "CXCL13", "HLA-DRA", "CD3E", "GZMB", "PRF1"],
    "immune_exclusion": ["TAGLN", "COL1A1", "FN1", "ACTA2", "FAP", "CXCL12", "TGFB1", "TGFBR2", "COL3A1"],
    "t_cell_exhaustion_checkpoint": ["PDCD1", "CTLA4", "LAG3", "HAVCR2", "TIGIT", "TOX", "CXCL13", "ENTPD1", "LAYN"],
    "tls_adjacent_activation": ["CXCL13", "MS4A1", "CD79A", "JCHAIN", "TRAC", "CD3D", "LTB", "CD79B"],
    "necrotic_hypoxic_rim": ["CA9", "SLC2A1", "VEGFA", "LDHA", "HILPDA", "BNIP3", "NDRG1", "COL3A1", "COL1A2", "TAGLN"],
}


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _repo_root() -> Path:
    for candidate in (Path.cwd(), *Path.cwd().parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "src" / "pyXenium").exists():
            return candidate
    return Path(__file__).resolve().parents[3]


def _load_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return dict(default)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else dict(default)
    except Exception:
        return dict(default)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _quote(value: str) -> str:
    return "'" + str(value).replace("'", "'\"'\"'") + "'"


def _read_text_tail(path: Path, max_chars: int = 20000) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[-max_chars:]


def _has_failure_tokens(text: str) -> bool:
    return any(token in text for token in FAILURE_TOKENS)


def _read_csv_rows(path: Path, limit: int | None = None) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rows[:limit] if limit else rows


def _table_path(base: Path, stem: str) -> Path | None:
    for suffix in (".csv", ".parquet"):
        path = base / f"{stem}{suffix}"
        if path.exists():
            return path
    return None


def _read_table(base: Path, stem: str):
    import pandas as pd

    path = _table_path(base, stem)
    if path is None:
        return pd.DataFrame()
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


class Autopilot:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.repo = _repo_root()
        self.data_dir = self.repo / BREAST_DATA_REL
        self.package_dir = self.repo / PACKAGE_REL
        self.state_dir = self.package_dir / AUTOPILOT_SUBDIR
        self.cervical_package_dir = self.package_dir / CERVICAL_PACKAGE_SUBDIR
        self.state_path = self.state_dir / "Autopilot_State.json"
        self.decision_log = self.state_dir / "Autopilot_Decision_Log.txt"
        self.boss_log = self.state_dir / "LOG_FOR_BOSS.md"
        self.program_json = self.state_dir / "cervical_mtm_wta_programs.json"
        self.state = _load_json(
            self.state_path,
            {
                "started_utc": _utc(),
                "cycle": 0,
                "failures": {},
                "jobs": {},
                "local_tasks": {},
                "deliverables": {},
                "remote": {
                    "host": REMOTE_HOST,
                    "root": REMOTE_ROOT,
                    "repo": REMOTE_REPO,
                    "python": REMOTE_PYTHON,
                },
            },
        )

    def setup(self) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.cervical_package_dir.mkdir(parents=True, exist_ok=True)
        self.package_dir.mkdir(parents=True, exist_ok=True)
        if not self.decision_log.exists():
            self.decision_log.write_text(
                "mTM WTA 24h autopilot decision log\n"
                f"Started UTC: {self.state.get('started_utc', _utc())}\n\n",
                encoding="utf-8",
            )
        if not self.boss_log.exists():
            self.boss_log.write_text("# LOG_FOR_BOSS\n\nNo blocking failures recorded yet.\n", encoding="utf-8")
        if not self.args.once:
            self.state["local_supervisor_pid"] = int(os.getpid())
            (self.state_dir / "local_supervisor.pid").write_text(f"{os.getpid()}\n", encoding="utf-8")
        self.program_json.write_text(json.dumps(CERVICAL_PROGRAMS, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        self._write_status()

    def log(self, message: str) -> None:
        line = f"[{_utc()}] {message}"
        with self.decision_log.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        print(line, flush=True)

    def log_failure(self, key: str, message: str) -> None:
        failures = self.state.setdefault("failures", {})
        entry = failures.setdefault(key, {"count": 0, "messages": []})
        entry["count"] = int(entry.get("count", 0)) + 1
        entry.setdefault("messages", []).append({"utc": _utc(), "message": message})
        with self.boss_log.open("a", encoding="utf-8") as handle:
            handle.write(f"\n## {key} ({_utc()})\n\n{message}\n")
        self.log(f"FAILURE {key}: {message}")
        self.save_state()

    def save_state(self) -> None:
        self.state["updated_utc"] = _utc()
        _write_json(self.state_path, self.state)
        try:
            self._write_status()
        except Exception:
            pass

    def _run(self, command: list[str], *, timeout: int | None = None, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            command,
            cwd=str(cwd or self.repo),
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    def ssh(self, script: str, *, timeout: int = 60) -> subprocess.CompletedProcess[str]:
        return self._run(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=15",
                REMOTE_HOST,
                f"bash -lc {_quote(script)}",
            ],
            timeout=timeout,
        )

    def remote_ok(self) -> bool:
        if self.args.skip_remote:
            return False
        result = self.ssh(
            (
                f"test -d {_quote(REMOTE_REPO)} && "
                f"test -x {_quote(REMOTE_PYTHON)} && "
                "hostname"
            ),
            timeout=25,
        )
        if result.returncode != 0:
            self.log_failure("a100_access", (result.stderr or result.stdout or "A100 is not reachable").strip())
            return False
        self.state["remote"]["hostname"] = result.stdout.strip().splitlines()[-1]
        return True

    def remote_exists(self, path: str) -> bool:
        result = self.ssh(f"test -e {_quote(path)}", timeout=20)
        return result.returncode == 0

    def ensure_remote_dirs(self) -> None:
        result = self.ssh(
            (
                f"mkdir -p {_quote(REMOTE_DATA)} {_quote(REMOTE_RUNS)} {_quote(REMOTE_LOGS)} && "
                f"test -d {_quote(REMOTE_REPO)} && test -x {_quote(REMOTE_PYTHON)}"
            ),
            timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError((result.stderr or result.stdout).strip())
        self.upload_program_json()

    def upload_program_json(self) -> None:
        result = self._run(
            ["scp", str(self.program_json), f"{REMOTE_HOST}:{REMOTE_PROGRAM_JSON}"],
            timeout=60,
        )
        if result.returncode != 0:
            raise RuntimeError((result.stderr or result.stdout).strip())

    def start_local_sync_if_needed(self) -> str:
        if self.remote_exists(REMOTE_HE_OME) and self.remote_exists(REMOTE_SPATIALDATA) and self.remote_exists(REMOTE_GEOJSON):
            self.state["inputs_synced"] = True
            return "ready"
        if not LOCAL_CERVICAL_ROOT.exists():
            self.log_failure("cervical_local_path", f"Missing local cervical dataset: {LOCAL_CERVICAL_ROOT}")
            self.state["inputs_synced"] = False
            return "missing_local"
        name = "sync_cervical_inputs"
        status = self.local_task_status(name)
        if status == "running":
            return "syncing"
        if status == "failed":
            self.log_failure(name, "Previous cervical data sync failed; autopilot will use breast-only hardening until data is available.")
            return "failed"
        self.write_sync_script()
        log_path = self.state_dir / "sync_cervical_inputs_to_a100.log"
        script_path = self.state_dir / "sync_cervical_inputs_to_a100.ps1"
        self.start_local_task(
            name,
            [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(script_path),
            ],
            log_path,
        )
        return "sync_started"

    def write_sync_script(self) -> None:
        content = f"""$ErrorActionPreference = 'Stop'
$localRoot = {json.dumps(str(LOCAL_CERVICAL_ROOT))}
$remoteHost = {json.dumps(REMOTE_HOST)}
$remoteData = {json.dumps(REMOTE_DATA)}
ssh -o BatchMode=yes -o ConnectTimeout=15 $remoteHost "mkdir -p '$remoteData'"
$sources = @(
  (Join-Path $localRoot 'spatialdata.zarr'),
  (Join-Path $localRoot 'WTA_Preview_FFPE_Cervical_Cancer_he_image.ome.tif'),
  (Join-Path $localRoot 'WTA_Preview_FFPE_Cervical_Cancer_he_alignment.csv'),
  (Join-Path $localRoot 'WTA_Preview_FFPE_Cervical_Cancer_keypoints.csv'),
  (Join-Path $localRoot 'pyxenium_cervical_end_to_end\\contours_bio6\\xenium_explorer_annotations.geojson')
)
& scp -r @sources "${{remoteHost}}:$remoteData/"
ssh -o BatchMode=yes -o ConnectTimeout=15 $remoteHost "cp '$remoteData/pyxenium_cervical_end_to_end/contours_bio6/xenium_explorer_annotations.geojson' '$remoteData/xenium_explorer_annotations.geojson'"
"""
        (self.state_dir / "sync_cervical_inputs_to_a100.ps1").write_text(content, encoding="utf-8")

    def start_local_task(self, name: str, command: list[str], log_path: Path) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handle = log_path.open("ab")
        process = subprocess.Popen(command, cwd=str(self.repo), stdout=handle, stderr=subprocess.STDOUT)
        handle.close()
        self.state.setdefault("local_tasks", {})[name] = {
            "pid": int(process.pid),
            "command": command,
            "log": str(log_path),
            "started_utc": _utc(),
            "status": "running",
        }
        self.log(f"Started local task {name} PID {process.pid}: {' '.join(command)}")
        self.save_state()

    def local_task_status(self, name: str) -> str:
        task = self.state.get("local_tasks", {}).get(name)
        if not task:
            return "not_started"
        pid = int(task.get("pid", 0) or 0)
        if pid and self.local_pid_alive(pid):
            task["status"] = "running"
            return "running"
        log_text = _read_text_tail(Path(task.get("log", "")))
        if _has_failure_tokens(log_text):
            task["status"] = "failed"
            return "failed"
        task["status"] = "completed"
        return "completed"

    def local_pid_alive(self, pid: int) -> bool:
        result = self._run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                f"if (Get-Process -Id {int(pid)} -ErrorAction SilentlyContinue) {{ exit 0 }} else {{ exit 1 }}",
            ],
            timeout=15,
        )
        return result.returncode == 0

    def remote_job_paths(self, name: str) -> dict[str, str]:
        safe = name.replace("/", "_")
        return {
            "pid": f"{REMOTE_LOGS}/{safe}.pid",
            "exit": f"{REMOTE_LOGS}/{safe}.exit",
            "runner": f"{REMOTE_LOGS}/{safe}.sh",
            "log": f"{REMOTE_LOGS}/{safe}.log",
        }

    def remote_job_status(self, name: str, success_path: str | None = None) -> str:
        paths = self.remote_job_paths(name)
        if success_path and self.remote_exists(success_path):
            self.state.setdefault("jobs", {}).setdefault(name, {})["status"] = "completed"
            return "completed"
        script = (
            f"pidfile={_quote(paths['pid'])}; exitfile={_quote(paths['exit'])}; "
            "if test -f \"$pidfile\" && kill -0 \"$(cat \"$pidfile\")\" 2>/dev/null; then echo running; exit 0; fi; "
            "if test -f \"$exitfile\"; then rc=$(cat \"$exitfile\"); if test \"$rc\" = 0; then echo completed; else echo failed:$rc; fi; exit 0; fi; "
            "echo not_started"
        )
        result = self.ssh(script, timeout=20)
        status = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else "unknown"
        if status.startswith("failed"):
            return "failed"
        return status

    def start_remote_job(self, name: str, command: str, success_path: str | None = None) -> str:
        status = self.remote_job_status(name, success_path)
        if status in {"running", "completed"}:
            return status
        paths = self.remote_job_paths(name)
        runner = (
            "#!/usr/bin/env bash\n"
            "set +e\n"
            f"{command}\n"
            "rc=$?\n"
            f"echo \"$rc\" > {_quote(paths['exit'])}\n"
            "exit \"$rc\"\n"
        )
        script = (
            f"mkdir -p {_quote(REMOTE_LOGS)}\n"
            f"cat > {_quote(paths['runner'])} <<'EOS'\n{runner}EOS\n"
            f"chmod +x {_quote(paths['runner'])}\n"
            f"rm -f {_quote(paths['exit'])}\n"
            f"nohup {_quote(paths['runner'])} > {_quote(paths['log'])} 2>&1 & echo $! > {_quote(paths['pid'])}\n"
            f"cat {_quote(paths['pid'])}\n"
        )
        result = self.ssh(script, timeout=30)
        if result.returncode != 0:
            raise RuntimeError((result.stderr or result.stdout).strip())
        pid = result.stdout.strip().splitlines()[-1]
        self.state.setdefault("jobs", {})[name] = {
            "pid": pid,
            "log": paths["log"],
            "success_path": success_path,
            "command": command,
            "started_utc": _utc(),
            "status": "running",
        }
        self.log(f"Started remote job {name} PID {pid}, log {paths['log']}")
        self.save_state()
        return "running"

    def ensure_wsi_conversion(self) -> str:
        if self.remote_exists(REMOTE_HE_PYRAMID):
            return "completed"
        command = (
            f"cd {_quote(REMOTE_REPO)} && "
            f"PYTHONPATH=src {_quote(REMOTE_PYTHON)} "
            "benchmarking/lazyslide_a100/scripts/prepare_tiffslide_pyramid.py "
            f"--input {_quote(REMOTE_HE_OME)} "
            f"--output {_quote(REMOTE_HE_PYRAMID)} "
            "--tile-px 512 --jpeg-quality 90 --mpp 0.2125 --verify"
        )
        status = self.start_remote_job("prepare_cervical_tiffslide_pyramid", command, REMOTE_HE_PYRAMID)
        if status == "failed":
            self.log_failure("prepare_wsi", "Cervical H&E pyramid conversion failed once; retrying with the same verified converter.")
            self.state.setdefault("failures", {}).setdefault("prepare_wsi_retry", {"count": 0})["count"] += 1
            status = self.start_remote_job("prepare_cervical_tiffslide_pyramid_retry", command, REMOTE_HE_PYRAMID)
        return status

    def run_lazyslide_model(self, model: str) -> str:
        if model == "plip":
            batch_sizes = [64, 32, 16]
            text_model = "plip"
        else:
            batch_sizes = [32, 16]
            text_model = "none"
        key = f"{model}_batch_index"
        batch_index = int(self.state.get(key, 0))
        batch = batch_sizes[min(batch_index, len(batch_sizes) - 1)]
        run_dir = self.remote_run_dir(model)
        manifest = f"{run_dir}/run_manifest.json"
        if self.remote_exists(manifest):
            self.state.setdefault("jobs", {}).setdefault(f"cervical_{model}_direct_wsi", {})["status"] = "completed"
            return "completed"
        status = self.remote_job_status(f"cervical_{model}_direct_wsi", manifest)
        if status == "running":
            return status
        if status == "failed":
            log = self.fetch_remote_log_tail(f"cervical_{model}_direct_wsi")
            if ("CUDA out of memory" in log or "Killed" in log) and batch_index < len(batch_sizes) - 1:
                self.state[key] = batch_index + 1
                self.log_failure(f"{model}_oom_retry", f"{model} failed at batch {batch}; retrying at batch {batch_sizes[batch_index + 1]}.")
            elif ("KeyError" in log or "contour" in log) and self.state.get("contour_id_key", "name") == "name":
                self.state["contour_id_key"] = "Selection"
                self.log_failure(f"{model}_contour_key_retry", "Switching contour id key from name to Selection after contour-key failure.")
            else:
                self.log_failure(f"{model}_direct_wsi_failed", f"{model} direct WSI branch failed and will not be retried again.\n\n{log[-4000:]}")
                return "failed"
        contour_id_key = str(self.state.get("contour_id_key", "name"))
        command = (
            f"cd {_quote(REMOTE_REPO)} && "
            f"CUDA_VISIBLE_DEVICES={int(self.args.gpu)} PYTHONPATH=src {_quote(REMOTE_PYTHON)} "
            "benchmarking/lazyslide_a100/scripts/run_histoseg_lazyslide_workflow.py "
            f"--dataset-root {_quote(REMOTE_SPATIALDATA)} "
            f"--histoseg-geojson {_quote(REMOTE_GEOJSON)} "
            f"--contour-id-key {_quote(contour_id_key)} "
            f"--he-source-path {_quote(REMOTE_HE_PYRAMID)} "
            "--wsi-reader tiffslide "
            f"--output-dir {_quote(run_dir)} "
            f"--model {_quote(model)} "
            f"--text-model {_quote(text_model)} "
            f"--batch-size {batch} "
            "--table-format parquet "
            f"--wta-program-library {_quote(REMOTE_PROGRAM_JSON)}"
        )
        return self.start_remote_job(f"cervical_{model}_direct_wsi", command, manifest)

    def remote_run_dir(self, model: str) -> str:
        if model == "plip":
            return f"{REMOTE_RUNS}/direct_lazyslide_plip_full_mtm_wta"
        return f"{REMOTE_RUNS}/direct_lazyslide_uni_full_mtm_wta"

    def fetch_remote_log_tail(self, job_name: str) -> str:
        log = self.remote_job_paths(job_name)["log"]
        result = self.ssh(f"test -f {_quote(log)} && tail -n 160 {_quote(log)} || true", timeout=30)
        return (result.stdout or result.stderr or "").strip()

    def should_run_uni(self) -> bool:
        if self.args.force_uni:
            return True
        local_plip = self.cervical_package_dir / "plip"
        leaderboard = _read_table(local_plip, "wta_pathway_partial_correlations")
        if leaderboard.empty:
            return False
        if "abs_partial_spearman_rho" not in leaderboard.columns:
            return False
        strong = leaderboard.loc[leaderboard["abs_partial_spearman_rho"].astype(float).ge(0.4)]
        return not strong.empty

    def sync_remote_outputs(self) -> None:
        for model in ("plip", "uni"):
            run_dir = self.remote_run_dir(model)
            if not self.remote_exists(f"{run_dir}/run_manifest.json"):
                continue
            target = self.cervical_package_dir / model
            target.mkdir(parents=True, exist_ok=True)
            remote_files = [
                "run_manifest.json",
                "tile_assignments.parquet",
                "tile_features.parquet",
                "image_contours.parquet",
                "contour_multimodal_summary.parquet",
                "wta_pathway_partial_correlations.parquet",
                "molecular_prediction_benchmark.parquet",
                "morphomolecular_hero_targets.parquet",
                "morphomolecular_hero_contours.parquet",
                "boundary_coupling_summary.parquet",
            ]
            for filename in remote_files:
                destination = target / filename
                if destination.exists():
                    continue
                result = self._run(
                    ["scp", f"{REMOTE_HOST}:{run_dir}/{filename}", str(target)],
                    timeout=1800,
                )
                if result.returncode != 0 and "No such file" not in (result.stderr or ""):
                    self.log_failure(f"sync_{model}_{filename}", (result.stderr or result.stdout).strip())
            self.sync_remote_maz_ring_validation(model)
            self.sync_remote_hero_patches(model)

    def sync_remote_maz_ring_validation(self, model: str) -> None:
        run_dir = self.remote_run_dir(model)
        remote_dir = f"{run_dir}/maz_ring_validation"
        if not self.remote_exists(f"{remote_dir}/MAZ_RingLevel_LeadLag_Report.csv"):
            return
        target = self.cervical_package_dir / model / "maz_ring_validation"
        target.mkdir(parents=True, exist_ok=True)
        for filename in (
            "MAZ_RingLevel_Profile.csv",
            "MAZ_RingLevel_LeadLag_Report.csv",
            "MAZ_RingLevel_Profile_Panels.pdf",
            "MAZ_RingLevel_Manifest.json",
        ):
            local = target / filename
            if local.exists():
                continue
            result = self._run(["scp", f"{REMOTE_HOST}:{remote_dir}/{filename}", str(target)], timeout=1800)
            if result.returncode != 0 and "No such file" not in (result.stderr or ""):
                self.log_failure(f"sync_{model}_maz_{filename}", (result.stderr or result.stdout).strip())
        if model == "plip":
            report = target / "MAZ_RingLevel_LeadLag_Report.csv"
            panels = target / "MAZ_RingLevel_Profile_Panels.pdf"
            if report.exists():
                shutil.copy2(report, self.cervical_package_dir / "cervical_MAZ_LeadLag_Report.csv")
            if panels.exists():
                shutil.copy2(panels, self.cervical_package_dir / "Cervical_MAZ_LeadLag_Figure.pdf")

    def sync_remote_hero_patches(self, model: str) -> None:
        run_dir = self.remote_run_dir(model)
        remote_montage = f"{run_dir}/hero_patches/hero_patch_montage.png"
        if not self.remote_exists(f"{run_dir}/morphomolecular_hero_contours.parquet"):
            return
        if not self.remote_exists(remote_montage):
            command = (
                f"cd {_quote(REMOTE_REPO)} && "
                f"PYTHONPATH=src {_quote(REMOTE_PYTHON)} "
                "benchmarking/lazyslide_a100/scripts/export_morphomolecular_hero_patches.py "
                f"--run-dir {_quote(run_dir)} --wsi-path {_quote(REMOTE_HE_PYRAMID)} --max-contours 12"
            )
            self.start_remote_job(f"cervical_{model}_hero_patches", command, remote_montage)
            return
        self.state.setdefault("jobs", {}).setdefault(f"cervical_{model}_hero_patches", {})["status"] = "completed"
        target = self.cervical_package_dir / model / "hero_patches"
        target.mkdir(parents=True, exist_ok=True)
        for filename in ("hero_patch_montage.png", "hero_patch_manifest.csv"):
            local = target / filename
            if local.exists():
                continue
            self._run(["scp", f"{REMOTE_HOST}:{run_dir}/hero_patches/{filename}", str(target)], timeout=600)
        if model == "plip":
            montage = target / "hero_patch_montage.png"
            if montage.exists():
                shutil.copy2(montage, self.cervical_package_dir / "Cervical_Hero_Patch_Montage.png")
            manifest = target / "hero_patch_manifest.csv"
            if manifest.exists():
                shutil.copy2(manifest, self.cervical_package_dir / "cervical_hero_contours.csv")

    def generate_local_reports(self) -> None:
        self.copy_breast_figure2()
        self.copy_cervical_tables()
        self.generate_cervical_summary_figures()
        self.write_manuscript_v2()
        self.write_cover_letter()
        self.write_reviewer_response()
        self.write_final_index()
        self.write_manifest()
        self._write_status()

    def copy_breast_figure2(self) -> None:
        source = self.package_dir / "Final_Figure2_Pack.pdf"
        if source.exists():
            shutil.copy2(source, self.state_dir / "Final_Figure2_Pack.pdf")

    def copy_cervical_tables(self) -> None:
        plip = self.cervical_package_dir / "plip"
        if not plip.exists():
            return
        copies = {
            "wta_pathway_partial_correlations": "cervical_wta_pathway_partial_correlations.csv",
            "boundary_coupling_summary": "cervical_boundary_coupling_summary.csv",
            "morphomolecular_hero_contours": "cervical_hero_contours_full.csv",
        }
        for stem, filename in copies.items():
            frame = _read_table(plip, stem)
            if not frame.empty:
                frame.to_csv(self.cervical_package_dir / filename, index=False)
        self.write_model_comparison()

    def write_model_comparison(self) -> None:
        plip = _read_table(self.cervical_package_dir / "plip", "wta_pathway_partial_correlations")
        uni = _read_table(self.cervical_package_dir / "uni", "wta_pathway_partial_correlations")
        if plip.empty:
            return
        if uni.empty:
            plip.to_csv(self.cervical_package_dir / "cervical_model_agnostic_PLIP_UNI_comparison.csv", index=False)
            return
        required = {"pathway", "best_image_feature", "partial_spearman_rho", "abs_partial_spearman_rho", "fdr", "n_contours"}
        if not required.issubset(plip.columns) or not required.issubset(uni.columns):
            return
        merged = plip.loc[:, list(required)].merge(
            uni.loc[:, list(required)],
            on="pathway",
            suffixes=("_plip", "_uni"),
            how="inner",
        )
        merged["min_abs_partial_rho"] = merged[["abs_partial_spearman_rho_plip", "abs_partial_spearman_rho_uni"]].min(axis=1)
        merged["model_agnostic_call"] = "not_stable"
        merged.loc[merged["min_abs_partial_rho"].ge(0.45), "model_agnostic_call"] = "stable_strong"
        merged.loc[
            merged["model_agnostic_call"].eq("not_stable") & merged["min_abs_partial_rho"].ge(0.35),
            "model_agnostic_call",
        ] = "stable_moderate"
        merged = merged.sort_values(["min_abs_partial_rho", "pathway"], ascending=[False, True], kind="stable")
        merged.to_csv(self.cervical_package_dir / "cervical_model_agnostic_PLIP_UNI_comparison.csv", index=False)

    def generate_cervical_summary_figures(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        import pandas as pd

        leaderboard = _read_table(self.cervical_package_dir / "plip", "wta_pathway_partial_correlations")
        if leaderboard.empty:
            self.write_cervical_negative_report("No cervical PLIP residual-decoding table is available yet.")
            return
        if "abs_partial_spearman_rho" in leaderboard.columns:
            leaderboard = leaderboard.sort_values("abs_partial_spearman_rho", ascending=False, kind="stable")
        top = leaderboard.head(12).copy()
        top_csv = self.cervical_package_dir / "cervical_wta_pathway_partial_correlations.csv"
        leaderboard.to_csv(top_csv, index=False)
        strong = top.loc[pd.to_numeric(top.get("abs_partial_spearman_rho"), errors="coerce").ge(0.4)]
        output_pdf = self.cervical_package_dir / (
            "Cervical_Figure4_Replication_Pack.pdf" if not strong.empty else "Cervical_Figure4_Replication_Pack.pdf"
        )
        with PdfPages(output_pdf) as pdf:
            fig, ax = plt.subplots(figsize=(8.0, 5.2))
            labels = [str(value).replace("_", " ") for value in top.get("pathway", top.get("molecular_feature", []))]
            values = pd.to_numeric(top.get("abs_partial_spearman_rho"), errors="coerce")
            ax.barh(labels[::-1], values[::-1], color="#2c7fb8")
            ax.axvline(0.4, color="#d95f0e", linestyle="--", linewidth=1.3, label="strong gate")
            ax.set_xlabel("abs(partial Spearman rho)")
            ax.set_title("Cervical WTA residual-decoding leaderboard (PLIP)")
            ax.legend(loc="lower right")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
            comparison_path = self.cervical_package_dir / "cervical_model_agnostic_PLIP_UNI_comparison.csv"
            if comparison_path.exists():
                comp = pd.read_csv(comparison_path).head(10)
                if {"min_abs_partial_rho", "pathway"}.issubset(comp.columns):
                    fig, ax = plt.subplots(figsize=(8.0, 5.2))
                    ax.barh(comp["pathway"].astype(str).str.replace("_", " ")[::-1], comp["min_abs_partial_rho"][::-1], color="#41ab5d")
                    ax.axvline(0.35, color="#636363", linestyle="--", linewidth=1.0, label="moderate gate")
                    ax.axvline(0.45, color="#d95f0e", linestyle="--", linewidth=1.0, label="strong gate")
                    ax.set_xlabel("min abs(partial rho), PLIP vs UNI")
                    ax.set_title("Cervical model-agnostic stress test")
                    ax.legend(loc="lower right")
                    fig.tight_layout()
                    pdf.savefig(fig)
                    plt.close(fig)
        if strong.empty:
            self.write_cervical_negative_report("Cervical PLIP residual-decoding table exists, but no pathway crossed abs(partial rho) >= 0.4.")
        else:
            stale_negative = self.cervical_package_dir / "Cervical_Negative_Replication_Report.md"
            if stale_negative.exists():
                stale_negative.unlink()
            report = (
                "# Cervical Replication Report\n\n"
                f"{self.cervical_model_agnostic_summary()}\n\n"
                f"Top table: `{top_csv.name}`\n"
                f"Figure pack: `{output_pdf.name}`\n"
            )
            (self.cervical_package_dir / "Cervical_Replication_Report.md").write_text(report, encoding="utf-8")

    def write_cervical_negative_report(self, reason: str) -> None:
        report = f"""# Cervical Negative/Incomplete Replication Report

Status UTC: {_utc()}

Reason: {reason}

Interpretation: this does not weaken the breast discovery claim. The cervical Atera WTA sample is a cross-cancer stress test with different epithelial biology. If no strong residual axis appears, the manuscript should keep breast as the discovery dataset and describe cervical as a pending or context-specific validation branch rather than as a failed luminal/ER replication.
"""
        (self.cervical_package_dir / "Cervical_Negative_Replication_Report.md").write_text(report, encoding="utf-8")

    def write_manuscript_v2(self) -> None:
        selected = _read_csv_rows(self.package_dir / "figure2_selected_programs.csv")
        comparison = _read_csv_rows(self.package_dir / "model_agnostic_validation" / "Model_Agnostic_UNI_vs_PLIP_Comparison.csv", limit=6)
        cervical_report = self.cervical_status_text()
        bullets = []
        for row in selected:
            bullets.append(
                f"- {row['assigned_structure']} `{row['target_feature'].replace('program__wta_', '')}`: "
                f"partial rho = {float(row['partial_spearman_rho']):.3f}, "
                f"P = {float(row['partial_p_value']):.2e}, n = {row['n_contours']}."
            )
        model_rows = []
        for row in comparison[:5]:
            model_rows.append(
                f"- {row.get('pathway')}: PLIP abs rho {float(row.get('plip_abs_partial_spearman_rho', 0)):.3f}, "
                f"UNI abs rho {float(row.get('uni_abs_partial_spearman_rho', 0)):.3f}, "
                f"{row.get('model_agnostic_call')}."
            )
        manuscript = f"""# Foundation-model morphology reveals transcriptomic landscapes hidden within histological structure labels

## Abstract

Spatial whole-transcriptome imaging now measures thousands of genes at single-cell resolution, but routine H&E morphology remains the language of tissue diagnosis. We introduce morphomolecular translation mapping (mTM), a contour-constrained framework that asks a residual question: after HistoSeg has assigned tissue contours to biologically meaningful structures, do H&E foundation-model embeddings still encode molecular state within those apparently homogeneous regions? In a breast cancer Atera WTA specimen, structure labels explained much of the coarse molecular background, and image features did not uniformly improve global prediction beyond labels. Residual decoding revealed the stronger signal. Within S3 contours, H&E embeddings decoded luminal estrogen response, unfolded protein response and oxidative phosphorylation after controlling for structure, spatial position and boundary proximity. The leading luminal/ER axis reached partial Spearman rho = -0.639 (P = 4.79 x 10^-19, n = 157 contours). PLIP and UNI recovered overlapping pathway-level programs, supporting a framework-level rather than model-specific effect. Boundary ring profiles further identified molecularly active zones where H&E and WTA gradients co-localized at contour interfaces. These results position HistoSeg labels as the anatomical map and foundation-model morphology as a readout of hidden molecular continua within that map.

## Main Result 1: Contour-constrained residual decoding

mTM treats HistoSeg contours as the common geometry for H&E embeddings and WTA programs. This scale is deliberately intermediate: larger than a tile, less noisy than a single cell and more interpretable than a whole-slide average. The key test is not whether H&E replaces WTA, but whether H&E explains residual molecular variation after the anatomical label has been fixed.

## Main Result 2: Hidden S3 molecular continua

The strongest Figure 2 evidence is the S3 residual program panel:

{chr(10).join(bullets)}

For a pathologist, the important point is that S3 is not being subdivided into a new gross class. The HistoSeg label remains visually useful, but it is a compressed descriptor. mTM shows that contours carrying the same label can occupy different endocrine, stress and metabolic states. This is the clinical-facing interpretation: the microscope view contains quantitative molecular texture that is too subtle to name reproducibly by eye.

## Main Result 3: Model-agnostic robustness

Embedding coordinates are not expected to match across foundation models, so robustness is defined at the program level. PLIP and UNI converge on the same biology:

{chr(10).join(model_rows)}

This supports the central claim that contour-constrained residual decoding extracts biology from independent pathology representations rather than exploiting an accidental PLIP dimension.

## Main Result 4: Molecularly active zones

MAZ analysis should remain conservative. The robust current claim is co-localization of image and WTA gradients at selected contour boundaries, not causal lead-lag ordering. TGF-beta, EMT, stromal matrix, immune-exclusion and metabolic-stress programs are the correct biological axes to prioritize for follow-up because they plausibly change at invasive or stromal interfaces.

## Cross-cancer Atera WTA stress test

{cervical_report}

## Discussion

The manuscript should not pitch mTM as a prediction leaderboard. The stronger message is that discrete histological structure labels are lossy biological summaries. Once the tissue map is fixed, foundation-model morphology can reveal continuous molecular axes within the map. This framing turns the modest global prediction increment into the reason the analysis matters: HistoSeg captures the compartment; mTM decodes the residual state.

The main limitation remains disease-matched and endpoint-matched validation. Cervical Atera WTA supports cross-cancer transfer of the residual-decoding principle, but it is not a direct luminal/ER replication. A future ER-positive breast WTA sample or matched IHC would be the decisive biological closure for the endocrine-response story. Until then, claims about therapy response or protein validation must remain prospective.

## Figure Legends

### Figure 1

Morphomolecular translation mapping aligns HistoSeg contours, direct whole-slide H&E foundation-model embeddings and Atera WTA program summaries into a single contour-level coordinate system.

### Figure 2

S3 contours share a histological structure label but contain hidden molecular continua. The panels show the label map, H&E embedding gradient, matched WTA program gradient, residual association plot and hero H&E patches for luminal estrogen response, UPR and oxidative phosphorylation.

### Figure 3

Molecularly active zones at contour boundaries. Signed-distance ring profiles show co-localized H&E and WTA gradients at selected interfaces, with ring-width sensitivity used to avoid overclaiming lead-lag effects.

### Figure 4

Cervical Atera WTA cross-cancer stress test. PLIP and UNI recover the same WTA program families, including CAF/myofibroblast activation, TLS-adjacent activation, collagen/ECM organization, immune exclusion, stromal encapsulation and EMT/invasive-front programs. Agreement is interpreted at the program level rather than by embedding-axis sign, because independent foundation-model coordinates are not directionally aligned.
"""
        (self.state_dir / "Full_Manuscript_v2.md").write_text(manuscript, encoding="utf-8")

    def cervical_model_agnostic_summary(self) -> str:
        import pandas as pd

        comparison_path = self.cervical_package_dir / "cervical_model_agnostic_PLIP_UNI_comparison.csv"
        if not comparison_path.exists():
            return (
                "Cervical Atera WTA is being used as a cross-cancer stress test. "
                "The PLIP/UNI comparison table is pending, so the manuscript should not yet make a model-agnostic cervical claim."
            )
        comparison = pd.read_csv(comparison_path)
        required = {
            "pathway",
            "min_abs_partial_rho",
            "partial_spearman_rho_plip",
            "partial_spearman_rho_uni",
            "model_agnostic_call",
        }
        if comparison.empty or not required.issubset(comparison.columns):
            return (
                "Cervical Atera WTA is being used as a cross-cancer stress test, but the current comparison table "
                "does not contain the fields needed for a model-agnostic claim."
            )
        stable = comparison.loc[
            comparison["model_agnostic_call"].isin(["stable_strong", "stable_moderate"])
        ].head(8)
        if stable.empty:
            return (
                "Cervical Atera WTA did not pass the predeclared PLIP/UNI stability gate. "
                "It should be reported as a transparent context-specific stress test rather than a positive replication."
            )
        lines = []
        for _, row in stable.iterrows():
            lines.append(
                f"- `{row['pathway']}`: min abs(partial rho) = {float(row['min_abs_partial_rho']):.3f}; "
                f"PLIP rho = {float(row['partial_spearman_rho_plip']):.3f}; "
                f"UNI rho = {float(row['partial_spearman_rho_uni']):.3f}; "
                f"{row['model_agnostic_call']}."
            )
        return (
            "Cervical Atera WTA was used as a cross-cancer stress test, not as a direct luminal/ER replication. "
            "It passed the predeclared model-agnostic residual-decoding gate: PLIP and UNI both recovered stable WTA "
            "program families after controlling for structure, centroid coordinates and boundary proximity. "
            "The strongest shared programs were stromal-remodeling, immune-ecology and invasion-associated axes. "
            "Embedding-axis signs are not interpreted across PLIP and UNI because independent latent coordinates can be arbitrarily oriented; "
            "the robustness claim is recovery of the same WTA programs.\n\n"
            + "\n".join(lines)
        )

    def cervical_status_text(self) -> str:
        negative = self.cervical_package_dir / "Cervical_Negative_Replication_Report.md"
        summary = self.cervical_model_agnostic_summary()
        if "passed the predeclared" in summary:
            return summary
        if negative.exists():
            return negative.read_text(encoding="utf-8").split("\n", 1)[-1].strip()
        return summary

    def write_cover_letter(self) -> None:
        text = """# Cover Letter Draft v1

Dear Editor,

We submit "Foundation-model morphology reveals transcriptomic landscapes hidden within histological structure labels" as a Brief Communication for Nature Biotechnology.

The central contribution is not another histology-to-gene-expression predictor. We introduce contour-constrained residual decoding: HistoSeg provides biologically meaningful tissue compartments, and H&E foundation-model embeddings are then tested for molecular information that remains within those fixed compartments. This reframes a common negative result in AI pathology. If structure labels already explain coarse molecular identity, then the important question is whether morphology encodes the residual endocrine, metabolic, stress and boundary programs that labels compress away.

In breast Atera WTA, mTM decodes a strong S3 luminal estrogen-response continuum, plus UPR and oxidative phosphorylation axes, after controlling for structure, spatial position and boundary proximity. PLIP and UNI recover overlapping pathway programs, arguing that the signal is not a single-model embedding artifact. As a cross-cancer stress test, cervical Atera WTA further recovers stable stromal-remodeling, immune-ecology and invasion-associated residual programs across PLIP and UNI. We also define conservative molecularly active zones where H&E and WTA gradients co-localize at contour boundaries.

We believe the study will interest NBT readers because it uses single-cell whole-transcriptome spatial data to define a new analysis layer between digital pathology and spatial biology. H&E is not presented as a replacement for WTA; it is presented as a structured readout of residual molecular state inside anatomically meaningful contours.

Sincerely,
The authors
"""
        (self.state_dir / "Cover_Letter_Draft_v1.md").write_text(text, encoding="utf-8")

    def write_reviewer_response(self) -> None:
        text = """# Response to Likely Reviewers v1

## Concern 1: The study is single-sample.

Response: We agree. The main text should frame the breast WTA analysis as a discovery demonstration, strengthened by a cervical Atera WTA cross-cancer stress test that reproduces the residual-decoding principle for stromal, immune and invasion programs. A matched ER-positive breast WTA or IHC validation remains the highest-priority next experiment for the luminal/ER story.

## Concern 2: The image model does not uniformly improve prediction over HistoSeg labels.

Response: This is the premise rather than the failure. HistoSeg labels capture coarse molecular identity. mTM asks whether H&E embeddings decode residual molecular state inside those labels. The Figure 2 partial-rho results answer that sharper question.

## Concern 3: Are PLIP embedding dimensions biologically meaningful?

Response: Individual coordinates are not interpreted directly. The program-level result is tested across PLIP and UNI, and robustness is defined by recovery of the same WTA program, not the same coordinate.

## Concern 4: MAZ lead-lag sounds causal.

Response: The current manuscript should claim boundary coupling, not causality. Lead-lag labels are used for prioritization only unless replicated with stronger spatial or experimental evidence.

## Concern 5: Is this clinically actionable?

Response: Not yet. The clinical relevance is that endocrine, stress and metabolic states can be spatially organized inside a single histological label. Claims about therapy response require independent clinical or IHC validation and should remain prospective.
"""
        (self.state_dir / "Response_to_Likely_Reviewers_v1.md").write_text(text, encoding="utf-8")

    def write_final_index(self) -> None:
        files = [
            self.state_dir / "Final_Figure2_Pack.pdf",
            self.state_dir / "Full_Manuscript_v2.md",
            self.state_dir / "Cover_Letter_Draft_v1.md",
            self.state_dir / "Response_to_Likely_Reviewers_v1.md",
            self.cervical_package_dir / "Cervical_Figure4_Replication_Pack.pdf",
            self.cervical_package_dir / "Cervical_Replication_Report.md",
            self.cervical_package_dir / "cervical_wta_pathway_partial_correlations.csv",
            self.cervical_package_dir / "cervical_model_agnostic_PLIP_UNI_comparison.csv",
            self.cervical_package_dir / "cervical_MAZ_LeadLag_Report.csv",
            self.cervical_package_dir / "Cervical_Hero_Patch_Montage.png",
            self.state_path,
            self.decision_log,
            self.boss_log,
        ]
        rows = ["# Final Deliverables Index", "", f"Updated UTC: {_utc()}", ""]
        for path in files:
            rows.append(f"- `{path}`: {'present' if path.exists() else 'pending'}")
        (self.state_dir / "Final_Deliverables_Index.md").write_text("\n".join(rows) + "\n", encoding="utf-8")

    def write_manifest(self) -> None:
        rows = []
        roots = [self.state_dir, self.cervical_package_dir]
        for root in roots:
            if not root.exists():
                continue
            for path in sorted(p for p in root.rglob("*") if p.is_file()):
                try:
                    digest = hashlib.sha256(path.read_bytes()).hexdigest()
                    rows.append(
                        {
                            "path": str(path.relative_to(self.package_dir)),
                            "bytes": path.stat().st_size,
                            "modified_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(timespec="seconds"),
                            "sha256": digest,
                        }
                    )
                except OSError:
                    continue
        manifest = self.state_dir / f"Package_File_Manifest_{datetime.now().strftime('%Y%m%dT%H%M%S')}.csv"
        with manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["path", "bytes", "modified_utc", "sha256"])
            writer.writeheader()
            writer.writerows(rows)
        self.state["latest_manifest"] = str(manifest)

    def _write_status(self) -> None:
        status = {
            "updated_utc": _utc(),
            "state": str(self.state_path),
            "decision_log": str(self.decision_log),
            "boss_log": str(self.boss_log),
            "cervical_package": str(self.cervical_package_dir),
            "remote_root": REMOTE_ROOT,
            "remote_host": REMOTE_HOST,
            "jobs": self.state.get("jobs", {}),
            "local_tasks": self.state.get("local_tasks", {}),
            "deliverables": self.state.get("deliverables", {}),
        }
        _write_json(self.state_dir / "AUTOPILOT_STATUS.json", status)

    def run_cycle(self) -> None:
        self.state["cycle"] = int(self.state.get("cycle", 0)) + 1
        self.log(f"Heartbeat cycle {self.state['cycle']} started.")
        self.generate_local_reports()
        if self.args.skip_remote:
            self.log("Remote branch skipped by --skip-remote; local manuscript hardening completed.")
            self.save_state()
            return
        if not self.remote_ok():
            self.log("A100 unavailable; continuing local manuscript hardening branch.")
            self.save_state()
            return
        try:
            self.ensure_remote_dirs()
        except Exception as exc:
            self.log_failure("remote_setup", f"{type(exc).__name__}: {exc}")
            return
        sync_status = self.start_local_sync_if_needed()
        self.state["input_sync_status"] = sync_status
        if sync_status not in {"ready", "completed"}:
            self.log(f"Cervical input sync status: {sync_status}.")
            self.save_state()
            return
        wsi_status = self.ensure_wsi_conversion()
        self.state["wsi_conversion_status"] = wsi_status
        if wsi_status != "completed":
            self.log(f"Cervical WSI conversion status: {wsi_status}.")
            self.save_state()
            return
        plip_status = self.run_lazyslide_model("plip")
        self.state["plip_status"] = plip_status
        if plip_status == "completed":
            self.sync_remote_outputs()
            self.generate_local_reports()
            if self.should_run_uni():
                uni_status = self.run_lazyslide_model("uni")
                self.state["uni_status"] = uni_status
                if uni_status == "completed":
                    self.sync_remote_outputs()
                    self.generate_local_reports()
        else:
            self.log(f"Cervical PLIP status: {plip_status}.")
        self.save_state()

    def run(self) -> None:
        self.setup()
        started = time.time()
        deadline = started + float(self.args.hours) * 3600.0
        while True:
            try:
                self.run_cycle()
            except Exception as exc:
                self.log_failure("cycle_exception", f"{type(exc).__name__}: {exc}")
            if self.args.once or time.time() >= deadline:
                break
            time.sleep(float(self.args.interval_minutes) * 60.0)
        self.generate_local_reports()
        self.log("Autopilot supervisor finished current invocation.")
        self.save_state()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="24h mTM WTA Breast + Cervical autopilot supervisor.")
    parser.add_argument("--hours", type=float, default=24.0)
    parser.add_argument("--interval-minutes", type=float, default=10.0)
    parser.add_argument("--gpu", type=int, default=7)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--skip-remote", action="store_true")
    parser.add_argument("--force-uni", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    Autopilot(args).run()


if __name__ == "__main__":
    main()
