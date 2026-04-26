from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PDC_SCRIPT_DIR = REPO_ROOT / "benchmarking" / "lr_2026_atera" / "scripts" / "pdc"


def _load_script(name: str):
    path = PDC_SCRIPT_DIR / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def test_pdc_stage_plan_uses_scratch_outputs_and_minimal_raw_inputs(tmp_path):
    stage = _load_script("stage_to_pdc.py")
    source = tmp_path / "outs"
    source.mkdir()
    for name in stage.REQUIRED_RAW_FILES:
        (source / name).write_text("x", encoding="utf-8")
    (source / "spatialdata.zarr").mkdir()
    (source / "spatialdata.zarr" / ".zgroup").write_text("{}", encoding="utf-8")

    plan = stage.build_stage_plan(
        repo_root=tmp_path,
        local_xenium_root=source,
        remote_root="/cfs/klemming/scratch/h/hutaobo/pyxenium_lr_benchmark_2026-04",
        host="pdc",
        include_smoke=True,
        include_spatialdata_zarr=True,
        skip_existing=True,
        archive_path=tmp_path / "payload.tar.gz",
    )

    assert plan["missing_required"] == []
    assert plan["remote_root"].startswith("/cfs/klemming/scratch/h/hutaobo/")
    assert all(item["remote"].startswith(plan["remote_source_cache"]) for item in plan["raw_items"])
    assert any(item["name"] == "spatialdata.zarr" for item in plan["raw_items"])
    assert "transcripts.parquet" not in {item["name"] for item in plan["raw_items"]}


def test_pdc_submit_matrix_builds_slurm_dependencies_and_path_policy():
    submit = _load_script("submit_pdc_matrix.py")

    jobs = submit.build_jobs(
        ["pyxenium", "cellchat"],
        "/cfs/klemming/scratch/h/hutaobo/pyxenium_lr_benchmark_2026-04",
        "naiss2026-4-680",
        {"prepare", "env", "smoke", "pilot"},
        include_full=False,
    )

    by_id = {job["job_id"]: job for job in jobs}
    assert "pdc_prepare_full_bundle" in by_id
    assert "pdc_env_pyxenium" in by_id
    assert "pdc_smoke_common_cellchat" in by_id
    assert "pdc_pilot_common_cellchat" in by_id
    assert "pdc_full_common_cellchat" not in by_id
    smoke = by_id["pdc_smoke_common_pyxenium"]
    assert set(smoke["dependencies"]) == {"pdc_prepare_full_bundle", "pdc_env_pyxenium"}
    script = smoke["script"]
    assert "#SBATCH -A naiss2026-4-680" in script
    assert "TMPDIR=\"$ROOT/tmp\"" in script
    assert "PYTHONPATH=\"$ROOT/repo/src" in script
    assert "/data/taobo.hu" not in script
