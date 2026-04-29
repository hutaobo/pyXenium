from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PDC_SCRIPT_DIR = REPO_ROOT / "benchmarking" / "cci_2026_atera" / "scripts" / "pdc"


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
        remote_root="/cfs/klemming/scratch/h/hutaobo/pyxenium_cci_benchmark_2026-04",
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
    data_root = tmp_path / "benchmarking" / "cci_2026_atera" / "data"
    data_root.mkdir(parents=True)
    (data_root / "input_manifest.json").write_text("{}", encoding="utf-8")
    (data_root / "cci_resource_common.tsv").write_text("ligand\treceptor\nA\tB\n", encoding="utf-8")
    (data_root / "atera_smoke_panel.tsv").write_text("ligand\treceptor\nA\tB\n", encoding="utf-8")
    (data_root / "celltype_pairs.tsv").write_text("source\ttarget\nT\tB\n", encoding="utf-8")
    archive_entries = stage.create_archive(tmp_path, tmp_path / "payload.tar.gz", include_smoke=True)
    assert "data/input_manifest.json" not in {entry["archive"] for entry in archive_entries}


def test_pdc_submit_matrix_builds_slurm_dependencies_and_path_policy():
    submit = _load_script("submit_pdc_matrix.py")

    jobs = submit.build_jobs(
        ["pyxenium", "cellchat"],
        "/cfs/klemming/scratch/h/hutaobo/pyxenium_cci_benchmark_2026-04",
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


def test_pdc_submit_matrix_full_backfill_chunks_commot_and_cellagentchat():
    submit = _load_script("submit_pdc_matrix.py")

    jobs = submit.build_jobs(
        ["commot", "cellchat", "cellagentchat", "cellnest", "scild"],
        "/cfs/klemming/scratch/h/hutaobo/pyxenium_cci_benchmark_2026-04",
        "naiss2026-4-680",
        {"prepare", "env"},
        include_full=True,
        commot_chunks=4,
        cellagentchat_chunks=4,
    )

    by_id = {job["job_id"]: job for job in jobs}
    commot_full = [job for job in jobs if job["method"] == "commot" and job["job_type"] == "full_common"]
    cellagentchat_full = [job for job in jobs if job["method"] == "cellagentchat" and job["job_type"] == "full_common"]
    cellagentchat_aggregate = by_id["pdc_full_common_cellagentchat_aggregate"]
    cellchat_full = by_id["pdc_full_common_cellchat"]

    assert submit.PDC_FULL_BACKFILL_METHODS == (
        "cellchat",
        "commot",
        "giotto",
        "spatalk",
        "niches",
        "cellnest",
        "cellagentchat",
        "scild",
    )
    assert len(commot_full) == 4
    assert {job["chunk_id"] for job in commot_full} == {0, 1, 2, 3}
    assert all(job["num_chunks"] == 4 for job in commot_full)
    assert "numpy<2" in by_id["pdc_env_commot"]["script"]
    assert "python -m pip install" in by_id["pdc_env_commot"]["script"]
    assert 'rm -f "$OUT/method_card.md"' in by_id["pdc_full_common_commot_chunk_000_of_004"]["script"]
    assert "--chunk-id 0 --num-chunks 4" in by_id["pdc_full_common_commot_chunk_000_of_004"]["script"]
    assert "/runs/full_common/commot/chunk_000_of_004" in by_id["pdc_full_common_commot_chunk_000_of_004"]["script"]
    assert len(cellagentchat_full) == 4
    assert {job["chunk_id"] for job in cellagentchat_full} == {0, 1, 2, 3}
    assert all(job["num_chunks"] == 4 for job in cellagentchat_full)
    assert by_id["pdc_full_common_cellagentchat_chunk_000_of_004"]["partition"] == "memory"
    assert by_id["pdc_full_common_cellagentchat_chunk_000_of_004"]["memory"] == "300G"
    assert "CELLAGENTCHAT_FEATURE_SELECTION=\"${CELLAGENTCHAT_FEATURE_SELECTION:-0}\"" in by_id["pdc_full_common_cellagentchat_chunk_000_of_004"]["script"]
    assert "--chunk-id 0 --num-chunks 4" in by_id["pdc_full_common_cellagentchat_chunk_000_of_004"]["script"]
    assert cellagentchat_aggregate["job_type"] == "full_common_aggregate"
    assert set(cellagentchat_aggregate["dependencies"]) == {
        "pdc_full_common_cellagentchat_chunk_000_of_004",
        "pdc_full_common_cellagentchat_chunk_001_of_004",
        "pdc_full_common_cellagentchat_chunk_002_of_004",
        "pdc_full_common_cellagentchat_chunk_003_of_004",
    }
    assert "cellagentchat_standardized.tsv.gz" in cellagentchat_aggregate["script"]
    assert "PyYAML>=6.0" in by_id["pdc_env_cellagentchat"]["script"]
    assert "PyYAML>=6.0" in by_id["pdc_env_cellnest"]["script"]
    assert "PyYAML>=6.0" in by_id["pdc_env_scild"]["script"]
    assert "rev-parse HEAD" in by_id["pdc_env_cellagentchat"]["script"]
    assert "source_checkout_ok cellagentchat already at" in by_id["pdc_env_cellagentchat"]["script"]
    assert "cat-file -e" in by_id["pdc_env_cellagentchat"]["script"]
    assert "namespace unavailable after install" in by_id["pdc_env_cellchat"]["script"]
    assert 'dependencies = TRUE' not in by_id["pdc_env_cellchat"]["script"]
    assert 'dependencies = c("Depends", "Imports", "LinkingTo")' in by_id["pdc_env_cellchat"]["script"]
    assert set(cellchat_full["dependencies"]) == {"pdc_prepare_full_bundle", "pdc_env_cellchat"}
    assert (
        submit.dependency_clause(cellchat_full, {"pdc_prepare_full_bundle": "101", "pdc_env_cellchat": "202"})
        == "afterok:101:202"
    )
