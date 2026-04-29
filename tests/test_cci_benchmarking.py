from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmwrite

from pyXenium.benchmarking.cci_atera import (
    STANDARDIZED_RESULT_COLUMNS,
    aggregate_standardized_results,
    compute_canonical_recovery,
    compute_novelty_support,
    compute_pathway_relevance,
    compute_robustness,
    compute_spatial_coherence,
    harmonize_adata_for_benchmark,
    prepare_atera_cci_benchmark,
    render_atera_cci_benchmark_report,
    score_biological_performance,
    standardize_result_table,
)
from pyXenium.benchmarking.cci_a100 import (
    DEFAULT_A100_READONLY_XENIUM_ROOT,
    build_a100_job_matrix,
    build_a100_job_manifest,
    build_a100_sidecar_matrix,
    build_a100_stage_plan,
    build_engineering_summary,
    collect_a100_results,
    execute_a100_stage_plan,
    finalize_a100_all,
    finalize_cci_source_of_truth,
    monitor_a100_jobs,
    prepare_a100_bundle,
    run_a100_plan,
    summarize_run_status,
    submit_a100_matrix,
    validate_a100_path_policy,
    write_failed_method_card,
)
from pyXenium.benchmarking.cci_adapters import (
    _call_liana_bivariate,
    aggregate_commot_obsp_result,
    aggregate_liana_bivariate_result,
    estimate_commot_distance_threshold,
    ensure_categorical_obs,
    ensure_spatial_connectivities,
    flatten_squidpy_cci_result,
    read_sparse_bundle_as_adata,
    validate_input_manifest,
)


def _toy_adata() -> ad.AnnData:
    adata = ad.AnnData(
        X=sparse.csr_matrix(
            np.array(
                [
                    [5.0, 0.0, 1.0],
                    [4.0, 0.0, 0.0],
                    [0.0, 6.0, 0.0],
                    [0.0, 5.0, 1.0],
                ]
            )
        ),
        obs=pd.DataFrame(
            {
                "cluster": ["CAFs, DCIS Associated", "CAFs, DCIS Associated", "T Lymphocytes", "T Lymphocytes"],
            },
            index=["cell_1", "cell_2", "cell_3", "cell_4"],
        ),
        var=pd.DataFrame(
            {
                "name": ["CXCL12", "CXCR4", "TGFB1"],
                "feature_type": ["Gene Expression", "Gene Expression", "Gene Expression"],
            },
            index=["ENSG1", "ENSG2", "ENSG3"],
        ),
    )
    adata.obsm["spatial"] = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [5.0, 0.0],
            [6.0, 0.0],
        ]
    )
    return adata


def _write_standardized_result(path: Path, *, method: str, ligand: str = "CXCL12", receptor: str = "CXCR4") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "method": method,
        "database_mode": "common-db",
        "ligand": ligand,
        "receptor": receptor,
        "sender": "A",
        "receiver": "B",
        "score_raw": 1.0,
        "score_std": 1.0,
        "rank_within_method": 1,
        "rank_fraction": 1.0,
        "fdr_or_pvalue": np.nan,
        "resolution": "celltype_pair",
        "spatial_support_type": "test",
        "artifact_path": str(path.parent),
    }
    pd.DataFrame([row], columns=STANDARDIZED_RESULT_COLUMNS).to_csv(path, sep="\t", index=False)
    (path.parent / "run_summary.json").write_text(
        json.dumps({"method": method, "status": "success", "n_rows": 1, "standardized_tsv": str(path)}),
        encoding="utf-8",
    )
    return path


def test_harmonize_adata_for_benchmark_promotes_gene_symbols_and_spatial_coords():
    bench = harmonize_adata_for_benchmark(_toy_adata())

    assert list(bench.var_names) == ["CXCL12", "CXCR4", "TGFB1"]
    assert list(bench.obs["cell_id"]) == ["cell_1", "cell_2", "cell_3", "cell_4"]
    assert list(bench.obs["cell_type"]) == ["CAFs, DCIS Associated", "CAFs, DCIS Associated", "T Lymphocytes", "T Lymphocytes"]
    assert list(bench.obs["x"]) == [0.0, 1.0, 5.0, 6.0]
    assert list(bench.obs["y"]) == [0.0, 0.0, 0.0, 0.0]
    assert list(bench.var["ensembl_id"]) == ["ENSG1", "ENSG2", "ENSG3"]


def test_prepare_atera_cci_benchmark_writes_sparse_bundle_and_manifest(monkeypatch, tmp_path):
    monkeypatch.setattr("pyXenium.io.read_xenium", lambda *args, **kwargs: _toy_adata())

    payload = prepare_atera_cci_benchmark(
        dataset_root=tmp_path / "dataset",
        benchmark_root=tmp_path / "benchmark",
        tbc_results=tmp_path / "dataset" / "results",
        smoke_n_cells=3,
        seed=123,
        prefer="h5",
    )

    manifest_path = tmp_path / "benchmark" / "data" / "input_manifest.json"
    full_h5ad = tmp_path / "benchmark" / "data" / "full" / "adata_full.h5ad"
    smoke_h5ad = tmp_path / "benchmark" / "data" / "smoke" / "adata_smoke.h5ad"
    common_db = tmp_path / "benchmark" / "data" / "cci_resource_common.tsv"

    assert manifest_path.exists()
    assert full_h5ad.exists()
    assert smoke_h5ad.exists()
    assert common_db.exists()
    assert payload["full_n_cells"] == 4
    assert payload["smoke_n_cells"] == 3
    assert payload["xenium_root"] == str(tmp_path / "dataset")
    assert payload["writable_benchmark_root"] == str(tmp_path / "benchmark")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["matrix_note"].startswith("Dense counts_symbol.tsv is intentionally omitted")
    assert Path(manifest["full_bundle"]["counts_symbol_mtx"]).exists()
    assert Path(manifest["smoke_bundle"]["coords_tsv"]).exists()
    assert manifest["full_bundle_fingerprints"]["counts_symbol_mtx"]["exists"] is True


def test_prepare_atera_cci_benchmark_can_skip_full_h5ad_but_keep_sparse_full_bundle(monkeypatch, tmp_path):
    monkeypatch.setattr("pyXenium.io.read_xenium", lambda *args, **kwargs: _toy_adata())

    payload = prepare_atera_cci_benchmark(
        dataset_root=tmp_path / "dataset",
        benchmark_root=tmp_path / "benchmark",
        tbc_results=tmp_path / "dataset" / "results",
        smoke_n_cells=3,
        seed=123,
        prefer="h5",
        export_full_bundle=True,
        write_full_h5ad=False,
    )

    assert payload["full_h5ad"] is None
    assert Path(payload["full_bundle"]["counts_symbol_mtx"]).exists()
    assert payload["write_full_h5ad"] is False


def test_read_sparse_bundle_as_adata_round_trips_bundle(tmp_path):
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    matrix_path = bundle_dir / "counts_symbol.mtx"
    barcodes_path = bundle_dir / "barcodes.tsv"
    genes_path = bundle_dir / "genes.tsv"
    meta_path = bundle_dir / "meta.tsv"
    coords_path = bundle_dir / "coords.tsv"

    mmwrite(matrix_path, sparse.coo_matrix(np.array([[1, 0, 2], [0, 3, 0]], dtype=float)))
    barcodes_path.write_text("c1\nc2\nc3\n", encoding="utf-8")
    pd.DataFrame({"gene_symbol": ["CXCL12", "CXCR4"], "ensembl_id": ["ENSG1", "ENSG2"]}).to_csv(genes_path, sep="\t", index=False)
    pd.DataFrame({"cell_id": ["c1", "c2", "c3"], "cell_type": ["A", "B", "A"], "x": [0, 1, 2], "y": [3, 4, 5]}).to_csv(meta_path, sep="\t", index=False)
    pd.DataFrame({"cell_id": ["c1", "c2", "c3"], "x": [0, 1, 2], "y": [3, 4, 5]}).to_csv(coords_path, sep="\t", index=False)

    adata = read_sparse_bundle_as_adata(
        {
            "counts_symbol_mtx": str(matrix_path),
            "barcodes_tsv": str(barcodes_path),
            "genes_tsv": str(genes_path),
            "meta_tsv": str(meta_path),
            "coords_tsv": str(coords_path),
        }
    )

    assert adata.shape == (3, 2)
    assert list(adata.obs_names) == ["c1", "c2", "c3"]
    assert list(adata.var_names) == ["CXCL12", "CXCR4"]
    assert list(adata.obs["cell_type"]) == ["A", "B", "A"]
    assert adata.obsm["spatial"].shape == (3, 2)


def test_validate_input_manifest_requires_full_input(tmp_path):
    cci_resource = tmp_path / "cci.tsv"
    cci_resource.write_text("ligand\treceptor\nCXCL12\tCXCR4\n", encoding="utf-8")
    manifest = {"cci_resource_common_tsv": str(cci_resource), "smoke_h5ad": str(tmp_path / "smoke.h5ad"), "full_bundle": {}}

    validation = validate_input_manifest(manifest, require_full=True)

    assert validation["valid"] is False
    assert any("full" in issue for issue in validation["issues"])


def test_standardize_and_score_biological_results():
    raw = pd.DataFrame(
        [
            {
                "ligand": "CXCL12",
                "receptor": "CXCR4",
                "sender_celltype": "CAFs, DCIS Associated",
                "receiver_celltype": "T Lymphocytes",
                "CCI_score": 0.8,
                "local_contact": 0.5,
            },
            {
                "ligand": "TGFB1",
                "receptor": "TGFBR2",
                "sender_celltype": "Endothelial Cells",
                "receiver_celltype": "Endothelial Cells",
                "CCI_score": 0.3,
                "local_contact": 0.2,
            },
        ]
    )
    standardized = standardize_result_table(
        raw,
        method="pyxenium",
        database_mode="common-db",
        extra_numeric_cols=("local_contact",),
    )

    assert set(STANDARDIZED_RESULT_COLUMNS).issubset(standardized.columns)
    assert list(standardized["rank_within_method"]) == [1.0, 2.0]
    assert list(standardized["score_std"]) == [1.0, 0.5]
    assert "local_contact" in standardized.columns

    canonical_axes = [
        {
            "name": "cxcl12_axis",
            "ligand": "CXCL12",
            "receptor": "CXCR4",
            "sender": "CAFs, DCIS Associated",
            "receiver": "T Lymphocytes",
            "weight": 1.0,
        }
    ]
    pathway_config = {"CXCL12/CXCR4": {"pairs": ["CXCL12^CXCR4"], "genes": ["CXCL12", "CXCR4"]}}
    _, canonical_summary = compute_canonical_recovery(standardized, canonical_axes=canonical_axes, top_k=5)
    _, pathway_summary = compute_pathway_relevance(standardized, pathway_config=pathway_config, top_n=5)
    spatial_summary = compute_spatial_coherence(standardized, top_n=5)
    _, novelty_summary = compute_novelty_support(standardized, top_n=5)
    biology = score_biological_performance(
        canonical_summary=canonical_summary,
        pathway_summary=pathway_summary,
        spatial_summary=spatial_summary,
        robustness_summary=None,
        novelty_summary=novelty_summary,
    )

    assert canonical_summary.iloc[0]["canonical_recovery_hits"] == 1
    assert pathway_summary.iloc[0]["pathway_hits"] >= 1
    assert spatial_summary.iloc[0]["spatial_coherence_score"] == 1.0
    assert biology.iloc[0]["biology_score"] > 0


def test_compute_robustness_uses_repeat_overlap():
    standardized = pd.DataFrame(
        [
            {
                "method": "pyxenium",
                "database_mode": "common-db",
                "ligand": "CXCL12",
                "receptor": "CXCR4",
                "sender": "CAFs, DCIS Associated",
                "receiver": "T Lymphocytes",
                "score_raw": 0.9,
                "score_std": 1.0,
                "rank_within_method": 1.0,
                "rank_fraction": 1.0,
                "fdr_or_pvalue": np.nan,
                "resolution": "celltype_pair",
                "spatial_support_type": "topology_local_contact",
                "artifact_path": "run1",
                "repeat_id": "r1",
            },
            {
                "method": "pyxenium",
                "database_mode": "common-db",
                "ligand": "CXCL12",
                "receptor": "CXCR4",
                "sender": "CAFs, DCIS Associated",
                "receiver": "T Lymphocytes",
                "score_raw": 0.8,
                "score_std": 1.0,
                "rank_within_method": 1.0,
                "rank_fraction": 1.0,
                "fdr_or_pvalue": np.nan,
                "resolution": "celltype_pair",
                "spatial_support_type": "topology_local_contact",
                "artifact_path": "run2",
                "repeat_id": "r2",
            },
        ]
    )

    summary = compute_robustness(standardized, repeat_col="repeat_id", top_n=5)

    assert summary.iloc[0]["repeat_pairs"] == 2
    assert summary.iloc[0]["robustness_score"] == 1.0


def test_standardize_uses_pvalue_to_break_score_ties():
    raw = pd.DataFrame(
        [
            {
                "ligand": "A",
                "receptor": "B",
                "sender_celltype": "s1",
                "receiver_celltype": "r1",
                "CCI_score": 1.0,
                "pvalue": 0.05,
            },
            {
                "ligand": "C",
                "receptor": "D",
                "sender_celltype": "s1",
                "receiver_celltype": "r1",
                "CCI_score": 1.0,
                "pvalue": 0.001,
            },
        ]
    )

    standardized = standardize_result_table(raw, method="toy", database_mode="common-db", pvalue_col="pvalue")

    assert list(standardized["ligand"]) == ["C", "A"]
    assert list(standardized["rank_within_method"]) == [1.0, 2.0]
    assert list(standardized["score_std"]) == [1.0, 0.5]


def test_aggregate_standardized_results_reads_gzip_outputs(tmp_path):
    standardized = standardize_result_table(
        pd.DataFrame(
            [
                {
                    "ligand": "CXCL12",
                    "receptor": "CXCR4",
                    "sender_celltype": "CAFs, DCIS Associated",
                    "receiver_celltype": "T Lymphocytes",
                    "CCI_score": 1.0,
                }
            ]
        ),
        method="laris",
        database_mode="common-db",
        spatial_support_type="knn_diffusion_lr",
    )
    gz_path = tmp_path / "standardized.tsv.gz"
    output_path = tmp_path / "combined.tsv"
    standardized.to_csv(gz_path, sep="\t", index=False, compression="gzip")

    combined = aggregate_standardized_results([gz_path], output_path=output_path)

    assert output_path.exists()
    assert len(combined) == 1
    assert combined.iloc[0]["method"] == "laris"


def test_flatten_squidpy_cci_result_handles_multiindex_columns():
    means = pd.DataFrame(
        [[0.8, 0.2], [0.1, 0.5]],
        index=pd.Index(["CXCL12^CXCR4", "TGFB1^TGFBR2"], name="interaction"),
        columns=pd.MultiIndex.from_tuples(
            [
                ("CAFs, DCIS Associated", "T Lymphocytes"),
                ("Endothelial Cells", "Pericytes"),
            ]
        ),
    )
    pvalues = pd.DataFrame(
        [[0.01, 0.2], [0.8, 0.03]],
        index=means.index,
        columns=means.columns,
    )
    metadata = pd.DataFrame({"source": ["CXCL12", "TGFB1"], "target": ["CXCR4", "TGFBR2"]}, index=means.index)

    flat = flatten_squidpy_cci_result({"means": means, "pvalues": pvalues, "metadata": metadata})

    assert len(flat) == 4
    top = flat.sort_values("CCI_score", ascending=False).iloc[0]
    assert top["ligand"] == "CXCL12"
    assert top["receptor"] == "CXCR4"
    assert top["sender_celltype"] == "CAFs, DCIS Associated"
    assert top["receiver_celltype"] == "T Lymphocytes"
    assert top["pvalue"] == 0.01


def test_aggregate_liana_bivariate_result_uses_spatial_neighbor_mix():
    source = harmonize_adata_for_benchmark(_toy_adata())
    ensure_spatial_connectivities(source, n_neighbors=1)
    local_scores = ad.AnnData(
        X=np.array([[0.9], [0.7], [0.2], [0.1]]),
        obs=source.obs.copy(),
        var=pd.DataFrame({"ligand": ["CXCL12"], "receptor": ["CXCR4"]}, index=["CXCL12^CXCR4"]),
    )

    flat = aggregate_liana_bivariate_result(local_scores, source)

    assert {"ligand", "receptor", "sender_celltype", "receiver_celltype", "CCI_score"}.issubset(flat.columns)
    assert set(flat["ligand"]) == {"CXCL12"}
    assert flat["CCI_score"].max() > 0


def test_call_liana_bivariate_disables_raw_lookup(monkeypatch):
    source = harmonize_adata_for_benchmark(_toy_adata())
    ensure_spatial_connectivities(source, n_neighbors=1)
    captured: dict[str, object] = {}

    def fake_bivariate(adata: ad.AnnData, **kwargs):
        captured.update(kwargs)
        captured["var_index_name"] = adata.var_names.name
        return ad.AnnData(
            X=np.array([[0.9], [0.7], [0.2], [0.1]]),
            obs=adata.obs.copy(),
            var=pd.DataFrame({"ligand": ["CXCL12"], "receptor": ["CXCR4"]}, index=["CXCL12^CXCR4"]),
        )

    fake_liana = types.SimpleNamespace(method=types.SimpleNamespace(bivariate=fake_bivariate))
    monkeypatch.setitem(sys.modules, "liana", fake_liana)

    result = _call_liana_bivariate(source, None, n_perms=7)

    assert isinstance(result, ad.AnnData)
    assert captured["use_raw"] is False
    assert captured["n_perms"] == 7
    assert captured["connectivity_key"] == "spatial_connectivities"
    assert captured["var_index_name"] is None


def test_aggregate_commot_obsp_result_collapses_cell_pair_matrix():
    adata = harmonize_adata_for_benchmark(_toy_adata())
    matrix = sparse.csr_matrix(
        np.array(
            [
                [0.0, 0.0, 2.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
    )
    adata.obsp["commot-pyx_common-CXCL12-CXCR4"] = matrix
    cci_resource = pd.DataFrame({"ligand": ["CXCL12"], "receptor": ["CXCR4"], "pathway": ["chemokine"]})

    flat = aggregate_commot_obsp_result(adata, cci_resource, database_name="pyx_common")

    assert len(flat) == 1
    row = flat.iloc[0]
    assert row["sender_celltype"] == "CAFs, DCIS Associated"
    assert row["receiver_celltype"] == "T Lymphocytes"
    assert row["CCI_score"] == 3.0
    assert row["edge_count"] == 2.0


def test_estimate_commot_distance_threshold_uses_neighbor_radius():
    adata = harmonize_adata_for_benchmark(_toy_adata())

    threshold = estimate_commot_distance_threshold(adata, n_neighbors=2, quantile=0.5)

    assert threshold > 0
    assert round(threshold, 6) == 4.5


def test_ensure_categorical_obs_promotes_string_labels():
    adata = harmonize_adata_for_benchmark(_toy_adata())

    promoted = ensure_categorical_obs(adata, "cell_type")

    assert isinstance(promoted.obs["cell_type"].dtype, pd.CategoricalDtype)
    assert list(promoted.obs["cell_type"].astype(str)) == list(adata.obs["cell_type"].astype(str))


def test_build_a100_stage_and_job_manifest(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    benchmark = repo / "benchmarking" / "cci_2026_atera"
    (benchmark / "configs").mkdir(parents=True)
    (benchmark / "envs").mkdir()
    (benchmark / "scripts").mkdir()
    (benchmark / "runners").mkdir()
    (benchmark / "data").mkdir()
    (repo / "src").mkdir()
    (repo / "pyproject.toml").write_text("[project]\nname='toy'\n", encoding="utf-8")
    (repo / "README.md").write_text("toy\n", encoding="utf-8")
    (repo / "LICENSE").write_text("toy\n", encoding="utf-8")
    (benchmark / "configs" / "methods.yaml").write_text(
        """methods:
  - slug: pyxenium
    display_name: pyXenium
    language: python
    env_name: pyx-cci-pyxenium
    runner: runners/python/run_pyxenium.py
  - slug: cellchat
    display_name: CellChat
    language: r
    env_name: r-cci-cellchat
    runner: runners/r/run_cellchat.R
""",
        encoding="utf-8",
    )
    monkeypatch.chdir(repo)

    remote_root = "/data/taobo.hu/test"
    stage = build_a100_stage_plan(benchmark_root=benchmark, remote_root=remote_root, transfer_mode="tar-scp")
    jobs = build_a100_job_manifest(benchmark_root=benchmark, remote_root=remote_root, methods=["pyxenium", "cellchat"])

    assert stage["requires_host"] is True
    assert stage["stage_data"] is False
    assert stage["transfer_mode"] == "tar-scp"
    assert stage["path_policy"]["valid"] is True
    assert any(item["destination"].endswith("/repo/pyproject.toml") for item in stage["local_items"])
    assert not any(item["destination"].endswith("/data") for item in stage["local_items"])
    assert "/data/taobo.hu/test/tmp" in stage["mkdir_command"]
    assert stage["local_archive"].endswith("a100_stage_payload.tar.gz")
    assert stage["remote_archive"].endswith("/logs/a100_stage_payload.tar.gz")
    assert len(stage["archive_entries"]) == len(stage["local_items"])
    assert any(command.startswith("scp ") for command in stage["copy_commands"])
    assert len(jobs["jobs"]) == 5
    assert jobs["jobs"][0]["job_id"] == "prepare_full_bundle"
    assert jobs["jobs"][0]["input_root"] == DEFAULT_A100_READONLY_XENIUM_ROOT
    assert "--xenium-root" in jobs["jobs"][0]["command"]
    assert DEFAULT_A100_READONLY_XENIUM_ROOT in jobs["jobs"][0]["command"]
    assert 'PATH="$HOME/miniconda3/bin:$HOME/miniconda3/condabin:$PATH"' in jobs["jobs"][0]["command"]
    assert "PYTHONNOUSERSITE=1" in jobs["jobs"][0]["command"]
    assert 'LD_LIBRARY_PATH="$HOME/miniconda3/envs/pyx-cci-prep/lib:$HOME/miniconda3/lib:${LD_LIBRARY_PATH:-}"' in jobs["jobs"][0]["command"]
    assert "TMPDIR=/data/taobo.hu/test/tmp" in jobs["jobs"][0]["command"]
    assert jobs["jobs"][0]["output_dir"] == "/data/taobo.hu/test/data"
    assert jobs["run_group"] == "full_common"
    assert jobs["path_policy"]["valid"] is True
    assert any(job["job_id"] == "full_common_db_pyxenium" for job in jobs["jobs"])
    assert any(job["output_dir"] == "/data/taobo.hu/test/runs/full_common/pyxenium" for job in jobs["jobs"])
    assert any("Rscript" in job["command"] for job in jobs["jobs"] if job["method"] == "cellchat")


def test_build_a100_job_matrix_chunks_commot_and_assigns_gpu_slots(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    benchmark = repo / "benchmarking" / "cci_2026_atera"
    (benchmark / "configs").mkdir(parents=True)
    (benchmark / "runs").mkdir(parents=True)
    (benchmark / "results").mkdir(parents=True)
    (repo / "src").mkdir()
    (repo / "pyproject.toml").write_text("[project]\nname='toy'\n", encoding="utf-8")
    (benchmark / "configs" / "methods.yaml").write_text(
        """methods:
  - slug: commot
    display_name: COMMOT
    language: python
    env_name: pyx-cci-commot
    runner: scripts/run_method.py
  - slug: cellnest
    display_name: CellNEST
    language: python
    env_name: pyx-cci-cellnest
    runner: scripts/run_method.py
  - slug: cellagentchat
    display_name: CellAgentChat
    language: python
    env_name: pyx-cci-cellagentchat
    runner: scripts/run_method.py
  - slug: giotto
    display_name: Giotto
    language: r
    env_name: r-cci-giotto
    runner: runners/r/run_external_cci_method.R
""",
        encoding="utf-8",
    )
    monkeypatch.chdir(repo)

    matrix = build_a100_job_matrix(
        benchmark_root=benchmark,
        remote_root="/data/taobo.hu/test",
        methods=["commot", "cellnest", "cellagentchat", "giotto"],
        phase="full",
        database_mode="common-db",
        max_cci_pairs=25,
        commot_chunks=4,
        cellagentchat_chunks=4,
        gpu_count=8,
        include_bootstrap=True,
        include_audit=True,
    )

    bootstrap = next(job for job in matrix["jobs"] if job["job_id"] == "bootstrap_pyx_cci_commot")
    commot_jobs = [job for job in matrix["jobs"] if job["method"] == "commot" and job["job_type"] == "method_run"]
    cellnest_job = next(job for job in matrix["jobs"] if job["method"] == "cellnest" and job["job_type"] == "method_run")
    cellagentchat_jobs = [job for job in matrix["jobs"] if job["method"] == "cellagentchat" and job["job_type"] == "method_run"]
    cellagentchat_aggregate = next(job for job in matrix["jobs"] if job["job_id"] == "full_common_db_cellagentchat_aggregate")
    giotto_audit = next(job for job in matrix["jobs"] if job["job_id"] == "audit_giotto")

    assert matrix["path_policy"]["valid"] is True
    assert matrix["include_bootstrap"] is True
    assert bootstrap["job_type"] == "env_bootstrap"
    assert "conda env create" in bootstrap["command"]
    assert "bootstrap_env.py" in bootstrap["command"]
    assert len(commot_jobs) == 4
    assert commot_jobs[0]["chunk_id"] == 0
    assert commot_jobs[0]["num_chunks"] == 4
    assert "--chunk-id 0" in commot_jobs[0]["command"]
    assert "--num-chunks 4" in commot_jobs[0]["command"]
    assert "--gzip-standardized" in commot_jobs[0]["command"]
    assert "OMP_NUM_THREADS=8" in commot_jobs[0]["command"]
    assert cellnest_job["gpu_id"] == 0
    assert "CUDA_VISIBLE_DEVICES=0" in cellnest_job["command"]
    assert "source_checkout_ok cellnest already at" in cellnest_job["command"]
    assert len(cellagentchat_jobs) == 4
    assert {job["chunk_id"] for job in cellagentchat_jobs} == {0, 1, 2, 3}
    assert all(job["num_chunks"] == 4 for job in cellagentchat_jobs)
    assert "--chunk-id 0" in cellagentchat_jobs[0]["command"]
    assert "--num-chunks 4" in cellagentchat_jobs[0]["command"]
    assert "CELLAGENTCHAT_FEATURE_SELECTION" in cellagentchat_jobs[0]["command"]
    assert "source_checkout_ok cellagentchat already at" in cellagentchat_jobs[0]["command"]
    assert cellagentchat_jobs[0]["gpu_id"] == 1
    assert cellagentchat_aggregate["job_type"] == "method_aggregate"
    assert cellagentchat_aggregate["output_dir"] == "/data/taobo.hu/test/runs/full_common/cellagentchat"
    assert set(cellagentchat_aggregate["dependencies"]) == {
        "full_common_db_cellagentchat_chunk_000_of_004",
        "full_common_db_cellagentchat_chunk_001_of_004",
        "full_common_db_cellagentchat_chunk_002_of_004",
        "full_common_db_cellagentchat_chunk_003_of_004",
    }
    assert "cellagentchat_standardized.tsv.gz" in cellagentchat_aggregate["command"]
    assert "Rscript" in giotto_audit["command"]
    assert "renv::load" in giotto_audit["command"]
    assert "intToUtf8(10)" in giotto_audit["command"]
    assert "chr(10)" not in giotto_audit["command"]
    assert "'/n'" not in giotto_audit["command"]


def test_build_a100_sidecar_matrix_uses_distinct_outputs_and_pilot_manifest(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    benchmark = repo / "benchmarking" / "cci_2026_atera"
    (benchmark / "configs").mkdir(parents=True)
    (repo / "src").mkdir()
    (repo / "pyproject.toml").write_text("[project]\nname='toy'\n", encoding="utf-8")
    (benchmark / "configs" / "methods.yaml").write_text(
        """methods:
  - slug: spatialdm
    display_name: SpatialDM
    language: python
    env_name: pyx-cci-spatialdm
    runner: scripts/run_method.py
  - slug: cellnest
    display_name: CellNEST
    language: python
    env_name: pyx-cci-cellnest
    runner: scripts/run_method.py
  - slug: giotto
    display_name: Giotto
    language: r
    env_name: r-cci-giotto
    runner: runners/r/run_external_cci_method.R
""",
        encoding="utf-8",
    )
    monkeypatch.chdir(repo)

    matrix = build_a100_sidecar_matrix(
        benchmark_root=benchmark,
        remote_root="/data/taobo.hu/test",
        methods=["spatialdm", "cellnest", "giotto"],
        ready_methods=["spatialdm", "cellnest"],
        completed_job_ids=["sidecar_audit_spatialdm"],
        smoke_max_cci_pairs=25,
        pilot_max_cci_pairs=100,
        gpu_start=3,
    )

    job_ids = {job["job_id"] for job in matrix["jobs"]}
    smoke = next(job for job in matrix["jobs"] if job["job_id"] == "sidecar_smoke_common_db_spatialdm")
    pilot = next(job for job in matrix["jobs"] if job["job_id"] == "sidecar_pilot50k_common_db_spatialdm")
    gpu_smoke = next(job for job in matrix["jobs"] if job["job_id"] == "sidecar_smoke_common_db_cellnest")

    assert matrix["kind"] == "a100_sidecar_matrix"
    assert matrix["ready_methods"] == ["spatialdm", "cellnest"]
    assert "sidecar_audit_spatialdm" not in job_ids
    assert "smoke_sidecar_common" in smoke["output_dir"]
    assert "/data/taobo.hu/test/data/input_manifest.json" in smoke["command"]
    assert "pilot50k_common" in pilot["output_dir"]
    assert "/data/taobo.hu/test/data/pilot50k/input_manifest.json" in pilot["command"]
    prep = next(job for job in matrix["jobs"] if job["job_id"] == "prepare_pilot50k_bundle")
    assert 'chr(10)' in prep["command"]
    assert '"/n"' not in prep["command"]
    assert gpu_smoke["gpu_id"] == 3
    assert "CUDA_VISIBLE_DEVICES=3" in gpu_smoke["command"]
    assert matrix["path_policy"]["valid"] is True


def test_finalize_cci_source_of_truth_prefers_a100_and_requires_commot_chunks(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    benchmark = repo / "benchmarking" / "cci_2026_atera"
    (benchmark / "configs").mkdir(parents=True)
    (benchmark / "runs").mkdir(parents=True)
    (benchmark / "results").mkdir(parents=True)
    (benchmark / "reports").mkdir(parents=True)
    (repo / "pyproject.toml").write_text("[project]\nname='toy'\n", encoding="utf-8")
    monkeypatch.chdir(repo)

    _write_standardized_result(benchmark / "runs" / "a100_collected" / "full_common" / "pyxenium" / "pyxenium_standardized.tsv", method="pyxenium")
    _write_standardized_result(
        benchmark / "pdc_collected" / "pdc_old" / "runs" / "full_common" / "pyxenium" / "pyxenium_standardized.tsv",
        method="pyxenium",
        ligand="OLD",
        receptor="OLD",
    )
    _write_standardized_result(
        benchmark / "pdc_collected" / "pdc_new" / "runs" / "full_common" / "cellchat" / "cellchat_standardized.tsv",
        method="cellchat",
    )
    _write_standardized_result(
        benchmark / "pdc_collected" / "pdc_new" / "runs" / "full_common" / "cellagentchat" / "cellagentchat_standardized.tsv",
        method="cellagentchat",
    )
    _write_standardized_result(
        benchmark / "pdc_collected" / "pdc_new" / "runs" / "full_common" / "commot" / "chunk_000_of_002" / "commot_standardized.tsv",
        method="commot",
    )
    _write_standardized_result(
        benchmark / "pdc_collected" / "pdc_new" / "runs" / "full_common" / "commot" / "chunk_001_of_002" / "commot_standardized.tsv",
        method="commot",
        ligand="DLL4",
        receptor="NOTCH3",
    )

    payload = finalize_cci_source_of_truth(
        benchmark_root=benchmark,
        pdc_tag="pdc_new",
        commot_chunks=2,
        render_report=False,
    )

    assert payload["methods"] == ["cellagentchat", "cellchat", "commot", "pyxenium"]
    selected_paths = payload["selection"]["input_paths"]
    assert any("a100_collected" in path and "pyxenium" in path for path in selected_paths)
    assert not any("pdc_old" in path for path in selected_paths)
    assert len([path for path in selected_paths if "commot" in path]) == 2
    assert payload["validation"]["status"] == "passed"
    combined = pd.read_csv(payload["outputs"]["combined_results"], sep="\t")
    assert set(combined["method"]) == {"pyxenium", "cellchat", "commot", "cellagentchat"}


def test_a100_path_policy_flags_write_targets_in_readonly_root():
    policy = validate_a100_path_policy(
        {
            "jobs": [
                {
                    "job_id": "bad",
                    "input_root": DEFAULT_A100_READONLY_XENIUM_ROOT,
                    "output_dir": f"{DEFAULT_A100_READONLY_XENIUM_ROOT}/bad_output",
                }
            ]
        },
        remote_root="/data/taobo.hu/test",
        readonly_xenium_root=DEFAULT_A100_READONLY_XENIUM_ROOT,
    )

    assert policy["valid"] is False
    assert policy["readonly_inputs"] == [DEFAULT_A100_READONLY_XENIUM_ROOT]
    assert any("read-only Xenium root" in issue for issue in policy["violations"])


def test_prepare_a100_bundle_reports_blocked_without_full_bundle(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    benchmark = tmp_path / "benchmarking" / "cci_2026_atera"
    (benchmark / "configs").mkdir(parents=True)
    (benchmark / "envs").mkdir()
    (benchmark / "scripts").mkdir()
    (benchmark / "runners").mkdir()
    (benchmark / "data").mkdir()
    (benchmark / "logs").mkdir()
    (tmp_path / "src").mkdir()
    (tmp_path / "pyproject.toml").write_text("[project]\nname='toy'\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("toy\n", encoding="utf-8")
    (tmp_path / "LICENSE").write_text("toy\n", encoding="utf-8")
    (benchmark / "configs" / "methods.yaml").write_text(
        """methods:
  - slug: pyxenium
    display_name: pyXenium
    language: python
    env_name: pyx-cci-pyxenium
    runner: runners/python/run_pyxenium.py
""",
        encoding="utf-8",
    )
    cci_resource = benchmark / "data" / "cci.tsv"
    cci_resource.write_text("ligand\treceptor\nCXCL12\tCXCR4\n", encoding="utf-8")
    (benchmark / "data" / "input_manifest.json").write_text(
        json.dumps({"cci_resource_common_tsv": str(cci_resource), "smoke_h5ad": str(benchmark / "data" / "smoke.h5ad"), "full_bundle": {}}),
        encoding="utf-8",
    )

    payload = prepare_a100_bundle(
        benchmark_root=benchmark,
        methods=["pyxenium"],
        remote_xenium_root=None,
        include_prepare=False,
        output_json=benchmark / "logs" / "plan.json",
    )

    assert payload["status"] == "blocked"
    assert Path(benchmark / "logs" / "plan.json").exists()


def test_prepare_a100_bundle_can_plan_remote_prepare_when_full_bundle_is_missing(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    benchmark = tmp_path / "benchmarking" / "cci_2026_atera"
    (benchmark / "configs").mkdir(parents=True)
    (benchmark / "envs").mkdir()
    (benchmark / "scripts").mkdir()
    (benchmark / "runners").mkdir()
    (benchmark / "data").mkdir()
    (benchmark / "logs").mkdir()
    (tmp_path / "src").mkdir()
    (tmp_path / "pyproject.toml").write_text("[project]\nname='toy'\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("toy\n", encoding="utf-8")
    (tmp_path / "LICENSE").write_text("toy\n", encoding="utf-8")
    (benchmark / "configs" / "methods.yaml").write_text(
        """methods:
  - slug: pyxenium
    display_name: pyXenium
    language: python
    env_name: pyx-cci-pyxenium
    runner: runners/python/run_pyxenium.py
""",
        encoding="utf-8",
    )
    cci_resource = benchmark / "data" / "cci.tsv"
    smoke_h5ad = benchmark / "data" / "smoke.h5ad"
    cci_resource.write_text("ligand\treceptor\nCXCL12\tCXCR4\n", encoding="utf-8")
    smoke_h5ad.touch()
    (benchmark / "data" / "input_manifest.json").write_text(
        json.dumps({"cci_resource_common_tsv": str(cci_resource), "smoke_h5ad": str(smoke_h5ad), "full_bundle": {}}),
        encoding="utf-8",
    )

    payload = prepare_a100_bundle(
        benchmark_root=benchmark,
        remote_root="/data/taobo.hu/test",
        transfer_mode="tar-scp",
        methods=["pyxenium"],
        output_json=benchmark / "logs" / "plan.json",
    )

    assert payload["status"] == "ready_after_remote_prepare"
    assert payload["remote_prepare_required"] is True
    assert payload["path_policy"]["valid"] is True
    assert payload["stage_plan"]["transfer_mode"] == "tar-scp"
    assert payload["job_manifest"]["jobs"][0]["job_id"] == "prepare_full_bundle"


def test_execute_a100_stage_plan_tar_scp_dry_run(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    benchmark = repo / "benchmarking" / "cci_2026_atera"
    (benchmark / "configs").mkdir(parents=True)
    (benchmark / "envs").mkdir()
    (benchmark / "scripts").mkdir()
    (benchmark / "runners").mkdir()
    (repo / "src").mkdir()
    (repo / "pyproject.toml").write_text("[project]\nname='toy'\n", encoding="utf-8")
    (repo / "README.md").write_text("toy\n", encoding="utf-8")
    (repo / "LICENSE").write_text("toy\n", encoding="utf-8")
    monkeypatch.chdir(repo)

    stage = build_a100_stage_plan(
        benchmark_root=benchmark,
        remote_root="/data/taobo.hu/test",
        host="sscb-a100.scilifelab.se",
        user="taobo.hu",
        transfer_mode="tar-scp",
    )
    payload = execute_a100_stage_plan(stage_plan=stage, dry_run=True)

    assert payload["transfer_mode"] == "tar-scp"
    assert [step["step"] for step in payload["steps"]] == ["mkdir", "pack", "upload", "extract"]


def test_run_a100_plan_remote_dry_run_exposes_wrapper_command(tmp_path):
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "kind": "a100_job_manifest",
                "jobs": [
                    {
                        "job_id": "full_common_db_pyxenium",
                        "command": "echo hello",
                        "stdout": "/data/taobo.hu/test/logs/stdout.log",
                        "stderr": "/data/taobo.hu/test/logs/stderr.log",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    payload = run_a100_plan(
        plan_json=plan,
        dry_run=True,
        remote=True,
        host="sscb-a100.scilifelab.se",
        user="taobo.hu",
    )

    assert payload["remote"] is True
    assert payload["jobs"][0]["status"] == "dry-run"
    assert payload["jobs"][0]["wrapper_command"].startswith("ssh taobo.hu@sscb-a100.scilifelab.se ")


def test_collect_a100_results_scp_dry_run(tmp_path, monkeypatch):
    benchmark = tmp_path / "benchmarking" / "cci_2026_atera"
    (tmp_path / "pyproject.toml").write_text("[project]\nname='toy'\n", encoding="utf-8")
    (benchmark / "configs").mkdir(parents=True)
    (benchmark / "results").mkdir()
    (benchmark / "logs").mkdir()
    (benchmark / "runs").mkdir()
    (benchmark / "reports").mkdir()
    monkeypatch.chdir(tmp_path)

    payload = collect_a100_results(
        benchmark_root=benchmark,
        remote_root="/data/taobo.hu/test",
        host="sscb-a100.scilifelab.se",
        user="taobo.hu",
        transfer_mode="scp",
        dry_run=True,
    )

    assert payload["transfer_mode"] == "scp"
    assert len(payload["commands"]) == 4
    assert all(command.startswith('scp -r taobo.hu@sscb-a100.scilifelab.se:"/data/taobo.hu/test/') for command in payload["commands"])


def test_collect_a100_results_records_since_last_flag(tmp_path, monkeypatch):
    benchmark = tmp_path / "benchmarking" / "cci_2026_atera"
    (tmp_path / "pyproject.toml").write_text("[project]\nname='toy'\n", encoding="utf-8")
    (benchmark / "configs").mkdir(parents=True)
    (benchmark / "results").mkdir()
    (benchmark / "logs").mkdir()
    (benchmark / "runs").mkdir()
    (benchmark / "reports").mkdir()
    monkeypatch.chdir(tmp_path)

    payload = collect_a100_results(
        benchmark_root=benchmark,
        remote_root="/data/taobo.hu/test",
        transfer_mode="scp",
        dry_run=True,
        since_last=True,
    )

    assert payload["since_last"] is True


def test_submit_monitor_and_finalize_a100_matrix_dry_run(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    benchmark = repo / "benchmarking" / "cci_2026_atera"
    (benchmark / "configs").mkdir(parents=True)
    (benchmark / "runs" / "a100_collected" / "full_common" / "laris").mkdir(parents=True)
    (benchmark / "results").mkdir(parents=True)
    (repo / "src").mkdir()
    (repo / "pyproject.toml").write_text("[project]\nname='toy'\n", encoding="utf-8")
    (benchmark / "configs" / "methods.yaml").write_text(
        """methods:
  - slug: laris
    display_name: LARIS
    language: python
    env_name: pyx-cci-laris
    runner: scripts/run_method.py
""",
        encoding="utf-8",
    )
    monkeypatch.chdir(repo)
    matrix = build_a100_job_matrix(
        benchmark_root=benchmark,
        remote_root="/data/taobo.hu/test",
        methods=["laris"],
        phase="full",
        database_mode="common-db",
        max_cci_pairs=3,
    )
    matrix_json = benchmark / "logs" / "matrix.json"
    matrix_json.parent.mkdir(parents=True, exist_ok=True)
    matrix_json.write_text(json.dumps(matrix, indent=2), encoding="utf-8")

    submit = submit_a100_matrix(
        matrix_json=matrix_json,
        dry_run=True,
        remote=True,
        host="sscb-a100.scilifelab.se",
        user="taobo.hu",
        job_types=["method_run"],
    )
    assert submit["jobs"][0]["status"] == "dry-run"
    assert all(job["job_type"] == "method_run" for job in submit["jobs"])
    assert "nohup bash -lc" in submit["jobs"][0]["background_command"]

    standardized = standardize_result_table(
        pd.DataFrame(
            [
                {
                    "ligand": "CXCL12",
                    "receptor": "CXCR4",
                    "sender_celltype": "A",
                    "receiver_celltype": "B",
                    "CCI_score": 1.0,
                }
            ]
        ),
        method="laris",
        database_mode="common-db",
        spatial_support_type="knn_diffusion_lr",
    )
    run_dir = benchmark / "runs" / "a100_collected" / "full_common" / "laris"
    standardized_path = run_dir / "standardized.tsv.gz"
    standardized.to_csv(standardized_path, sep="\t", index=False, compression="gzip")
    (run_dir / "run_summary.json").write_text(
        json.dumps(
            {
                "method": "laris",
                "phase": "full",
                "database_mode": "common-db",
                "status": "success",
                "n_rows": 1,
                "standardized_tsv": str(standardized_path),
            }
        ),
        encoding="utf-8",
    )

    status_tsv = benchmark / "results" / "a100_job_status.tsv"
    monitored = monitor_a100_jobs(matrix_json=matrix_json, benchmark_root=benchmark, output_tsv=status_tsv)
    finalized = finalize_a100_all(benchmark_root=benchmark, output_path=benchmark / "results" / "a100_all.tsv")

    assert status_tsv.exists()
    assert monitored.iloc[0]["observed_status"] == "success"
    assert finalized["n_rows"] == 1
    assert finalized["n_methods"] == 1


def test_summarize_run_status_and_engineering_summary_and_method_card(tmp_path):
    run_dir = tmp_path / "runs" / "method"
    run_dir.mkdir(parents=True)
    summary = {
        "method": "squidpy",
        "phase": "full",
        "database_mode": "common-db",
        "status": "failed",
        "elapsed_seconds": 12.5,
        "exit_code": 1,
        "error": "boom",
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary), encoding="utf-8")

    status = summarize_run_status(tmp_path / "runs")
    engineering = build_engineering_summary(status)
    card = write_failed_method_card(run_dir, summary)

    assert status.iloc[0]["method"] == "squidpy"
    assert engineering.iloc[0]["status"] == "failed"
    assert card.exists()


def test_render_report_includes_run_status_and_canonical_rank_matrix():
    combined = standardize_result_table(
        pd.DataFrame(
            [
                {
                    "ligand": "CXCL12",
                    "receptor": "CXCR4",
                    "sender_celltype": "CAFs, DCIS Associated",
                    "receiver_celltype": "T Lymphocytes",
                    "CCI_score": 1.0,
                }
            ]
        ),
        method="pyxenium",
        database_mode="common-db",
    )
    canonical_axes = [
        {
            "name": "cxcl12_axis",
            "ligand": "CXCL12",
            "receptor": "CXCR4",
            "sender": "CAFs, DCIS Associated",
            "receiver": "T Lymphocytes",
        }
    ]
    canonical_detail, canonical_summary = compute_canonical_recovery(combined, canonical_axes=canonical_axes)
    _, pathway_summary = compute_pathway_relevance(combined, pathway_config={"CXCL12": {"pairs": ["CXCL12^CXCR4"], "genes": []}})
    biology = score_biological_performance(canonical_summary=canonical_summary, pathway_summary=pathway_summary)
    run_status = pd.DataFrame([{"method": "pyxenium", "phase": "full", "database_mode": "common-db", "status": "success", "n_rows": 1}])

    report = render_atera_cci_benchmark_report(
        combined_results=combined,
        canonical_summary=canonical_summary,
        pathway_summary=pathway_summary,
        biology_summary=biology,
        benchmark_root="benchmark",
        run_status=run_status,
        engineering_summary=run_status,
        canonical_detail=canonical_detail,
    )

    assert "## Run Status" in report
    assert "## Engineering Reproducibility" in report
    assert "## Canonical Pair Rank Matrix" in report
