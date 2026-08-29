"""Build the cached graph dataset for every SimJEB model.

Runs once, on Kaggle, and writes one ``.pt`` per model (~6 MB each, ~2.4 GB total).
Everything afterwards -- every training run, every ablation -- loads those files and
never touches the raw archives again.

Two properties matter more than speed here:

**Restartable.** 381 models at ~3 s each is roughly 20 minutes, comfortably inside a
session, but a failure at model 300 should not discard the first 299. Existing outputs
are skipped, so re-running resumes.

**Fails loudly, per model.** A model that cannot be built is recorded with its reason
and the run continues. A pipeline that silently drops samples is how a dataset quietly
becomes something other than what you think you are training on.
"""

from __future__ import annotations

import json
import shutil
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from src.data.build_graph import GraphBuildError, build_graph, save_graph
from src.data.fetch import SimJEBSource
from src.data.parse_fem import parse_fem


@dataclass
class BuildReport:
    """What happened to every model, kept as the audit trail for the QA stage."""

    built: list[int] = field(default_factory=list)
    skipped: list[int] = field(default_factory=list)      # already present
    failed: dict[str, str] = field(default_factory=dict)  # model id -> reason
    stats: dict[str, dict] = field(default_factory=dict)  # model id -> summary
    seconds: float = 0.0

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))
        return path


def build_one(source: SimJEBSource, model_id: int, out_dir: Path,
              scratch: Path, load_set: int = 2) -> dict:
    """Materialise one model's files, build its graph, then clean up.

    The files are written to disk and deleted immediately because the parsers work on
    paths, and holding 381 models' worth of raw data would need 20+ GB.
    """
    model_scratch = scratch / str(model_id)
    try:
        paths = source.extract_model(model_id, model_scratch)
        deck = parse_fem(paths["fem"])
        graph = build_graph(
            model_id=model_id,
            vtk_path=paths["vtk"],
            csv_path=paths["csv"],
            fem_path=paths["fem"],
            load_set=load_set,
            deck=deck,
        )
        save_graph(graph, out_dir)
        return graph["meta"]
    finally:
        shutil.rmtree(model_scratch, ignore_errors=True)


def make_dataset(archive_dir: str | Path, out_dir: str | Path,
                 scratch_dir: str | Path = "/kaggle/temp/simjeb",
                 model_ids: list[int] | None = None,
                 limit: int | None = None,
                 overwrite: bool = False) -> BuildReport:
    """Build cached graphs for every model found in the archives.

    Parameters
    ----------
    archive_dir
        Where the downloaded ``.zip`` files live.
    out_dir
        Destination for the ``<model_id>.pt`` files.
    scratch_dir
        Temporary space for one model at a time. On Kaggle, ``/kaggle/temp`` is
        larger than the working directory and is not saved as output.
    limit
        Stop after this many models -- for a quick end-to-end check before committing
        a full run.
    overwrite
        Rebuild models whose output already exists. Off by default so a re-run
        resumes rather than repeating.
    """
    out_dir, scratch = Path(out_dir), Path(scratch_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)

    source = SimJEBSource.open(archive_dir)
    ids = model_ids if model_ids is not None else source.model_ids()
    if limit:
        ids = ids[:limit]

    report = BuildReport()
    began = time.time()

    for n, model_id in enumerate(ids, start=1):
        target = out_dir / f"{model_id}.pt"
        if target.exists() and not overwrite:
            report.skipped.append(model_id)
            continue

        try:
            meta = build_one(source, model_id, out_dir, scratch)
            report.built.append(model_id)
            report.stats[str(model_id)] = {
                key: meta[key] for key in (
                    "n_mesh_nodes", "n_surface_nodes", "n_tets",
                    "n_edges_undirected", "surface_fraction",
                    "n_fixed_nodes", "n_loaded_nodes",
                )
            }
        except (GraphBuildError, Exception) as error:  # noqa: B014 - intent is explicit
            # One bad model must not end the run, but it must be recorded. Silent
            # drops are how a dataset stops being what you think it is.
            report.failed[str(model_id)] = (
                str(error) if isinstance(error, GraphBuildError)
                else f"{type(error).__name__}: {error}"
            )
            if not isinstance(error, GraphBuildError):
                traceback.print_exc()

        if n % 25 == 0 or n == len(ids):
            rate = (time.time() - began) / max(len(report.built), 1)
            remaining = (len(ids) - n) * rate / 60
            print(f"{n}/{len(ids)}  built {len(report.built)}  "
                  f"failed {len(report.failed)}  ~{remaining:.0f} min left", flush=True)

    report.seconds = time.time() - began
    print(f"\nbuilt {len(report.built)}, skipped {len(report.skipped)}, "
          f"failed {len(report.failed)} in {report.seconds / 60:.1f} min")
    if report.failed:
        print("failures:")
        for model_id, reason in report.failed.items():
            print(f"  {model_id}: {reason}")
    return report


def dataset_summary(graph_dir: str | Path) -> dict:
    """Aggregate figures over the built cache -- a cheap sanity check before training."""
    graph_dir = Path(graph_dir)
    files = sorted(graph_dir.glob("*.pt"))
    if not files:
        raise FileNotFoundError(f"no cached graphs in {graph_dir}")

    nodes, edges, sizes = [], [], []
    for path in files:
        cached = torch.load(path, weights_only=False)
        nodes.append(cached["meta"]["n_surface_nodes"])
        edges.append(cached["meta"]["n_edges_undirected"])
        sizes.append(path.stat().st_size)

    return {
        "n_models": len(files),
        "surface_nodes": {"min": min(nodes), "max": max(nodes),
                          "mean": sum(nodes) / len(nodes), "total": sum(nodes)},
        "undirected_edges": {"min": min(edges), "max": max(edges),
                             "total": sum(edges)},
        "disk_gb": round(sum(sizes) / 1e9, 2),
    }
