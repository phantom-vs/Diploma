from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from preproc.manifest import build_run_manifest, write_manifest_yaml
from preproc.registry import register_postprocessor


@register_postprocessor("depd_intervals_csv")
def run_depd_intervals_csv(
    ctx: dict[str, Any], params: dict[str, Any], out_dir: Path
) -> list[Path]:
    """CSV: channel, start_seconds, end_seconds (номера каналов как в разметке, с 1)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    basename = params.get("output_basename", "depd_intervals.csv")
    csv_path = out_dir / basename

    intervals = ctx.get("depd_intervals") or []
    if intervals:
        df = pd.DataFrame(intervals)
        df["start_seconds"] = df["start"]
        df["end_seconds"] = df["end"]
        out_df = df[["channel", "start_seconds", "end_seconds"]].sort_values(
            "start_seconds"
        )
        out_df.to_csv(csv_path, index=False, encoding="utf-8")
    else:
        pd.DataFrame(
            columns=["channel", "start_seconds", "end_seconds"]
        ).to_csv(csv_path, index=False, encoding="utf-8")

    manifest_path = out_dir / "manifest.yaml"
    write_manifest_yaml(
        manifest_path,
        build_run_manifest(
            ctx,
            params,
            kind="depd_intervals_csv",
            artifacts=[csv_path.name],
        ),
    )
    return [csv_path, manifest_path]
