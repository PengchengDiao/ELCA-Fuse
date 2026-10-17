"""Command-line evaluation for SA-LCID."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
from PIL import Image
from scipy.stats import spearmanr
from tqdm.auto import tqdm

from .core import (
    DiagnosticConfig,
    diagnose,
    inject_chroma_noise,
    inject_ir_false_colour,
    reconstruct_with_fixed_luminance,
    relative_luminance,
    rotate_oklab_hue,
    scale_oklab_chroma,
)


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
SCALAR_KEYS = [
    "provisional_artifact_index",
    "visible_hue_damage",
    "ir_structure_chroma_leakage",
    "high_frequency_chroma_noise",
    "chroma_attenuation_penalty",
    "chroma_overshoot_penalty",
    "fixed_luminance_mae",
    "visible_reliable_weight",
    "ir_only_weight",
    "mean_log_chroma_response",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sa-lcid",
        description=(
            "Evaluate registered visible/infrared/fused images with the "
            "source-aware luminance--chroma intervention diagnostic."
        ),
    )
    parser.add_argument("--visible-dir", type=Path, required=True)
    parser.add_argument("--infrared-dir", type=Path, required=True)
    parser.add_argument(
        "--method-root",
        type=Path,
        required=True,
        help="Directory containing one subdirectory per fusion method.",
    )
    parser.add_argument(
        "--fixed-luminance-method",
        required=True,
        help="Method subdirectory whose luminance is shared by all methods.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--include-methods",
        nargs="+",
        default=None,
        help="Optional method names. Default: all subdirectories.",
    )
    parser.add_argument("--analysis-max-side", type=int, default=384)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--synthetic-samples", type=int, default=8)
    parser.add_argument("--no-synthetic", action="store_true")
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--day-suffix", default="D")
    parser.add_argument("--night-suffix", default="N")
    return parser


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0


def load_gray(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.float64) / 255.0


def analysis_shape(height: int, width: int, maximum: int) -> tuple[int, int]:
    if maximum <= 0 or max(height, width) <= maximum:
        return height, width
    scale = maximum / float(max(height, width))
    return max(1, round(height * scale)), max(1, round(width * scale))


def resize_to(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if image.shape[:2] == shape:
        return image
    height, width = shape
    encoded = Image.fromarray(np.uint8(np.clip(image, 0.0, 1.0) * 255.0))
    resampling = getattr(Image, "Resampling", Image)
    resized = encoded.resize((width, height), resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float64) / 255.0


def discover_methods(
    method_root: Path,
    include_methods: Sequence[str] | None,
    fixed_luminance_method: str,
) -> Dict[str, Path]:
    if not method_root.is_dir():
        raise FileNotFoundError(f"Method root does not exist: {method_root}")
    allowed = None if include_methods is None else set(include_methods)
    methods = {
        path.name: path
        for path in sorted(method_root.iterdir())
        if path.is_dir() and (allowed is None or path.name in allowed)
    }
    if not methods:
        raise RuntimeError(f"No method directories found under {method_root}")
    if fixed_luminance_method not in methods:
        raise RuntimeError(
            "The fixed-luminance method must be included in the evaluation: "
            f"{fixed_luminance_method}"
        )
    return methods


def common_names(
    visible_dir: Path,
    infrared_dir: Path,
    methods: Dict[str, Path],
    limit: int | None,
) -> List[str]:
    for path in (visible_dir, infrared_dir):
        if not path.is_dir():
            raise FileNotFoundError(f"Input directory does not exist: {path}")
    names = [
        path.name
        for path in sorted(visible_dir.iterdir())
        if path.is_file()
        and path.suffix.lower() in IMAGE_SUFFIXES
        and (infrared_dir / path.name).is_file()
        and all((directory / path.name).is_file() for directory in methods.values())
    ]
    if limit is not None:
        names = names[: max(0, limit)]
    return names


def write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    values = list(rows)
    if not values:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(values[0].keys()))
        writer.writeheader()
        writer.writerows(values)


def aggregate(rows: List[Dict[str, object]], split: bool) -> List[Dict[str, object]]:
    groups = sorted(
        {
            (str(row["method"]), str(row["stratum"]) if split else "all")
            for row in rows
        }
    )
    output: List[Dict[str, object]] = []
    for method, stratum in groups:
        selected = [
            row
            for row in rows
            if row["method"] == method
            and (not split or row["stratum"] == stratum)
        ]
        item: Dict[str, object] = {
            "method": method,
            "stratum": stratum,
            "images": len(selected),
        }
        for key in SCALAR_KEYS:
            values = np.asarray([float(row[key]) for row in selected])
            item[key] = float(np.nanmean(values))
        output.append(item)
    return output


def stratum(name: str, day_suffix: str, night_suffix: str) -> str:
    stem = Path(name).stem
    if day_suffix and stem.endswith(day_suffix):
        return "day"
    if night_suffix and stem.endswith(night_suffix):
        return "night"
    return "other"


def synthetic_case(
    case: str,
    severities: List[float],
    metric: str,
    visible: np.ndarray,
    infrared: np.ndarray,
    seed: int,
    config: DiagnosticConfig,
) -> Dict[str, object]:
    values: List[float] = []
    for severity in severities:
        if case == "hue_rotation":
            candidate = rotate_oklab_hue(visible, severity)
        elif case in {"attenuation", "overshoot"}:
            candidate = scale_oklab_chroma(visible, severity)
        elif case == "ir_false_colour":
            candidate = inject_ir_false_colour(
                visible, visible, infrared, severity, config
            )
        elif case == "chroma_noise":
            candidate = inject_chroma_noise(
                visible, severity, np.random.default_rng(seed)
            )
        else:
            raise ValueError(f"Unknown synthetic case: {case}")
        fixed_y = relative_luminance(visible)
        controlled = reconstruct_with_fixed_luminance(candidate, fixed_y)
        values.append(
            float(
                diagnose(
                    visible,
                    infrared,
                    controlled,
                    fixed_luminance=fixed_y,
                    config=config,
                )[metric]
            )
        )
    expected = [-value for value in severities] if case == "attenuation" else severities
    correlation = float(spearmanr(expected, values).statistic)
    return {
        "case": case,
        "metric": metric,
        "severities": severities,
        "values": values,
        "spearman": correlation,
        "passed": bool(
            np.isfinite(correlation) and correlation >= 0.90 - 1e-12
        ),
    }


def run(args: argparse.Namespace) -> Path:
    methods = discover_methods(
        args.method_root, args.include_methods, args.fixed_luminance_method
    )
    names = common_names(args.visible_dir, args.infrared_dir, methods, args.limit)
    if not names:
        raise RuntimeError("No common image names were found across all directories")

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    config = DiagnosticConfig()
    rows: List[Dict[str, object]] = []

    progress = tqdm(names, desc=f"SA-LCID ({len(methods)} methods)", unit="image")
    for name in progress:
        progress.set_postfix_str(name, refresh=False)
        visible_full = load_rgb(args.visible_dir / name)
        target_shape = analysis_shape(
            visible_full.shape[0], visible_full.shape[1], args.analysis_max_side
        )
        visible = resize_to(visible_full, target_shape)
        infrared = resize_to(load_gray(args.infrared_dir / name), target_shape)
        reference = resize_to(
            load_rgb(methods[args.fixed_luminance_method] / name), target_shape
        )
        fixed_y = relative_luminance(reference)
        for method, directory in methods.items():
            fused = resize_to(load_rgb(directory / name), target_shape)
            controlled = reconstruct_with_fixed_luminance(fused, fixed_y)
            result = diagnose(
                visible,
                infrared,
                controlled,
                fixed_luminance=fixed_y,
                config=config,
            )
            rows.append(
                {
                    "image": name,
                    "stratum": stratum(name, args.day_suffix, args.night_suffix),
                    "method": method,
                    **{key: result[key] for key in SCALAR_KEYS},
                }
            )

    synthetic: List[Dict[str, object]] = []
    if not args.no_synthetic and args.synthetic_samples > 0:
        for sample_index, name in enumerate(
            tqdm(
                names[: args.synthetic_samples],
                desc="Synthetic monotonicity",
                unit="image",
            )
        ):
            visible_full = load_rgb(args.visible_dir / name)
            target_shape = analysis_shape(
                visible_full.shape[0], visible_full.shape[1], args.analysis_max_side
            )
            visible = resize_to(visible_full, target_shape)
            infrared = resize_to(load_gray(args.infrared_dir / name), target_shape)
            cases = [
                ("hue_rotation", [0, 5, 15, 30, 60], "visible_hue_damage"),
                (
                    "attenuation",
                    [1.0, 0.9, 0.75, 0.5, 0.25],
                    "chroma_attenuation_penalty",
                ),
                (
                    "overshoot",
                    [1.0, 1.25, 1.5, 2.0, 3.0],
                    "chroma_overshoot_penalty",
                ),
                (
                    "ir_false_colour",
                    [0, 0.01, 0.03, 0.06, 0.10],
                    "ir_structure_chroma_leakage",
                ),
                (
                    "chroma_noise",
                    [0, 0.002, 0.005, 0.01, 0.02],
                    "high_frequency_chroma_noise",
                ),
            ]
            for case, severities, metric in cases:
                item = synthetic_case(
                    case,
                    severities,
                    metric,
                    visible,
                    infrared,
                    args.seed + sample_index,
                    config,
                )
                item["image"] = name
                synthetic.append(item)

    summary = aggregate(rows, split=False)
    summary_stratified = aggregate(rows, split=True)
    write_csv(output_dir / "per_image.csv", rows)
    write_csv(output_dir / "summary.csv", summary)
    write_csv(output_dir / "summary_day_night.csv", summary_stratified)
    if synthetic:
        (output_dir / "synthetic_monotonicity.json").write_text(
            json.dumps(synthetic, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    run_config = {
        "visible_dir": str(args.visible_dir.resolve()),
        "infrared_dir": str(args.infrared_dir.resolve()),
        "method_root": str(args.method_root.resolve()),
        "methods": list(methods),
        "fixed_luminance_method": args.fixed_luminance_method,
        "images": len(names),
        "analysis_max_side": args.analysis_max_side,
        "seed": args.seed,
        "diagnostic_config": config.to_dict(),
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Output: {output_dir.resolve()}")
    print(f"Methods: {len(methods)}; images: {len(names)}")
    if synthetic:
        passed = sum(bool(item["passed"]) for item in synthetic)
        print(f"Synthetic checks: {passed}/{len(synthetic)}")
    for item in sorted(summary, key=lambda row: row["provisional_artifact_index"]):
        print(
            "{method}: aggregate={provisional_artifact_index:.6f}, "
            "hue={visible_hue_damage:.6f}, "
            "IR-leakage={ir_structure_chroma_leakage:.6f}, "
            "noise={high_frequency_chroma_noise:.6f}".format(**item)
        )
    return output_dir


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()

