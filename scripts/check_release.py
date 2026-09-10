"""Validate that the ELCA-Fuse release directory is source-only and parseable."""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {
    ".gitignore",
    "LICENSE",
    "README.md",
    "RELEASE_CHECKLIST.md",
    "requirements.txt",
    "data/README.md",
    "checkpoints/README.md",
    "infer_fusion.py",
    "fusion_model.py",
    "credibility/models/prob_fusion.py",
    "credibility/models/prob_fusion_calibrated.py",
    "chroma_polar_interpretable/model.py",
    "third_party/LYT/LICENSE",
}
FORBIDDEN_SUFFIXES = {
    ".pt",
    ".pth",
    ".ckpt",
    ".pkl",
    ".plk",
    ".onnx",
    ".safetensors",
    ".npy",
    ".npz",
    ".mat",
    ".xlsx",
    ".xls",
}
FORBIDDEN_DIRECTORIES = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".idea",
    ".vscode",
    "ICASSP2027",
    "outputs",
    "results",
}
ABSOLUTE_WINDOWS_PATH = re.compile(r"(?i)(?<![A-Za-z])[A-Z]:[\\/]")


def main() -> int:
    problems: list[str] = []
    files = [path for path in ROOT.rglob("*") if path.is_file()]

    present = {path.relative_to(ROOT).as_posix() for path in files}
    missing = sorted(REQUIRED - present)
    problems.extend(f"missing required file: {path}" for path in missing)

    for path in files:
        relative = path.relative_to(ROOT)
        if any(part in FORBIDDEN_DIRECTORIES for part in relative.parts):
            problems.append(f"forbidden generated/private directory: {relative}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            problems.append(f"forbidden data/weight artifact: {relative}")
        if path.suffix == ".py":
            source = path.read_text(encoding="utf-8-sig")
            try:
                ast.parse(source, filename=str(relative))
            except SyntaxError as error:
                problems.append(f"syntax error in {relative}: {error}")
            for line_number, line in enumerate(source.splitlines(), 1):
                if ABSOLUTE_WINDOWS_PATH.search(line):
                    problems.append(
                        f"machine-specific path in {relative}:{line_number}"
                    )

    if problems:
        print("Release audit failed:")
        for problem in problems:
            print(f"- {problem}")
        return 1

    print(
        f"Release audit passed: {len(files)} files, no weights/data/caches, "
        "all Python files parse successfully."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
