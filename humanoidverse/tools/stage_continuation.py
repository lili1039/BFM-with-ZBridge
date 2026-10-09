"""Stage an independent training work directory from a completed checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


REQUIRED_CHECKPOINT_FILES = (
    "train_status.json",
    "config.json",
    "init_kwargs.json",
    "optimizers.pth",
    "model/config.json",
    "model/init_kwargs.json",
    "model/model.safetensors",
)


def resolve_checkpoint(path: Path) -> tuple[Path, Path]:
    source = path.expanduser().resolve(strict=True)
    checkpoint = source if source.name == "checkpoint" else source / "checkpoint"
    if not checkpoint.is_dir():
        raise ValueError(f"Missing checkpoint directory: {checkpoint}")
    missing = [name for name in REQUIRED_CHECKPOINT_FILES if not (checkpoint / name).is_file()]
    if missing:
        raise ValueError(f"Incomplete checkpoint {checkpoint}: missing {missing}")
    buffers = checkpoint / "buffers"
    if not buffers.is_dir() or not any(buffers.rglob("buffer.hdf5")):
        raise ValueError(f"Missing replay buffer under {buffers}; full training resume requires it")
    with (checkpoint / "train_status.json").open() as file:
        status = json.load(file)
    if int(status.get("global_time", status.get("time", 0))) <= 0:
        raise ValueError("Checkpoint has no positive global_time")
    return checkpoint, source.parent if source.name == "checkpoint" else source


def stage_continuation(source: Path, work_dir: Path) -> Path:
    checkpoint, run_dir = resolve_checkpoint(source)
    work_dir = work_dir.expanduser().absolute()
    if work_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing work directory: {work_dir}")
    if work_dir == run_dir or run_dir in work_dir.parents or work_dir in run_dir.parents:
        raise ValueError("The continuation work directory must be separate from the source run")
    work_dir.parent.mkdir(parents=True, exist_ok=True)
    symlinks = [str(path.relative_to(checkpoint)) for path in checkpoint.rglob("*") if path.is_symlink()]
    if symlinks:
        raise ValueError(f"Checkpoint contains symlinks that could still point to the source: {symlinks}")
    source_state = {
        str(file.relative_to(checkpoint)): (file.stat().st_size, file.stat().st_mtime_ns)
        for file in checkpoint.rglob("*")
        if file.is_file()
    }
    staging = Path(tempfile.mkdtemp(prefix=f".{work_dir.name}.staging-", dir=work_dir.parent))
    created_work_dir = False
    try:
        # --reflink=auto uses independent copy-on-write inodes where available,
        # otherwise copies bytes. It never creates hard links to the source.
        subprocess.run(["cp", "-a", "--reflink=auto", str(checkpoint), str(staging / "checkpoint")], check=True)
        for name in ("config.json", "config.yaml"):
            if (run_dir / name).is_file():
                shutil.copy2(run_dir / name, staging / name)
        status = json.loads((checkpoint / "train_status.json").read_text())
        (staging / "resume_origin.json").write_text(
            json.dumps(
                {
                    "source_checkpoint": str(checkpoint),
                    "source_global_time": int(status.get("global_time", status.get("time", 0))),
                    "source_optimizer_steps": int(status.get("optimizer_steps", 0)),
                },
                indent=2,
            )
            + "\n"
        )
        for name in REQUIRED_CHECKPOINT_FILES:
            original = checkpoint / name
            copied = staging / "checkpoint" / name
            if copied.stat().st_size != original.stat().st_size:
                raise RuntimeError(f"Copy size mismatch: {name}")
            if copied.stat().st_ino == original.stat().st_ino and copied.stat().st_dev == original.stat().st_dev:
                raise RuntimeError(f"Copy shares an inode with source: {name}")
        if (staging / "checkpoint/train_status.json").read_bytes() != (checkpoint / "train_status.json").read_bytes():
            raise RuntimeError("Copied train_status.json differs from source")
        for name, state in source_state.items():
            original = checkpoint / name
            copied = staging / "checkpoint" / name
            if (original.stat().st_size, original.stat().st_mtime_ns) != state:
                raise RuntimeError(f"Source checkpoint changed during copy: {name}")
            if copied.stat().st_size != state[0]:
                raise RuntimeError(f"Copy size mismatch: {name}")
            if (copied.stat().st_dev, copied.stat().st_ino) == (original.stat().st_dev, original.stat().st_ino):
                raise RuntimeError(f"Copy shares an inode with source: {name}")
        # mkdir is the atomic no-clobber gate. The following rename only
        # replaces the empty directory created by this process.
        work_dir.mkdir(exist_ok=False)
        created_work_dir = True
        os.rename(staging, work_dir)
    except BaseException:
        shutil.rmtree(staging)
        if created_work_dir and work_dir.is_dir() and not any(work_dir.iterdir()):
            work_dir.rmdir()
        raise
    return work_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="Completed run directory or its checkpoint directory")
    parser.add_argument("--work-dir", required=True, type=Path, help="New, nonexistent continuation directory")
    args = parser.parse_args()
    staged = stage_continuation(args.source, args.work_dir)
    print(f"Staged independent continuation checkpoint: {staged / 'checkpoint'}")
    print("Training has not started. Run ./run_train.sh with --work-dir pointing to this directory.")


if __name__ == "__main__":
    main()
