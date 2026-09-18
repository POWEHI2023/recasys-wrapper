from __future__ import annotations

import runpy
import subprocess
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import ModuleType
from typing import Generator, Sequence


def install() -> ModuleType:
    """Alias ``torch.cuda`` and tensor/module ``.cuda()`` to their NPU APIs."""

    try:
        import torch
        import torch_npu  # noqa: F401 - registers the NPU backend with PyTorch
    except ImportError as exc:
        raise RuntimeError(
            "npu_wrapper requires compatible torch and torch_npu installations"
        ) from exc

    npu = getattr(torch, "npu", None)
    if not isinstance(npu, ModuleType):
        raise RuntimeError("torch_npu did not register the torch.npu module")

    tensor_npu = getattr(torch.Tensor, "npu", None)
    module_npu = getattr(torch.nn.Module, "npu", None)
    if not callable(tensor_npu) or not callable(module_npu):
        raise RuntimeError("torch_npu did not register Tensor.npu and nn.Module.npu")

    torch.Tensor.cuda = tensor_npu
    torch.nn.Module.cuda = module_npu
    torch.cuda = npu
    sys.modules["torch.cuda"] = npu
    return npu


def _git(repo: Path, *args: str, patch: bytes | None = None) -> bytes:
    result = subprocess.run(
        ["git", *args], cwd=repo, input=patch, capture_output=True, check=False
    )
    if result.returncode:
        detail = result.stderr.decode(errors="replace").strip()
        raise RuntimeError(f"git {' '.join(args)} failed in {repo}: {detail}")
    return result.stdout


@contextmanager
def temporary_patches(
    patch_files: Sequence[str | Path], repo: Path
) -> Generator[None, None, None]:
    """Apply Git diffs in order and reverse applied diffs on context exit."""

    patches = []
    for patch_file in patch_files:
        patch_path = Path(patch_file).resolve()
        if not patch_path.is_file():
            raise RuntimeError(f"patch file not found: {patch_path}")
        # Keep immutable copies: a later patch or the target may edit a file.
        patches.append((patch_path, patch_path.read_bytes()))

    if not (repo / ".git").exists():
        raise RuntimeError(f"recsys-examples is not a Git worktree: {repo}")
    if _git(repo, "status", "--porcelain"):
        raise RuntimeError(f"recsys-examples has changes; patch mode requires a clean worktree: {repo}")

    applied = []
    try:
        for patch_path, patch in patches:
            _git(repo, "apply", "--check", "-", patch=patch)
            _git(repo, "apply", "-", patch=patch)
            applied.append((patch_path, patch))
        yield
    finally:
        cleanup_errors = []
        for patch_path, patch in reversed(applied):
            try:
                _git(repo, "apply", "--reverse", "--check", "-", patch=patch)
                _git(repo, "apply", "--reverse", "-", patch=patch)
            except RuntimeError as exc:
                cleanup_errors.append(f"{patch_path}: {exc}")
        if cleanup_errors:
            raise RuntimeError("Could not reverse applied patches:\n" + "\n".join(cleanup_errors))


def main(argv: list[str] | None = None) -> None:
    """Run a script or ``-m module`` with optional temporary source patch."""
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        print(
            "Usage: python npu_wrapper.py [--patch FILE]... [-m module | script.py] [args...]",
            file=sys.stderr,
        )
        if not args:
            raise SystemExit(2)
        return

    patch_files = []
    while args and args[0] == "--patch":
        if len(args) < 2:
            raise SystemExit("npu_wrapper: --patch requires a Git diff file")
        patch_files.append(args[1])
        args = args[2:]
    if not args:
        raise SystemExit("npu_wrapper: --patch requires a script or -m module")

    module_mode = args[0] == "-m"
    if module_mode:
        if len(args) < 2:
            raise SystemExit("npu_wrapper: -m requires a module name")
        target, script_args = args[1], args[2:]
    else:
        target, script_args = args[0], args[1:]
        if not Path(target).is_file():
            raise SystemExit(f"npu_wrapper: script not found: {target}")

    repo = Path(__file__).resolve().parent / "recsys-examples"
    patch_context = temporary_patches(patch_files, repo) if patch_files else nullcontext()
    with patch_context:
        install()
        sys.argv = [target, *script_args]
        if module_mode:
            runpy.run_module(target, run_name="__main__", alter_sys=True)
        else:
            sys.path.insert(0, str(Path(target).resolve().parent))
            runpy.run_path(target, run_name="__main__")


if __name__ == "__main__":
    main()
