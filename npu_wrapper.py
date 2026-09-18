from __future__ import annotations

import io
import os
import re
import runpy
import subprocess
import sys
import tokenize
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


_DEVICE_LITERAL = re.compile(r"([rRuU]*)(['\"])(cuda(?::[0-9]+)?)\2\Z")
_FSTRING_START = getattr(tokenize, "FSTRING_START", -1)
_FSTRING_MIDDLE = getattr(tokenize, "FSTRING_MIDDLE", -1)


def _rewrite_cuda_device_literals(data: bytes) -> bytes:
    """Rewrite Python string tokens that denote a CUDA device."""

    encoding, _ = tokenize.detect_encoding(io.BytesIO(data).readline)
    source = data.decode(encoding)
    lines = source.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))

    edits = []
    previous_type = None
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        replacement = None
        if token.type == tokenize.STRING:
            match = _DEVICE_LITERAL.fullmatch(token.string)
            if match:
                replacement = f"{match[1]}{match[2]}npu{match[3][4:]}{match[2]}"
            elif _FSTRING_MIDDLE == -1:
                # Before Python 3.12, tokenize emits the whole f-string as STRING.
                fstring = re.fullmatch(r"([fFrR]+)(['\"])(cuda:)(\{.*)\2", token.string)
                if fstring and "f" in fstring[1].lower():
                    replacement = f"{fstring[1]}{fstring[2]}npu:{fstring[4]}{fstring[2]}"
        elif token.type == _FSTRING_MIDDLE and previous_type == _FSTRING_START:
            if re.fullmatch(r"cuda(?::[0-9]*)?", token.string):
                replacement = "npu" + token.string[4:]
        if replacement is not None:
            start = offsets[token.start[0] - 1] + token.start[1]
            end = offsets[token.end[0] - 1] + token.end[1]
            edits.append((start, end, replacement))
        previous_type = token.type

    for start, end, replacement in reversed(edits):
        source = source[:start] + replacement + source[end:]
    return source.encode(encoding)


@contextmanager
def temporary_cuda_device_strings(
    repo: Path, *, require_clean: bool = True
) -> Generator[None, None, None]:
    """Temporarily rewrite CUDA device string literals in repository Python files."""
    if not (repo / ".git").exists():
        raise RuntimeError(f"recsys-examples is not a Git worktree: {repo}")
    if require_clean and _git(repo, "status", "--porcelain"):
        raise RuntimeError(f"recsys-examples has changes; rewrite mode requires a clean worktree: {repo}")

    changes = []
    try:
        for path in repo.rglob("*.py"):
            if ".git" in path.parts or path.is_symlink():
                continue
            original = path.read_bytes()
            try:
                rewritten = _rewrite_cuda_device_literals(original)
            except (SyntaxError, tokenize.TokenError, UnicodeError) as exc:
                raise RuntimeError(f"could not scan Python source {path}: {exc}") from exc
            if rewritten != original:
                stat = path.stat()
                changes.append((path, original, rewritten, stat.st_atime_ns, stat.st_mtime_ns))
                path.write_bytes(rewritten)
        yield
    finally:
        cleanup_errors = []
        for path, original, rewritten, atime_ns, mtime_ns in reversed(changes):
            try:
                if path.read_bytes() != rewritten:
                    raise RuntimeError("file changed during execution; left it untouched")
                path.write_bytes(original)
                os.utime(path, ns=(atime_ns, mtime_ns))
            except (OSError, RuntimeError) as exc:
                cleanup_errors.append(f"{path}: {exc}")
        if cleanup_errors:
            raise RuntimeError("Could not restore rewritten files:\n" + "\n".join(cleanup_errors))


@contextmanager
def rewrite_cuda(repo: Path, *, require_clean: bool = True) -> Generator[None, None, None]:
    """Install NPU aliases and temporarily rewrite CUDA device strings."""
    with temporary_cuda_device_strings(repo, require_clean=require_clean):
        install()
        yield


def main(argv: list[str] | None = None) -> None:
    """Run a script or module with optional patches and CUDA-to-NPU rewrite."""
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        print(
            "Usage: python npu_wrapper.py [--patch FILE]... [--rewrite-cuda] "
            "[-m module | script.py] [args...]",
            file=sys.stderr,
        )
        if not args:
            raise SystemExit(2)
        return

    patch_files = []
    rewrite_cuda_enabled = False
    while args and args[0] in {"--patch", "--rewrite-cuda"}:
        if args[0] == "--patch":
            if len(args) < 2:
                raise SystemExit("npu_wrapper: --patch requires a Git diff file")
            patch_files.append(args[1])
            args = args[2:]
        else:
            rewrite_cuda_enabled = True
            args = args[1:]
    if not args:
        raise SystemExit("npu_wrapper: requires a script or -m module")

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
        rewrite_context = (
            rewrite_cuda(repo, require_clean=not patch_files)
            if rewrite_cuda_enabled
            else nullcontext()
        )
        with rewrite_context:
            sys.argv = [target, *script_args]
            if module_mode:
                runpy.run_module(target, run_name="__main__", alter_sys=True)
            else:
                sys.path.insert(0, str(Path(target).resolve().parent))
                runpy.run_path(target, run_name="__main__")


if __name__ == "__main__":
    main()
