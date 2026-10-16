"""Portable single-component names and containment checks for user-owned files."""

import re
import os
from pathlib import Path


def resolve_results_root(value: str, project_root: Path) -> Path:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError("结果目录不能为空或带有首尾空白。")
    if re.search(r"[\x00-\x1f\x7f]", value):
        raise ValueError("结果目录包含非法字符。")
    path = Path(value).expanduser()
    if os.name == "nt":
        if value.startswith(("\\\\", "//")) or ((path.drive or path.root) and not path.is_absolute()):
            raise ValueError("结果目录不支持网络共享、磁盘相对路径或缺少盘符的根路径。")
        for part in path.parts:
            if part in {path.anchor, ".", ".."}:
                continue
            if re.search(r'[<>:"|?*]', part) or part.endswith((".", " ")):
                raise ValueError("结果目录包含 Windows 不支持的字符或结尾。")
            stem = part.split(".", 1)[0].upper()
            if stem in {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"} or re.fullmatch(r"(?:COM|LPT)[1-9¹²³]", stem):
                raise ValueError("结果目录不能使用 Windows 保留设备名。")
    path = path if path.is_absolute() else Path(project_root) / path
    # A user can choose an absolute root, but links must never redirect it.
    for parent in (path, *path.parents):
        if parent.is_symlink() or (hasattr(parent, "is_junction") and parent.is_junction()):
            raise ValueError("结果目录不能通过符号链接或目录联接访问。")
        if parent.exists() and not parent.is_dir():
            raise ValueError("结果目录及其父路径必须是文件夹，不能是文件。")
    resolved = path.resolve()
    if path.absolute() != resolved:
        # Lexical '..' is allowed; an existing junction that resolves elsewhere isn't.
        lexical = Path(os.path.abspath(path))
        if lexical != resolved:
            raise ValueError("结果目录不能通过链接访问。")
    if resolved.exists() and not resolved.is_dir():
        raise ValueError("结果目录必须是文件夹，不能是文件。")
    return resolved


def validate_name(value: str, *, label: str = "Name", reserved=(), max_length: int = 120) -> str:
    if not isinstance(value, str) or not value or len(value) > max_length:
        raise ValueError(f"{label} must contain 1 to {max_length} characters.")
    if value != value.strip() or value in {".", ".."} or value.startswith((".", "-")):
        raise ValueError(f"{label} must not start with a dot/dash or contain surrounding whitespace.")
    if value.endswith((".", " ")) or re.search(r'[<>:"/\\|?*\x00-\x1f\x7f]', value):
        raise ValueError(f"{label} must be a filename, not a path, and contain no reserved characters.")
    stem = value.split(".", 1)[0].upper()
    if stem in {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"} or re.fullmatch(r"(?:COM|LPT)[1-9¹²³]", stem):
        raise ValueError(f"{label} is a reserved Windows device name.")
    if value.casefold() in {str(item).casefold() for item in reserved}:
        raise ValueError(f"{label} is reserved for application data.")
    return value


def child_path(root: Path, name: str, *, label: str = "Name", reserved=()) -> Path:
    validate_name(name, label=label, reserved=reserved, max_length=255)
    root = Path(root).resolve()
    path = root / name
    # Resolve existing symlinks/junctions as well as non-existent leaf paths.
    try:
        resolved = path.resolve()
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"{label} cannot be resolved safely.") from exc
    if path.is_symlink() or resolved.parent != root or resolved != path:
        raise ValueError(f"{label} must stay inside its data directory; links are not allowed.")
    return path


def experiment_path(root: Path, name: str) -> Path:
    validate_name(name, label="Experiment name")
    return child_path(root, name, label="Experiment name", reserved={"summaries", "batches", "config_backups"})


def summary_root(root: Path) -> Path:
    return child_path(root, "summaries")


def summary_path(root: Path, name: str) -> Path:
    validate_name(name, label="Summary name")
    return child_path(summary_root(root), name, label="Summary name")


def direct_files(directory: Path, pattern: str = "*") -> list[Path]:
    """List regular direct children, skipping links and invalid filenames."""
    files = []
    for candidate in directory.glob(pattern):
        try:
            path = child_path(directory, candidate.name)
            if path.is_file():
                files.append(path)
        except ValueError:
            continue
    return files