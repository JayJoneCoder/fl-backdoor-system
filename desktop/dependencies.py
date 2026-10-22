"""Export the installed dependency closure, excluding only torch/torchvision.

Run in the verified Windows x64 Python 3.11 build environment. This is a
manifest/checker, not an installer: it never downloads or changes packages.
"""

import argparse
from importlib.metadata import distribution, PackageNotFoundError
import json
from pathlib import Path

import tomli
from pip._vendor.packaging.requirements import Requirement
from pip._vendor.packaging.utils import canonicalize_name

from fl_backdoor.utils.app_paths import resource_root


EXTERNAL = {"torch", "torchvision"}


def bundled_requirements() -> list[str]:
    with (resource_root() / "pyproject.toml").open("rb") as stream:
        document = tomli.load(stream)
    # Include pip: first-run wheel install and portable Flower console launchers
    # both depend on it. Traverse ML metadata too, but exclude their wheels.
    pending = [(Requirement(value), "") for value in [*document["project"]["dependencies"], "pip"]]
    visited = set()
    versions = {}
    while pending:
        requirement, parent_extra = pending.pop()
        if requirement.marker and not requirement.marker.evaluate({"extra": parent_extra}):
            continue
        name = canonicalize_name(requirement.name)
        try:
            installed = distribution(name)
        except PackageNotFoundError as exc:
            raise RuntimeError(f"发行依赖缺失：{name}。请先修复构建环境，不要让评委补装。") from exc
        if requirement.specifier and not requirement.specifier.contains(installed.version):
            raise RuntimeError(f"发行依赖不满足要求：{requirement}，实际为 {installed.version}")
        if name not in EXTERNAL:
            versions[name] = installed.version
        for extra in {"", *requirement.extras}:
            key = (name, extra)
            if key in visited:
                continue
            visited.add(key)
            pending.extend((Requirement(value), extra) for value in installed.requires or [])
    return [f"{name}=={version}" for name, version in sorted(versions.items())]


def main() -> None:
    parser = argparse.ArgumentParser(description="导出已验证环境中的随包依赖，不下载任何包")
    parser.add_argument("--output", type=Path, help="写出锁定版本清单；不指定则只打印")
    args = parser.parse_args()
    entries = bundled_requirements()
    header = "# Windows x64 / CPython 3.11; export from verified environment.\n# torch/torchvision are the only first-run downloads.\n"
    content = header + "\n".join(entries) + "\n"
    if args.output:
        args.output.resolve().write_text(content, encoding="utf-8")
        print(json.dumps({"bundled_packages": len(entries), "output": str(args.output.resolve()),
                          "external_packages": sorted(EXTERNAL)}, ensure_ascii=False))
    else:
        print(content, end="")


if __name__ == "__main__":
    main()