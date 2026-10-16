"""Manage the local experiment catalog without starting a training process."""

import argparse
import json
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.catalog import Catalog, default_database_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="实验数据库：导入旧结果、查询、检查、备份与恢复")
    parser.add_argument("--database", type=Path, default=None, help="数据库路径；默认 platform_state/catalog.sqlite3")
    commands = parser.add_subparsers(dest="command", required=True)
    sync = commands.add_parser("import", help="增量导入，不修改原始实验文件")
    sync.add_argument("--results-dir", type=Path, help="不填写时使用当前项目配置")
    sync.add_argument("--force", action="store_true", help="重新读取全部文件并计算校验值")
    commands.add_parser("status", help="查看版本、完整性和记录数量")
    query = commands.add_parser("list", help="分页查询实验")
    query.add_argument("--results-dir", type=Path)
    query.add_argument("--search", default="")
    query.add_argument("--status")
    query.add_argument("--dataset")
    query.add_argument("--limit", type=int, default=20)
    query.add_argument("--offset", type=int, default=0)
    backup = commands.add_parser("backup", help="在线备份数据库（不包含日志、模型等原文件）")
    backup.add_argument("target", type=Path)
    restore = commands.add_parser("restore", help="校验后恢复到一个新文件，绝不覆盖现有数据库")
    restore.add_argument("source", type=Path)
    restore.add_argument("target", type=Path)
    args = parser.parse_args(argv)
    try:
        catalog = Catalog(args.database or default_database_path())
        if args.command in {"import", "list"}:
            from backend.config_manager import get_results_dir
            root = args.results_dir.resolve() if args.results_dir else get_results_dir()
        if args.command == "import":
            result = catalog.sync_root(root, force=args.force)
        elif args.command == "list":
            result = catalog.query(root, search=args.search, status=args.status, dataset=args.dataset,
                                   limit=args.limit, offset=args.offset)
        elif args.command == "status":
            result = catalog.stats()
        elif args.command == "backup":
            catalog.backup(args.target)
            result = {"status": "success", "backup": str(args.target.resolve()),
                      "notice": "仅备份数据库；完整恢复还需保留原始结果目录。"}
        else:
            Catalog.restore(args.source, args.target)
            result = {"status": "success", "database": str(args.target.resolve()),
                      "notice": "停止后端后设置 FL_BACKDOOR_DATABASE 指向新文件，再启动并同步结果。"}
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (OSError, ValueError, sqlite3.Error) as exc:
        print(f"数据库操作失败：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())