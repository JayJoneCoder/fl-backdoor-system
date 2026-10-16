"""SQLite read model for existing experiment artifacts (no training dependencies)."""

import csv
import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from backend.results_scanner import get_experiment_detail
from fl_backdoor.utils.paths import child_path, direct_files, experiment_path

SCHEMA_VERSION = 1
APPLICATION_ID = 0x464C4253
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCHEMA = (
    "CREATE TABLE workspaces (id TEXT PRIMARY KEY, name TEXT NOT NULL)",
    "INSERT INTO workspaces VALUES ('local', '本地工作空间')",
    """CREATE TABLE plans (id TEXT PRIMARY KEY, workspace_id TEXT NOT NULL REFERENCES workspaces(id),
        root TEXT NOT NULL, name TEXT NOT NULL, UNIQUE(root, name))""",
    """CREATE TABLE config_versions (id TEXT PRIMARY KEY, plan_id TEXT NOT NULL REFERENCES plans(id),
        sha256 TEXT NOT NULL, toml TEXT NOT NULL, created_at TEXT NOT NULL, UNIQUE(plan_id, sha256))""",
    """CREATE TABLE runs (id TEXT PRIMARY KEY, plan_id TEXT NOT NULL REFERENCES plans(id),
        config_id TEXT REFERENCES config_versions(id), root TEXT NOT NULL, name TEXT NOT NULL,
        source_run_id TEXT, present INTEGER NOT NULL DEFAULT 1, status TEXT NOT NULL, dataset TEXT,
        created_at TEXT, imported_at TEXT NOT NULL, fingerprint TEXT NOT NULL, detail_json TEXT NOT NULL)""",
    "CREATE UNIQUE INDEX active_location ON runs(root, name) WHERE present = 1",
    "CREATE INDEX run_filters ON runs(root, present, status, dataset, created_at)",
    """CREATE TABLE round_metrics (run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
        round INTEGER NOT NULL, accuracy REAL, asr REAL, loss REAL, PRIMARY KEY(run_id, round))""",
    """CREATE TABLE metric_records (run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
        kind TEXT NOT NULL, row_number INTEGER NOT NULL, payload_json TEXT NOT NULL,
        PRIMARY KEY(run_id, kind, row_number))""",
    """CREATE TABLE artifacts (run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
        name TEXT NOT NULL, kind TEXT NOT NULL, size INTEGER NOT NULL, mtime_ns INTEGER NOT NULL,
        sha256 TEXT NOT NULL, PRIMARY KEY(run_id, name))""",
    """CREATE TABLE batches (root TEXT NOT NULL, id TEXT NOT NULL, fingerprint TEXT NOT NULL,
        payload_json TEXT NOT NULL, imported_at TEXT NOT NULL, PRIMARY KEY(root, id))""",
)


def default_database_path() -> Path:
    """Keep mutable platform state outside the training project/config/results."""
    value = os.environ.get("FL_BACKDOOR_DATABASE")
    return Path(value).expanduser().resolve() if value else PROJECT_ROOT / "platform_state" / "catalog.sqlite3"


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _key(*parts) -> str:
    return uuid.uuid5(uuid.NAMESPACE_URL, _json(parts)).hex


def _root(path: Path) -> str:
    return os.path.normcase(str(Path(path).resolve()))


def _manifest(directory: Path) -> list[tuple]:
    return [(p.name, p.stat().st_size, p.stat().st_mtime_ns)
            for p in sorted(direct_files(directory), key=lambda p: p.name)]


def _fingerprint(manifest) -> str:
    return hashlib.sha256(_json(manifest).encode()).hexdigest()


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# `present` used to hide records when files disappeared. Prefer the current
# generation at a location, but also recover older catalogs with only absent
# rows. File availability is independent of whether a run is queryable.
LATEST_RUN = """id=(SELECT candidate.id FROM runs AS candidate
    WHERE candidate.root=runs.root AND candidate.name=runs.name
    ORDER BY candidate.present DESC, candidate.imported_at DESC, candidate.id DESC LIMIT 1)"""


class Catalog:
    """Connections are local to a call/thread. Every imported run is atomic.

    Files remain the ingestion source in this transitional release. Only this
    importer writes the read model; clients cannot edit a second copy of metrics.
    """

    def __init__(self, path: Path):
        self.path = Path(path).resolve()
        self._lock = threading.RLock()
        self._initialized = False

    @contextmanager
    def connection(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(self.path, timeout=15, isolation_level=None)
        con.row_factory = sqlite3.Row
        try:
            con.execute("PRAGMA foreign_keys = ON")
            con.execute("PRAGMA busy_timeout = 15000")
            yield con
        finally:
            con.close()

    def initialize(self) -> None:
        with self._lock:
            if self._initialized:
                return
            with self.connection() as con:
                con.execute("BEGIN IMMEDIATE")
                try:
                    version = con.execute("PRAGMA user_version").fetchone()[0]
                    application = con.execute("PRAGMA application_id").fetchone()[0]
                    if version == 0 and application == 0:
                        if con.execute("SELECT 1 FROM sqlite_master WHERE type='table'").fetchone():
                            raise ValueError("该文件不是平台数据库，拒绝修改。")
                        for statement in SCHEMA:
                            con.execute(statement)
                        con.execute(f"PRAGMA application_id = {APPLICATION_ID}")
                        con.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
                    elif application != APPLICATION_ID or version != SCHEMA_VERSION:
                        raise ValueError("数据库版本不受当前程序支持，请使用匹配版本或备份恢复。")
                    con.commit()
                except BaseException:
                    con.rollback()
                    raise
                con.execute("PRAGMA journal_mode = WAL")
            self._initialized = True

    def sync_run(self, directory: Path, *, force: bool = False) -> str:
        """Return imported/unchanged/skipped. Never modify a source artifact."""
        self.initialize()
        directory = experiment_path(directory.parent, directory.name)
        if not directory.is_dir():
            return "skipped"
        root = _root(directory.parent)
        with self._lock, self.connection() as con:
            manifest = _manifest(directory)
            fingerprint = _fingerprint(manifest)
            previous = con.execute("SELECT * FROM runs WHERE root=? AND name=? AND " + LATEST_RUN,
                                   (root, directory.name)).fetchone()
            if previous and previous["fingerprint"] == fingerprint and not force:
                return "unchanged"
            detail = get_experiment_detail(directory)
            if detail is None:
                return "skipped"
            # Keep the on-disk UUID separate from our stable primary key. Old
            # APIs use the former as the deletion/monitor ownership token.
            source_id = detail.get("run_id")
            if source_id is None and previous and not any(name == "run.json" for name, *_ in manifest):
                # Losing the metadata file does not create a new legacy run.
                source_id = previous["source_run_id"]
            if not isinstance(source_id, str):
                source_id = None
                detail["run_id"] = None
            run_id = _key("run", root, directory.name, source_id)
            plan_id = _key("plan", root, directory.name)
            detail["catalog_id"] = run_id
            saved = json.loads(previous["detail_json"]) if previous and previous["id"] == run_id else None
            if saved:
                # Import is additive for one immutable run. Missing/truncated
                # source files must not erase data that was already persisted.
                prior_metrics = saved["metrics"]
                metrics = {**prior_metrics, **{key: value for key, value in detail["metrics"].items()
                                               if value is not None and value != "" and key != "curve_data"}}
                curves = {}
                for source in (prior_metrics.get("curve_data", {}), detail["metrics"].get("curve_data", {})):
                    for index, round_number in enumerate(source.get("round", [])):
                        row = curves.setdefault(round_number, {})
                        for key in ("accuracy", "asr", "loss"):
                            values = source.get(key, [])
                            if index < len(values) and values[index] is not None:
                                row[key] = values[index]
                rounds = sorted(curves)
                metrics["curve_data"] = {"round": rounds, **{key: [curves[n].get(key) for n in rounds]
                                                             for key in ("accuracy", "asr", "loss")}}
                if rounds:
                    metrics["last_round"] = rounds[-1]
                    for key in ("accuracy", "asr", "loss"):
                        metrics[key] = curves[rounds[-1]].get(key)
                detail["metrics"] = metrics
                detail["has_metrics"] = saved["has_metrics"] or detail["has_metrics"]
                detail["can_plot"] = saved["can_plot"] or detail["can_plot"]
                detail["created"] = saved["created"]
                for key in ("images", "csv_files"):
                    detail[key] = sorted(set(saved.get(key, [])) | set(detail.get(key, [])))
                detail["model_file"] = detail["model_file"] or saved.get("model_file")
                if not any(name == "run.json" for name, *_ in manifest):
                    for key in ("run_id", "status", "error", "exit_code"):
                        detail[key] = saved.get(key)
                if detail["has_metrics"]:
                    detail["warnings"] = [warning for warning in detail["warnings"]
                                           if warning != "尚无指标 CSV，可查看运行状态和日志。"]
            snapshot = child_path(directory, "config_snapshot.toml")
            config = None
            if snapshot.is_file():
                try:
                    config = snapshot.read_text(encoding="utf-8")
                except (OSError, UnicodeError):
                    detail["warnings"].append("配置快照无法解码，已保留原文件索引。")
            old = {r["name"]: r for r in con.execute("SELECT * FROM artifacts WHERE run_id=?", (run_id,))}
            artifacts = []
            records = []
            for name, size, mtime in manifest:
                path = child_path(directory, name)
                cached = old.get(name)
                digest = (cached["sha256"] if cached and cached["size"] == size
                          and cached["mtime_ns"] == mtime and not force else _hash(path))
                kind = {".csv": "metrics", ".png": "image", ".pt": "model", ".log": "log",
                        ".toml": "config", ".json": "metadata"}.get(path.suffix.lower(), "file")
                artifacts.append((run_id, name, kind, size, mtime, digest))
                if name.endswith(("_clients.csv", "_metrics.csv")):
                    try:
                        with path.open(encoding="utf-8", newline="") as stream:
                            rows = list(csv.DictReader(stream, strict=True))
                        kind = "clients" if name.endswith("_clients.csv") else "detection_aggregation"
                        records.extend((run_id, kind + ":" + name, i, _json(row)) for i, row in enumerate(rows))
                    except (csv.Error, UnicodeError, ValueError):
                        detail["warnings"].append(f"{name} 明细无法完整导入，原文件未改变。")
            # If training changed any source while it was being read, retry on
            # the next poll rather than committing a mixed-generation snapshot.
            if manifest != _manifest(directory):
                raise OSError("实验文件仍在写入，稍后重新同步。")
            imported = _now()
            config_id = (_key("config", plan_id, hashlib.sha256(config.encode()).hexdigest()) if config is not None
                         else previous["config_id"] if saved else None)
            con.execute("BEGIN IMMEDIATE")
            try:
                con.execute("INSERT OR IGNORE INTO plans VALUES (?, 'local', ?, ?)", (plan_id, root, directory.name))
                if config is not None:
                    con.execute("INSERT OR IGNORE INTO config_versions VALUES (?, ?, ?, ?, ?)",
                                (config_id, plan_id, hashlib.sha256(config.encode()).hexdigest(), config, imported))
                con.execute("UPDATE runs SET present=0 WHERE root=? AND name=? AND id<>?", (root, directory.name, run_id))
                con.execute("""INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET config_id=excluded.config_id, present=1,
                    status=excluded.status, dataset=excluded.dataset, created_at=excluded.created_at,
                    imported_at=excluded.imported_at, fingerprint=excluded.fingerprint, detail_json=excluded.detail_json""",
                            (run_id, plan_id, config_id, root, directory.name, source_id, detail["status"],
                             detail["metrics"].get("dataset", ""), detail["created"], imported, fingerprint, _json(detail)))
                curves = detail["metrics"].get("curve_data", {})
                con.executemany("INSERT OR REPLACE INTO round_metrics VALUES (?, ?, ?, ?, ?)",
                                [(run_id, *row) for row in zip(*(curves.get(k, []) for k in ("round", "accuracy", "asr", "loss")))])
                con.executemany("INSERT OR REPLACE INTO metric_records VALUES (?, ?, ?, ?)", records)
                con.executemany("INSERT OR REPLACE INTO artifacts VALUES (?, ?, ?, ?, ?, ?)", artifacts)
                con.commit()
            except BaseException:
                con.rollback()
                raise
            return "imported"

    def sync_root(self, root: Path, *, force: bool = False) -> dict:
        self.initialize()
        root = Path(root).resolve()
        report = {"imported": 0, "unchanged": 0, "skipped": 0, "missing": 0, "warnings": []}
        # Hold the whole reconciliation lock so another request cannot import
        # a just-created run between our directory listing and absence update.
        with self._lock:
            if not root.exists():
                report["warnings"].append("结果目录尚不存在，已保留数据库中的历史记录。")
                with self.connection() as con:
                    report["missing"] = con.execute("SELECT count(*) FROM runs WHERE root=? AND " + LATEST_RUN,
                                                    (_root(root),)).fetchone()[0]
                return report
            directories = []
            for path in sorted(root.iterdir()):
                if path.name.casefold() in {"summaries", "batches", "config_backups"}:
                    continue
                try:
                    directory = experiment_path(root, path.name)
                    if not directory.is_dir():
                        continue
                    directories.append(path.name)
                    report[self.sync_run(directory, force=force)] += 1
                except (OSError, ValueError, TypeError, KeyError, OverflowError) as exc:
                    report["warnings"].append(f"{path.name} 未同步：{exc}")
            with self.connection() as con:
                rows = con.execute("SELECT name FROM runs WHERE root=? AND " + LATEST_RUN, (_root(root),)).fetchall()
                report["missing"] = sum(row["name"] not in directories for row in rows)
            self._sync_batches(root, report)
        return report

    def _sync_batches(self, root: Path, report: dict) -> None:
        try:
            directory = child_path(root, "batches")
        except ValueError:
            report["warnings"].append("批次目录包含链接，已跳过。")
            return
        if not directory.is_dir():
            return
        for path in direct_files(directory, "*.json"):
            try:
                raw = path.read_bytes()
                data = json.loads(raw)
                if not isinstance(data, dict) or data.get("batch_id") != path.stem:
                    raise ValueError("批次编号无效")
                digest = hashlib.sha256(raw).hexdigest()
                with self.connection() as con:
                    con.execute("""INSERT INTO batches VALUES (?, ?, ?, ?, ?)
                        ON CONFLICT(root,id) DO UPDATE SET fingerprint=excluded.fingerprint,
                        payload_json=excluded.payload_json, imported_at=excluded.imported_at
                        WHERE batches.fingerprint<>excluded.fingerprint""",
                                (_root(root), path.stem, digest, _json(data), _now()))
            except (OSError, ValueError, TypeError) as exc:
                report["warnings"].append(f"批次 {path.name} 未同步：{exc}")

    def detail(self, root: Path, name: str) -> dict | None:
        self.initialize()
        with self.connection() as con:
            row = con.execute("SELECT * FROM runs WHERE root=? AND name=? AND " + LATEST_RUN, (_root(root), name)).fetchone()
            return self._read_detail(con, row) if row else None

    def _read_detail(self, con, row) -> dict:
        """Stored experiment data survives; only attachment actions depend on disk."""
        detail = json.loads(row["detail_json"])
        known = {item[0] for item in con.execute("SELECT name FROM artifacts WHERE run_id=?", (row["id"],))}
        available = set()
        try:
            directory = experiment_path(Path(row["root"]), row["name"])
            available = {path.name for path in direct_files(directory)} if directory.is_dir() else set()
            if "run.json" in available and row["source_run_id"]:
                metadata = json.loads(child_path(directory, "run.json").read_text(encoding="utf-8"))
                if not isinstance(metadata, dict) or metadata.get("run_id") != row["source_run_id"]:
                    available = set()
        except (OSError, ValueError, RuntimeError):
            available = set()
        # Never claim that a new, unimported attachment belongs to an old run.
        available &= known
        detail["available_files"] = sorted(available)
        detail["missing_files"] = sorted(known - available)
        detail["files_available"] = bool(available)
        detail["attachments_missing"] = bool(known - available) or not available
        if detail["attachments_missing"]:
            detail["warnings"].append("部分或全部结果文件缺失，数据库中的实验信息、配置和指标仍保留；缺失附件无法查看或下载。")
        detail["images"] = [name for name in detail.get("images", []) if name in available]
        detail["csv_files"] = [name for name in detail.get("csv_files", []) if name in available]
        if detail.get("model_file") not in available:
            detail["model_file"] = None
        detail["can_plot"] = bool(detail.get("can_plot") and any(
            not Path(name).stem.endswith(("_clients", "_metrics")) for name in detail["csv_files"]))
        config = con.execute("SELECT toml FROM config_versions WHERE id=?", (row["config_id"],)).fetchone()
        detail["config_toml"] = config[0] if config else None
        return detail

    def query(self, root: Path, *, search: str = "", status: str | None = None,
              dataset: str | None = None, limit: int | None = None, offset: int = 0) -> dict:
        if (limit is not None and (type(limit) is not int or not 1 <= limit <= 200)) or type(offset) is not int or offset < 0:
            raise ValueError("分页大小应为 1～200 的整数，偏移量不得为负数。")
        self.initialize()
        clauses = ["root=?", LATEST_RUN]
        values = [_root(root)]
        if search:
            clauses.append("instr(lower(name), lower(?)) > 0")
            values.append(search)
        for column, value in (("status", status), ("dataset", dataset)):
            if value is not None:
                clauses.append(f"{column}=?")
                values.append(value)
        where = " AND ".join(clauses)
        with self.connection() as con:
            con.execute("BEGIN")
            total = con.execute("SELECT count(*) FROM runs WHERE " + where, values).fetchone()[0]
            sql = "SELECT * FROM runs WHERE " + where + " ORDER BY created_at DESC, id"
            rows = con.execute(sql + " LIMIT ? OFFSET ?", [*values, limit if limit is not None else -1, offset]).fetchall()
            items = []
            for row in rows:
                detail = self._read_detail(con, row)
                items.append({**{k: detail.get(k) for k in ("name", "run_id", "catalog_id", "status", "error", "exit_code", "created", "has_metrics", "can_plot", "warnings",
                                                          "files_available", "available_files", "missing_files", "attachments_missing", "csv_files")},
                              **{k: detail["metrics"].get(k) for k in ("accuracy", "asr", "loss", "last_round", "precision", "recall", "fpr", "auc", "dataset")}})
        return {"items": items, "total": total, "limit": limit, "offset": offset}

    def list_batches(self, root: Path) -> list[dict]:
        self.initialize()
        with self.connection() as con:
            rows = con.execute("SELECT payload_json FROM batches WHERE root=? ORDER BY id DESC", (_root(root),)).fetchall()
        return [json.loads(row[0]) for row in rows]

    def record_sync(self, root: Path, report: dict) -> None:
        # Operational status only; source records and import failures remain independent.
        self.last_sync = {"root": str(root), "finished_at": _now(), **report}

    def mark_removed(self, root: Path, name: str) -> None:
        self.initialize()
        with self._lock, self.connection() as con:
            # User-requested deletion removes derived records/configs as well.
            con.execute("BEGIN IMMEDIATE")
            try:
                plan = con.execute("SELECT id FROM plans WHERE root=? AND name=?", (_root(root), name)).fetchone()
                if plan:
                    con.execute("DELETE FROM runs WHERE plan_id=?", (plan[0],))
                    con.execute("DELETE FROM config_versions WHERE plan_id=?", (plan[0],))
                    con.execute("DELETE FROM plans WHERE id=?", (plan[0],))
                con.commit()
            except BaseException:
                con.rollback()
                raise

    def stats(self) -> dict:
        self.initialize()
        with self.connection() as con:
            counts = {table: con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
                      for table in ("plans", "runs", "config_versions", "round_metrics", "metric_records", "artifacts", "batches")}
            integrity = con.execute("PRAGMA quick_check").fetchone()[0]
        return {"schema_version": SCHEMA_VERSION, "integrity": integrity, "counts": counts,
                "last_sync": getattr(self, "last_sync", None)}

    def backup(self, target: Path) -> None:
        """Online SQLite backup, never a raw copy of a live WAL database."""
        self.initialize()
        target = Path(target).resolve()
        if target == self.path or target.exists():
            raise FileExistsError("备份目标已存在，请使用新的文件名。")
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=".catalog-", suffix=".sqlite3", dir=target.parent)
        os.close(fd)
        try:
            with self._lock, self.connection() as source:
                dest = sqlite3.connect(temporary)
                try:
                    source.backup(dest)
                    if dest.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                        raise ValueError("数据库备份校验失败。")
                finally:
                    dest.close()
            # Exclusive creation prevents replacing another backup on races.
            created = False
            try:
                with target.open("xb") as output, open(temporary, "rb") as stream:
                    created = True
                    shutil.copyfileobj(stream, output)
                    output.flush()
                    os.fsync(output.fileno())
            except BaseException:
                if created:
                    target.unlink(missing_ok=True)
                raise
        finally:
            Path(temporary).unlink(missing_ok=True)

    @staticmethod
    def restore(source: Path, target: Path) -> None:
        """Restore to a NEW database only, never overwrite a running backend."""
        source = Path(source).resolve()
        if not source.is_file():
            raise ValueError("数据库备份文件不存在。")
        con = sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)
        try:
            if (con.execute("PRAGMA application_id").fetchone()[0] != APPLICATION_ID
                    or con.execute("PRAGMA user_version").fetchone()[0] != SCHEMA_VERSION
                    or con.execute("PRAGMA integrity_check").fetchone()[0] != "ok"):
                raise ValueError("不是有效的平台数据库备份。")
        finally:
            con.close()
        # Copy from a read-only connection too; validation alone must not turn
        # into an implicit migration or a write to the supplied backup.
        target = Path(target).resolve()
        if target.exists() or target == source:
            raise FileExistsError("恢复目标已存在，请使用新的文件名。")
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=".restore-", suffix=".sqlite3", dir=target.parent)
        os.close(fd)
        try:
            original = sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)
            restored = sqlite3.connect(temporary)
            try:
                original.backup(restored)
                if (restored.execute("PRAGMA integrity_check").fetchone()[0] != "ok"
                        or restored.execute("PRAGMA foreign_key_check").fetchone() is not None):
                    raise ValueError("恢复数据库校验失败。")
            finally:
                restored.close()
                original.close()
            created = False
            try:
                with target.open("xb") as output, open(temporary, "rb") as stream:
                    created = True
                    shutil.copyfileobj(stream, output)
                    output.flush()
                    os.fsync(output.fileno())
            except BaseException:
                if created:
                    target.unlink(missing_ok=True)
                raise
        finally:
            Path(temporary).unlink(missing_ok=True)


_catalogs: dict[Path, Catalog] = {}
_catalog_lock = threading.Lock()


def get_catalog() -> Catalog:
    path = default_database_path()
    with _catalog_lock:
        return _catalogs.setdefault(path, Catalog(path))