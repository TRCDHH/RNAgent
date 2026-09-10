"""MySQL 数据访问层（本地库，pymysql）。

- 数据库：rnagent
- 用户：rnagent / 123456
- 表：dataset(id, name, path, create_time)
      task(id, name, path, model, dataset_id, process(json), create_time)
"""

import json
from contextlib import contextmanager

import pymysql

DB_CONFIG = {
    "host": "127.0.0.1",
    "port": 3306,
    "user": "rnagent",
    "password": "123456",
    "database": "rnagent",
    "charset": "utf8mb4",
    "cursorclass": pymysql.cursors.DictCursor,
    "autocommit": False,
}


def get_conn():
    return pymysql.connect(**DB_CONFIG)


@contextmanager
def conn_ctx():
    conn = get_conn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_schema():
    """建表（库/用户需先在 MySQL 中创建，见 init.sql）。"""
    with conn_ctx() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS dataset (
                    id BIGINT PRIMARY KEY AUTO_INCREMENT,
                    name VARCHAR(255) NOT NULL,
                    path VARCHAR(512) NOT NULL,
                    create_time DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS task (
                    id BIGINT PRIMARY KEY AUTO_INCREMENT,
                    name VARCHAR(255) NOT NULL,
                    path VARCHAR(512),
                    model VARCHAR(32) NOT NULL DEFAULT 'sclinformer',
                    dataset_id BIGINT NULL,
                    process TEXT,
                    create_time DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """
            )
            # 兼容历史库：缺列时补上（MySQL 不支持 ADD COLUMN IF NOT EXISTS）
            added = []
            for col, ddl in (
                ("model", "ALTER TABLE task ADD COLUMN model VARCHAR(32) NOT NULL DEFAULT 'sclinformer'"),
                ("dataset_id", "ALTER TABLE task ADD COLUMN dataset_id BIGINT NULL"),
            ):
                cur.execute(
                    """
                    SELECT COUNT(*) AS n FROM information_schema.COLUMNS
                    WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'task' AND COLUMN_NAME = %s
                    """,
                    (col,),
                )
                if not (cur.fetchone() or {}).get("n"):
                    cur.execute(ddl)
                    added.append(col)
            if "dataset_id" in added:
                _backfill_dataset_id(cur)


def _backfill_dataset_id(cur):
    """给历史任务回填 dataset_id：从 process JSON 的事件里提取（一次性、幂等）。

    老任务的 process 事件里带有 dataset_id 字段（pipeline_start 等），
    建列后跑一次即可补齐；取不到的行保持 NULL（前端显示「数据集已删除/未知」）。
    """
    cur.execute("SELECT id, process FROM task WHERE dataset_id IS NULL")
    for row in cur.fetchall():
        ds_id = None
        try:
            data = json.loads(row["process"]) if row["process"] else {}
        except Exception:
            data = {}
        for ev in (data.get("events") or []):
            if ev.get("dataset_id") is not None:
                ds_id = ev["dataset_id"]
                break
        if ds_id is not None:
            cur.execute("UPDATE task SET dataset_id = %s WHERE id = %s", (ds_id, row["id"]))


# ---------- dataset ----------
def list_datasets():
    with conn_ctx() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id, name, path, create_time FROM dataset ORDER BY id DESC")
            return cur.fetchall()


def create_dataset(name: str, path: str) -> int:
    with conn_ctx() as conn:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO dataset (name, path) VALUES (%s, %s)", (name, path))
            return cur.lastrowid


def get_dataset(dataset_id: int):
    with conn_ctx() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id, name, path, create_time FROM dataset WHERE id = %s", (dataset_id,))
            return cur.fetchone()


def delete_dataset(dataset_id: int):
    with conn_ctx() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM dataset WHERE id = %s", (dataset_id,))


# ---------- task ----------
def list_tasks():
    with conn_ctx() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id, name, path, model, dataset_id, process, create_time "
                        "FROM task ORDER BY id DESC")
            rows = cur.fetchall()
    for r in rows:
        r.setdefault("model", "sclinformer")
        r.setdefault("dataset_id", None)
    return rows


def create_task(name: str, path: str, model: str = "sclinformer", dataset_id: int = None) -> int:
    with conn_ctx() as conn:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO task (name, path, model, dataset_id) VALUES (%s, %s, %s, %s)",
                        (name, path, model or "sclinformer", dataset_id))
            return cur.lastrowid


def delete_task(task_id: int):
    with conn_ctx() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM task WHERE id = %s", (task_id,))


def update_task_meta(task_id: int, name: str, path: str):
    with conn_ctx() as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE task SET name = %s, path = %s WHERE id = %s", (name, path, task_id))


def update_task_process(task_id: int, process_json: str):
    with conn_ctx() as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE task SET process = %s WHERE id = %s", (process_json, task_id))


def get_task(task_id: int):
    with conn_ctx() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id, name, path, model, dataset_id, process, create_time "
                        "FROM task WHERE id = %s", (task_id,))
            row = cur.fetchone()
            if row is not None:
                row.setdefault("model", "sclinformer")
                row.setdefault("dataset_id", None)
            return row
