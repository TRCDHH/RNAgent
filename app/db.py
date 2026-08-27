"""MySQL 数据访问层（本地库，pymysql）。

- 数据库：rnagent
- 用户：rnagent / 123456
- 表：dataset(id, name, path, create_time)
      task(id, name, path, process(json), create_time)
"""

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
                    process TEXT,
                    create_time DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """
            )


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
            cur.execute("SELECT id, name, path, process, create_time FROM task ORDER BY id DESC")
            rows = cur.fetchall()
    return rows


def create_task(name: str, path: str) -> int:
    with conn_ctx() as conn:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO task (name, path) VALUES (%s, %s)", (name, path))
            return cur.lastrowid


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
            cur.execute("SELECT id, name, path, process, create_time FROM task WHERE id = %s", (task_id,))
            return cur.fetchone()
