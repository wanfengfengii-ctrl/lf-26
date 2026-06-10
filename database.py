import sqlite3
import os
from contextlib import contextmanager
from typing import List, Dict, Optional, Tuple

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'salt_cavern.db')


@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    with get_db() as conn:
        conn.executescript('''
            CREATE TABLE IF NOT EXISTS caves (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS survey_batches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cave_id INTEGER NOT NULL,
                batch_name TEXT NOT NULL,
                survey_date TEXT,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (cave_id) REFERENCES caves(id) ON DELETE CASCADE,
                UNIQUE(cave_id, batch_name)
            );

            CREATE TABLE IF NOT EXISTS measurements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                batch_id INTEGER NOT NULL,
                angle REAL NOT NULL,
                distance REAL NOT NULL,
                depth REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (batch_id) REFERENCES survey_batches(id) ON DELETE CASCADE,
                UNIQUE(batch_id, angle)
            );

            CREATE TABLE IF NOT EXISTS volume_estimates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                batch_id INTEGER NOT NULL UNIQUE,
                volume REAL NOT NULL,
                max_depth REAL NOT NULL,
                max_distance REAL NOT NULL,
                calculation_method TEXT NOT NULL,
                calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (batch_id) REFERENCES survey_batches(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS anomaly_regions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                batch_id INTEGER NOT NULL,
                start_angle REAL NOT NULL,
                end_angle REAL NOT NULL,
                anomaly_type TEXT NOT NULL,
                description TEXT,
                FOREIGN KEY (batch_id) REFERENCES survey_batches(id) ON DELETE CASCADE
            );
        ''')


def get_all_caves() -> List[Dict]:
    with get_db() as conn:
        rows = conn.execute('SELECT * FROM caves ORDER BY name').fetchall()
        return [dict(row) for row in rows]


def create_cave(name: str, description: str = '') -> int:
    with get_db() as conn:
        cursor = conn.execute(
            'INSERT INTO caves (name, description) VALUES (?, ?)',
            (name, description)
        )
        return cursor.lastrowid


def get_cave(cave_id: int) -> Optional[Dict]:
    with get_db() as conn:
        row = conn.execute('SELECT * FROM caves WHERE id = ?', (cave_id,)).fetchone()
        return dict(row) if row else None


def delete_cave(cave_id: int):
    with get_db() as conn:
        conn.execute('DELETE FROM caves WHERE id = ?', (cave_id,))


def get_batches_by_cave(cave_id: int) -> List[Dict]:
    with get_db() as conn:
        rows = conn.execute(
            'SELECT * FROM survey_batches WHERE cave_id = ? ORDER BY survey_date, batch_name',
            (cave_id,)
        ).fetchall()
        return [dict(row) for row in rows]


def create_batch(cave_id: int, batch_name: str, survey_date: str = None, notes: str = '') -> int:
    with get_db() as conn:
        cursor = conn.execute(
            'INSERT INTO survey_batches (cave_id, batch_name, survey_date, notes) VALUES (?, ?, ?, ?)',
            (cave_id, batch_name, survey_date, notes)
        )
        return cursor.lastrowid


def get_batch(batch_id: int) -> Optional[Dict]:
    with get_db() as conn:
        row = conn.execute('SELECT * FROM survey_batches WHERE id = ?', (batch_id,)).fetchone()
        return dict(row) if row else None


def delete_batch(batch_id: int):
    with get_db() as conn:
        conn.execute('DELETE FROM survey_batches WHERE id = ?', (batch_id,))


def get_measurements_by_batch(batch_id: int) -> List[Dict]:
    with get_db() as conn:
        rows = conn.execute(
            'SELECT * FROM measurements WHERE batch_id = ? ORDER BY angle',
            (batch_id,)
        ).fetchall()
        return [dict(row) for row in rows]


def add_measurement(batch_id: int, angle: float, distance: float, depth: float) -> int:
    with get_db() as conn:
        cursor = conn.execute(
            'INSERT INTO measurements (batch_id, angle, distance, depth) VALUES (?, ?, ?, ?)',
            (batch_id, angle, distance, depth)
        )
        return cursor.lastrowid


def update_measurement(measurement_id: int, angle: float, distance: float, depth: float):
    with get_db() as conn:
        conn.execute(
            'UPDATE measurements SET angle = ?, distance = ?, depth = ? WHERE id = ?',
            (angle, distance, depth, measurement_id)
        )


def delete_measurement(measurement_id: int):
    with get_db() as conn:
        conn.execute('DELETE FROM measurements WHERE id = ?', (measurement_id,))


def get_measurement(measurement_id: int) -> Optional[Dict]:
    with get_db() as conn:
        row = conn.execute('SELECT * FROM measurements WHERE id = ?', (measurement_id,)).fetchone()
        return dict(row) if row else None


def check_angle_duplicate(batch_id: int, angle: float, exclude_id: int = None) -> bool:
    with get_db() as conn:
        if exclude_id:
            row = conn.execute(
                'SELECT id FROM measurements WHERE batch_id = ? AND angle = ? AND id != ?',
                (batch_id, angle, exclude_id)
            ).fetchone()
        else:
            row = conn.execute(
                'SELECT id FROM measurements WHERE batch_id = ? AND angle = ?',
                (batch_id, angle)
            ).fetchone()
        return row is not None


def save_volume_estimate(batch_id: int, volume: float, max_depth: float, max_distance: float, method: str):
    with get_db() as conn:
        conn.execute(
            '''INSERT OR REPLACE INTO volume_estimates 
               (batch_id, volume, max_depth, max_distance, calculation_method, calculated_at)
               VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)''',
            (batch_id, volume, max_depth, max_distance, method)
        )


def get_volume_estimate(batch_id: int) -> Optional[Dict]:
    with get_db() as conn:
        row = conn.execute(
            'SELECT * FROM volume_estimates WHERE batch_id = ?',
            (batch_id,)
        ).fetchone()
        return dict(row) if row else None


def save_anomaly_regions(batch_id: int, regions: List[Dict]):
    with get_db() as conn:
        conn.execute('DELETE FROM anomaly_regions WHERE batch_id = ?', (batch_id,))
        for region in regions:
            conn.execute(
                '''INSERT INTO anomaly_regions 
                   (batch_id, start_angle, end_angle, anomaly_type, description)
                   VALUES (?, ?, ?, ?, ?)''',
                (batch_id, region['start_angle'], region['end_angle'],
                 region['anomaly_type'], region.get('description', ''))
            )


def get_anomaly_regions(batch_id: int) -> List[Dict]:
    with get_db() as conn:
        rows = conn.execute(
            'SELECT * FROM anomaly_regions WHERE batch_id = ? ORDER BY start_angle',
            (batch_id,)
        ).fetchall()
        return [dict(row) for row in rows]


def batch_has_measurements(batch_id: int) -> bool:
    with get_db() as conn:
        row = conn.execute(
            'SELECT COUNT(*) as cnt FROM measurements WHERE batch_id = ?',
            (batch_id,)
        ).fetchone()
        return row['cnt'] > 0


def get_cave_by_name(name: str) -> Optional[Dict]:
    with get_db() as conn:
        row = conn.execute('SELECT * FROM caves WHERE name = ?', (name,)).fetchone()
        return dict(row) if row else None


def get_batch_by_name(cave_id: int, batch_name: str) -> Optional[Dict]:
    with get_db() as conn:
        row = conn.execute(
            'SELECT * FROM survey_batches WHERE cave_id = ? AND batch_name = ?',
            (cave_id, batch_name)
        ).fetchone()
        return dict(row) if row else None
