import sqlite3
import os
import hashlib
import json

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUTH_DB_PATH = os.path.join(ROOT_DIR, "data", "auth.db")
FACES_DB_PATH = os.path.join(ROOT_DIR, "data", "faces.db")
os.makedirs(os.path.join(ROOT_DIR, "data"), exist_ok=True)


class DualDatabaseManager:
    """
    Manager for two separate databases:
    1. auth.db -> Stores enrollment_number, password_hash, name, created_at
    2. faces.db -> Stores enrollment_number, image_path, encoding_json, created_at
    """

    def __init__(self, auth_db=AUTH_DB_PATH, faces_db=FACES_DB_PATH):
        self.auth_db = auth_db
        self.faces_db = faces_db
        self.init_databases()

    def _get_auth_conn(self):
        conn = sqlite3.connect(self.auth_db)
        conn.row_factory = sqlite3.Row
        return conn

    def _get_faces_conn(self):
        conn = sqlite3.connect(self.faces_db)
        conn.row_factory = sqlite3.Row
        return conn

    def init_databases(self):
        """Create tables in both databases if they do not exist."""
        # Database 1: auth.db (Enrollment Number & Password)
        conn1 = self._get_auth_conn()
        try:
            conn1.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    enrollment_number TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    name TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn1.commit()
        finally:
            conn1.close()

        # Database 2: faces.db (Enrollment Number & Face Biometrics/Image)
        conn2 = self._get_faces_conn()
        try:
            conn2.execute("""
                CREATE TABLE IF NOT EXISTS face_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    enrollment_number TEXT UNIQUE NOT NULL,
                    image_path TEXT,
                    encoding_json TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn2.commit()
        finally:
            conn2.close()

    @staticmethod
    def hash_password(password: str) -> str:
        """Hash password using SHA-256 with salt."""
        salt = "meet_app_salt_2026"
        return hashlib.sha256((password + salt).encode('utf-8')).hexdigest()

    # --- Database 1 Operations (auth.db) ---

    def create_user_credentials(self, enrollment_number: str, password: str, name: str) -> bool:
        """Store user credentials in Database 1 (auth.db)."""
        enrollment_number = enrollment_number.strip().upper()
        pwd_hash = self.hash_password(password)
        conn = self._get_auth_conn()
        try:
            try:
                conn.execute(
                    "INSERT INTO users (enrollment_number, password_hash, name) VALUES (?, ?, ?)",
                    (enrollment_number, pwd_hash, name)
                )
                conn.commit()
            except sqlite3.IntegrityError:
                conn.execute(
                    "UPDATE users SET password_hash = ?, name = ? WHERE enrollment_number = ?",
                    (pwd_hash, name, enrollment_number)
                )
                conn.commit()
            return True
        finally:
            conn.close()

    def verify_user_credentials(self, enrollment_number: str, password: str):
        """
        Check enrollment_number and password in Database 1.
        Returns user dict if valid, else None.
        """
        enrollment_number = enrollment_number.strip().upper()
        pwd_hash = self.hash_password(password)
        conn = self._get_auth_conn()
        try:
            row = conn.execute(
                "SELECT * FROM users WHERE enrollment_number = ?",
                (enrollment_number,)
            ).fetchone()
            if row and row['password_hash'] == pwd_hash:
                return dict(row)
            return None
        finally:
            conn.close()

    def get_user_by_enrollment(self, enrollment_number: str):
        enrollment_number = enrollment_number.strip().upper()
        conn = self._get_auth_conn()
        try:
            row = conn.execute(
                "SELECT * FROM users WHERE enrollment_number = ?",
                (enrollment_number,)
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    # --- Database 2 Operations (faces.db) ---

    def store_face_record(self, enrollment_number: str, encoding: list, image_path: str = None) -> bool:
        """Store face encoding and image path in Database 2 (faces.db)."""
        enrollment_number = enrollment_number.strip().upper()
        encoding_json = json.dumps(list(encoding))
        conn = self._get_faces_conn()
        try:
            try:
                conn.execute(
                    "INSERT INTO face_records (enrollment_number, image_path, encoding_json) VALUES (?, ?, ?)",
                    (enrollment_number, image_path, encoding_json)
                )
                conn.commit()
            except sqlite3.IntegrityError:
                conn.execute(
                    "UPDATE face_records SET encoding_json = ?, image_path = ? WHERE enrollment_number = ?",
                    (encoding_json, image_path, enrollment_number)
                )
                conn.commit()
            return True
        finally:
            conn.close()

    def get_face_record(self, enrollment_number: str):
        """Retrieve stored face encoding from Database 2 for given enrollment number."""
        enrollment_number = enrollment_number.strip().upper()
        conn = self._get_faces_conn()
        try:
            row = conn.execute(
                "SELECT * FROM face_records WHERE enrollment_number = ?",
                (enrollment_number,)
            ).fetchone()
            if row:
                res = dict(row)
                res['encoding'] = json.loads(res['encoding_json'])
                return res
            return None
        finally:
            conn.close()


db_manager = DualDatabaseManager()
