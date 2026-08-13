"""
Imports already-registered people from an older PROCTOR_APP install.

Only works where the source used the same SFace engine (128-dimension encodings).
Encodings from InsightFace/ArcFace are 512-dimension and mean something different,
so those cannot be copied - re-register those people from their photo instead,
which this script will do if you point it at an image directory.

    python migrate_users.py /path/to/PROCTOR_APP
    python migrate_users.py /path/to/PROCTOR_APP --photos /path/to/known_faces
"""
import argparse
import json
import os
import shutil
import sqlite3
import sys

from proctor.db_manager import db_manager
from proctor.face_engine import face_engine

EXPECTED_DIMS = 128
ROOT = os.path.dirname(os.path.abspath(__file__))


def rows(db_path, query):
    if not os.path.exists(db_path):
        return []
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute(query)]
    except sqlite3.Error as e:
        print(f"  ! could not read {db_path}: {e}")
        return []
    finally:
        conn.close()


def migrate(source_dir, photos_dir=None):
    users = rows(os.path.join(source_dir, "auth.db"), "SELECT * FROM users")
    faces = {f["enrollment_number"]: f
             for f in rows(os.path.join(source_dir, "faces.db"), "SELECT * FROM face_records")}

    if not users:
        print(f"No users found in {source_dir}. Nothing to migrate.")
        return

    print(f"Found {len(users)} user(s) in {source_dir}\n")
    db_manager.init_databases()

    imported = skipped = reencoded = 0

    for user in users:
        enrollment = user["enrollment_number"]
        name = user.get("name") or enrollment

        if db_manager.get_user_by_enrollment(enrollment):
            print(f"  = {enrollment} ({name}) already exists - left alone")
            skipped += 1
            continue

        record = faces.get(enrollment)
        encoding = None

        if record:
            candidate = json.loads(record["encoding_json"])
            if len(candidate) == EXPECTED_DIMS:
                encoding = candidate
            else:
                print(f"  ! {enrollment}: {len(candidate)}-dim encoding is from a "
                      f"different engine, cannot be reused")

        # Fall back to re-encoding from a photo when the stored vector is unusable.
        if encoding is None and photos_dir:
            for ext in (".jpg", ".jpeg", ".png"):
                photo = os.path.join(photos_dir, enrollment + ext)
                if os.path.exists(photo):
                    ok, fresh, err = face_engine.extract_encoding(photo)
                    if ok:
                        encoding = fresh
                        reencoded += 1
                        print(f"  ~ {enrollment}: re-encoded from {os.path.basename(photo)}")
                    else:
                        print(f"  ! {enrollment}: photo unusable - {err}")
                    break

        if encoding is None:
            print(f"  x {enrollment} ({name}) skipped - no usable face data")
            skipped += 1
            continue

        # Password hashes carry over unchanged: same hashing code on both sides,
        # so people keep the passwords they already have.
        conn = sqlite3.connect(db_manager.auth_db)
        conn.execute(
            "INSERT INTO users (enrollment_number, password_hash, name) VALUES (?, ?, ?)",
            (enrollment, user["password_hash"], name))
        conn.commit()
        conn.close()

        image_path = None
        source_image = record.get("image_path") if record else None
        if source_image:
            src = os.path.join(source_dir, source_image)
            if os.path.exists(src):
                os.makedirs(os.path.join(ROOT, "registered_faces"), exist_ok=True)
                dest_rel = os.path.join("registered_faces", enrollment + os.path.splitext(src)[1])
                shutil.copy(src, os.path.join(ROOT, dest_rel))
                image_path = dest_rel

        db_manager.store_face_record(enrollment, encoding, image_path=image_path)
        print(f"  + {enrollment} ({name}) imported"
              f"{' with photo' if image_path else ''}")
        imported += 1

    print(f"\nImported {imported}, re-encoded {reencoded}, skipped {skipped}.")
    print("Everyone imported keeps their existing password.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Import users from an older PROCTOR_APP install.")
    parser.add_argument("source", help="Path to the old app directory (containing auth.db and faces.db)")
    parser.add_argument("--photos", help="Directory of reference photos named <enrollment>.jpg, "
                                         "used when a stored encoding is unusable")
    args = parser.parse_args()

    if not os.path.isdir(args.source):
        sys.exit(f"Not a directory: {args.source}")

    migrate(args.source, args.photos)
