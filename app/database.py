import sqlite3
import numpy as np
from datetime import datetime


class Database:

    def __init__(self, database_path):

        self.database_path = database_path

        self.connection = sqlite3.connect(
            database_path
        )

        self.create_tables()


    # ========================================================
    # CREATE TABLES
    # ========================================================

    def create_tables(self):

        cursor = self.connection.cursor()

        # ----------------------------------------------------
        # Persistent faces
        # ----------------------------------------------------

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                face_id TEXT UNIQUE NOT NULL,
                created_at TEXT NOT NULL
            )
        """)

        # ----------------------------------------------------
        # Multiple embeddings per Face ID
        # ----------------------------------------------------

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS face_embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                face_id TEXT NOT NULL,
                embedding BLOB NOT NULL,
                embedding_dimension INTEGER NOT NULL,
                created_at TEXT NOT NULL,

                FOREIGN KEY (face_id)
                REFERENCES faces(face_id)
            )
        """)

        # ----------------------------------------------------
        # Entry / Exit events
        # ----------------------------------------------------

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                face_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                track_id INTEGER,
                image_path TEXT
            )
        """)

        self.connection.commit()


    # ========================================================
    # SAVE NEW FACE
    # ========================================================

    def save_face(
        self,
        face_id,
        embedding
    ):

        embedding = np.asarray(
            embedding,
            dtype=np.float32
        )

        timestamp = datetime.now().isoformat()

        cursor = self.connection.cursor()

        # ----------------------------------------------------
        # Create persistent face record
        # ----------------------------------------------------

        cursor.execute("""
            INSERT OR IGNORE INTO faces (
                face_id,
                created_at
            )
            VALUES (?, ?)
        """, (
            face_id,
            timestamp
        ))

        # ----------------------------------------------------
        # Save first embedding
        # ----------------------------------------------------

        cursor.execute("""
            INSERT INTO face_embeddings (
                face_id,
                embedding,
                embedding_dimension,
                created_at
            )
            VALUES (?, ?, ?, ?)
        """, (
            face_id,
            embedding.tobytes(),
            embedding.size,
            timestamp
        ))

        self.connection.commit()


    # ========================================================
    # SAVE ADDITIONAL EMBEDDING
    # ========================================================

    def save_face_embedding(
        self,
        face_id,
        embedding
    ):

        embedding = np.asarray(
            embedding,
            dtype=np.float32
        )

        timestamp = datetime.now().isoformat()

        cursor = self.connection.cursor()

        # Make sure the Face ID exists.

        cursor.execute("""
            INSERT OR IGNORE INTO faces (
                face_id,
                created_at
            )
            VALUES (?, ?)
        """, (
            face_id,
            timestamp
        ))

        # ----------------------------------------------------
        # Save additional embedding
        # ----------------------------------------------------

        cursor.execute("""
            INSERT INTO face_embeddings (
                face_id,
                embedding,
                embedding_dimension,
                created_at
            )
            VALUES (?, ?, ?, ?)
        """, (
            face_id,
            embedding.tobytes(),
            embedding.size,
            timestamp
        ))

        self.connection.commit()


    # ========================================================
    # LOAD ALL FACE EMBEDDINGS
    # ========================================================

    def get_all_faces(self):

        cursor = self.connection.cursor()

        cursor.execute("""
            SELECT
                face_id,
                embedding,
                embedding_dimension
            FROM face_embeddings
            ORDER BY id
        """)

        rows = cursor.fetchall()

        faces = []

        for (
            face_id,
            embedding_blob,
            dimension
        ) in rows:

            embedding = np.frombuffer(
                embedding_blob,
                dtype=np.float32
            )

            faces.append({
                "face_id": face_id,
                "embedding": embedding,
                "dimension": dimension
            })

        return faces


    # ========================================================
    # GET EMBEDDING COUNT FOR FACE
    # ========================================================

    def get_embedding_count(
        self,
        face_id
    ):

        cursor = self.connection.cursor()

        cursor.execute("""
            SELECT COUNT(*)
            FROM face_embeddings
            WHERE face_id = ?
        """, (
            face_id,
        ))

        result = cursor.fetchone()

        return result[0]


    # ========================================================
    # GET UNIQUE VISITOR COUNT
    # ========================================================

    def get_unique_visitor_count(self):

        cursor = self.connection.cursor()

        cursor.execute("""
            SELECT COUNT(*)
            FROM faces
        """)

        result = cursor.fetchone()

        return result[0]


    # ========================================================
    # SAVE EVENT
    # ========================================================

    def save_event(
        self,
        face_id,
        event_type,
        track_id=None,
        image_path=None
    ):

        timestamp = datetime.now().isoformat()

        cursor = self.connection.cursor()

        cursor.execute("""
            INSERT INTO events (
                face_id,
                event_type,
                timestamp,
                track_id,
                image_path
            )
            VALUES (?, ?, ?, ?, ?)
        """, (
            face_id,
            event_type,
            timestamp,
            track_id,
            image_path
        ))

        self.connection.commit()


    # ========================================================
    # GET EVENTS FOR FACE
    # ========================================================

    def get_face_events(
        self,
        face_id
    ):

        cursor = self.connection.cursor()

        cursor.execute("""
            SELECT
                face_id,
                event_type,
                timestamp,
                track_id,
                image_path
            FROM events
            WHERE face_id = ?
            ORDER BY id
        """, (
            face_id,
        ))

        rows = cursor.fetchall()

        events = []

        for row in rows:

            events.append({
                "face_id": row[0],
                "event_type": row[1],
                "timestamp": row[2],
                "track_id": row[3],
                "image_path": row[4]
            })

        return events


    # ========================================================
    # CLOSE DATABASE
    # ========================================================

    def close(self):

        self.connection.close()