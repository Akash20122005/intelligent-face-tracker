import os
import sys
import tempfile
import unittest

import numpy as np

# Allow imports from app/
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "app"
        )
    )
)

from face_registry import FaceRegistry
from database import Database


class TestFaceRegistry(unittest.TestCase):

    def setUp(self):
        self.registry = FaceRegistry(
            similarity_threshold=0.75,
            max_embeddings_per_face=5
        )

    def test_cosine_similarity_identical(self):
        embedding = np.array(
            [1.0, 0.0, 0.0],
            dtype=np.float32
        )

        similarity = self.registry.cosine_similarity(
            embedding,
            embedding
        )

        self.assertAlmostEqual(
            similarity,
            1.0,
            places=5
        )

    def test_cosine_similarity_orthogonal(self):
        embedding1 = np.array(
            [1.0, 0.0, 0.0],
            dtype=np.float32
        )

        embedding2 = np.array(
            [0.0, 1.0, 0.0],
            dtype=np.float32
        )

        similarity = self.registry.cosine_similarity(
            embedding1,
            embedding2
        )

        self.assertAlmostEqual(
            similarity,
            0.0,
            places=5
        )

    def test_register_face(self):
        embedding = np.array(
            [1.0, 0.0, 0.0],
            dtype=np.float32
        )

        face_id = self.registry.register_face(
            embedding
        )

        self.assertEqual(
            face_id,
            "FACE_000001"
        )

        self.assertEqual(
            self.registry.get_face_count(),
            1
        )

    def test_matching_registered_face(self):
        embedding = np.array(
            [1.0, 0.0, 0.0],
            dtype=np.float32
        )

        face_id = self.registry.register_face(
            embedding
        )

        matched_id, similarity = (
            self.registry.find_match(
                embedding
            )
        )

        self.assertEqual(
            matched_id,
            face_id
        )

        self.assertAlmostEqual(
            similarity,
            1.0,
            places=5
        )

    def test_unknown_face(self):
        embedding1 = np.array(
            [1.0, 0.0, 0.0],
            dtype=np.float32
        )

        embedding2 = np.array(
            [0.0, 1.0, 0.0],
            dtype=np.float32
        )

        self.registry.register_face(
            embedding1
        )

        matched_id, similarity = (
            self.registry.find_match(
                embedding2
            )
        )

        self.assertIsNone(
            matched_id
        )

        self.assertLess(
            similarity,
            0.75
        )


class TestDatabase(unittest.TestCase):

    def setUp(self):
        self.temp_directory = tempfile.TemporaryDirectory()

        self.database_path = os.path.join(
            self.temp_directory.name,
            "test.db"
        )

        self.database = Database(
            self.database_path
        )

    def tearDown(self):
        self.database.close()
        self.temp_directory.cleanup()

    def test_save_face(self):
        embedding = np.array(
            [1.0, 0.0, 0.0],
            dtype=np.float32
        )

        self.database.save_face(
            "FACE_000001",
            embedding
        )

        faces = self.database.get_all_faces()

        self.assertEqual(
            len(faces),
            1
        )

        self.assertEqual(
            faces[0]["face_id"],
            "FACE_000001"
        )

    def test_unique_visitor_count(self):
        embedding1 = np.array(
            [1.0, 0.0, 0.0],
            dtype=np.float32
        )

        embedding2 = np.array(
            [0.0, 1.0, 0.0],
            dtype=np.float32
        )

        self.database.save_face(
            "FACE_000001",
            embedding1
        )

        self.database.save_face(
            "FACE_000002",
            embedding2
        )

        count = (
            self.database.get_unique_visitor_count()
        )

        self.assertEqual(
            count,
            2
        )

    def test_save_and_get_event(self):
        embedding = np.array(
            [1.0, 0.0, 0.0],
            dtype=np.float32
        )

        self.database.save_face(
            "FACE_000001",
            embedding
        )

        self.database.save_event(
            face_id="FACE_000001",
            event_type="ENTRY",
            track_id=10,
            image_path="logs/entries/test.jpg"
        )

        events = (
            self.database.get_face_events(
                "FACE_000001"
            )
        )

        self.assertEqual(
            len(events),
            1
        )

        self.assertEqual(
            events[0]["event_type"],
            "ENTRY"
        )

        self.assertEqual(
            events[0]["track_id"],
            10
        )


if __name__ == "__main__":
    unittest.main(
        verbosity=2
    )