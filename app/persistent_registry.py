import numpy as np

from face_registry import FaceRegistry
from database import Database
from event_logger import EventLogger


class PersistentFaceRegistry:

    def __init__(
        self,
        database,
        event_logger=None,
        similarity_threshold=0.75,
        max_embeddings_per_face=5,
        confirmation_frames=3
    ):
        self.database = database
        self.event_logger = event_logger
        self.similarity_threshold = similarity_threshold
        self.max_embeddings_per_face = max_embeddings_per_face
        self.confirmation_frames = confirmation_frames

        self.registry = FaceRegistry(
            similarity_threshold=similarity_threshold,
            max_embeddings_per_face=max_embeddings_per_face
        )

        self.pending_tracks = {}

        self.load_faces_from_database()


    # ========================================================
    # LOAD REGISTERED FACES
    # ========================================================

    def load_faces_from_database(self):

        faces = self.database.get_all_faces()

        for face in faces:

            face_id = face["face_id"]

            embedding = np.asarray(
                face["embedding"],
                dtype=np.float32
            )

            if face_id not in self.registry.faces:

                self.registry.faces[face_id] = {
                    "embeddings": []
                }

            embeddings = self.registry.faces[
                face_id
            ]["embeddings"]

            embeddings.append(
                embedding
            )

            # Keep only the configured number
            # of embeddings in runtime memory.

            if (
                len(embeddings)
                > self.max_embeddings_per_face
            ):

                embeddings.pop(0)

            # Keep Face IDs sequential.

            try:

                number = int(
                    face_id.split("_")[1]
                )

                if number >= self.registry.next_id:

                    self.registry.next_id = (
                        number + 1
                    )

            except (
                IndexError,
                ValueError
            ):
                pass

        print(
            f"Loaded {len(faces)} "
            f"registered face embeddings "
            f"from database."
        )


    # ========================================================
    # FIND REGISTERED MATCH
    # ========================================================

    def find_registered_match(
        self,
        embedding
    ):

        return self.registry.find_match(
            embedding
        )


    # ========================================================
    # CALCULATE EMBEDDING CONSISTENCY
    # ========================================================

    def calculate_consistency(
        self,
        embeddings
    ):

        if len(embeddings) < 2:

            return 0.0

        similarities = []

        for i in range(
            len(embeddings)
        ):

            for j in range(
                i + 1,
                len(embeddings)
            ):

                similarity = (
                    self.registry.cosine_similarity(
                        embeddings[i],
                        embeddings[j]
                    )
                )

                similarities.append(
                    similarity
                )

        if not similarities:

            return 0.0

        return float(
            min(similarities)
        )


    # ========================================================
    # IDENTIFY OR REGISTER
    # ========================================================

    def identify_or_register(
        self,
        embedding,
        track_id=None
    ):

        embedding = np.asarray(
            embedding,
            dtype=np.float32
        )

        # ====================================================
        # EMBEDDING GENERATED
        # ====================================================

        if self.event_logger is not None:

            self.event_logger.embedding_generated()


        # ====================================================
        # STEP 1
        # SEARCH EXISTING IDENTITIES
        # ====================================================

        matched_face_id, similarity = (
            self.find_registered_match(
                embedding
            )
        )

        if matched_face_id is not None:

            # ------------------------------------------------
            # Add observation to runtime gallery
            # ------------------------------------------------

            self.registry.add_embedding(
                matched_face_id,
                embedding
            )

            # ------------------------------------------------
            # Persist embedding
            # ------------------------------------------------

            self.database.save_face_embedding(
                matched_face_id,
                embedding
            )

            # ------------------------------------------------
            # Remove pending state
            # ------------------------------------------------

            if track_id is not None:

                self.pending_tracks.pop(
                    track_id,
                    None
                )

            # ------------------------------------------------
            # Log recognition
            # ------------------------------------------------

            if self.event_logger is not None:

                self.event_logger.recognition(
                    face_id=matched_face_id,
                    similarity=similarity
                )

            print(
                f"Face recognized: "
                f"{matched_face_id} "
                f"(similarity={similarity:.3f})"
            )

            return (
                matched_face_id,
                False,
                similarity
            )


        # ====================================================
        # STEP 2
        # NO EXISTING IDENTITY MATCHED
        # ====================================================

        if track_id is None:

            face_id = (
                self.registry.register_face(
                    embedding
                )
            )

            self.database.save_face(
                face_id,
                embedding
            )

            # ------------------------------------------------
            # Log registration
            # ------------------------------------------------

            if self.event_logger is not None:

                self.event_logger.registration(
                    face_id
                )

            print(
                f"New face registered: "
                f"{face_id}"
            )

            return (
                face_id,
                True,
                1.0
            )


        # ====================================================
        # STEP 3
        # COLLECT TRACK OBSERVATIONS
        # ====================================================

        if track_id not in self.pending_tracks:

            self.pending_tracks[
                track_id
            ] = {
                "embeddings": []
            }

        pending = self.pending_tracks[
            track_id
        ]

        pending["embeddings"].append(
            embedding
        )

        # Keep only latest observations.

        if (
            len(pending["embeddings"])
            > self.confirmation_frames
        ):

            pending["embeddings"].pop(0)

        observation_count = len(
            pending["embeddings"]
        )

        print(
            f"Pending face | "
            f"Track {track_id} | "
            f"Observation "
            f"{observation_count}/"
            f"{self.confirmation_frames}"
        )


        # ====================================================
        # STEP 4
        # WAIT FOR ENOUGH OBSERVATIONS
        # ====================================================

        if (
            observation_count
            < self.confirmation_frames
        ):

            return (
                None,
                False,
                similarity
            )


        # ====================================================
        # STEP 5
        # CHECK CONSISTENCY
        # ====================================================

        consistency = (
            self.calculate_consistency(
                pending["embeddings"]
            )
        )

        print(
            f"Track {track_id} "
            f"embedding consistency: "
            f"{consistency:.3f}"
        )


        # ====================================================
        # STEP 6
        # REGISTER STABLE FACE
        # ====================================================

        minimum_consistency = 0.75

        if (
            consistency
            >= minimum_consistency
        ):

            embedding_matrix = np.vstack(
                pending["embeddings"]
            )

            # Average observations.

            final_embedding = (
                np.mean(
                    embedding_matrix,
                    axis=0
                )
            )

            # Normalize final embedding.

            norm = np.linalg.norm(
                final_embedding
            )

            if norm > 0:

                final_embedding = (
                    final_embedding / norm
                )

            # ------------------------------------------------
            # Register persistent identity
            # ------------------------------------------------

            face_id = (
                self.registry.register_face(
                    final_embedding
                )
            )

            # Save face.

            self.database.save_face(
                face_id,
                final_embedding
            )

            # ------------------------------------------------
            # Log registration
            # ------------------------------------------------

            if self.event_logger is not None:

                self.event_logger.registration(
                    face_id
                )

            # Remove pending state.

            del self.pending_tracks[
                track_id
            ]

            print(
                f"New face registered: "
                f"{face_id} "
                f"(Track {track_id})"
            )

            return (
                face_id,
                True,
                consistency
            )


        # ====================================================
        # STEP 7
        # STILL UNCERTAIN
        # ====================================================

        return (
            None,
            False,
            consistency
        )


    # ========================================================
    # GET FACE COUNT
    # ========================================================

    def get_face_count(self):

        return self.registry.get_face_count()