import numpy as np


class FaceRegistry:

    def __init__(
        self,
        similarity_threshold=0.75,
        max_embeddings_per_face=5
    ):
        """
        In-memory face registry.

        Each Face ID can store multiple embeddings.
        """

        self.similarity_threshold = similarity_threshold
        self.max_embeddings_per_face = max_embeddings_per_face

        # Example:
        #
        # FACE_000001:
        #     embeddings = [embedding1, embedding2, ...]
        #
        self.faces = {}

        # Next Face ID number
        self.next_id = 1

    # ========================================================
    # COSINE SIMILARITY
    # ========================================================

    def cosine_similarity(
        self,
        embedding1,
        embedding2
    ):
        """
        Calculate cosine similarity between two embeddings.
        """

        embedding1 = np.asarray(
            embedding1,
            dtype=np.float32
        )

        embedding2 = np.asarray(
            embedding2,
            dtype=np.float32
        )

        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = np.dot(
            embedding1,
            embedding2
        ) / (norm1 * norm2)

        return float(similarity)

    # ========================================================
    # REGISTER NEW FACE
    # ========================================================

    def register_face(self, embedding):
        """
        Register a completely new face.
        """

        face_id = f"FACE_{self.next_id:06d}"

        embedding = np.asarray(
            embedding,
            dtype=np.float32
        )

        self.faces[face_id] = {
            "embeddings": [embedding]
        }

        self.next_id += 1

        return face_id

    # ========================================================
    # ADD EMBEDDING TO EXISTING FACE
    # ========================================================

    def add_embedding(
        self,
        face_id,
        embedding
    ):
        """
        Add a new embedding to an existing Face ID.

        A maximum number of embeddings is maintained
        for each person.
        """

        if face_id not in self.faces:
            return

        embedding = np.asarray(
            embedding,
            dtype=np.float32
        )

        embeddings = self.faces[
            face_id
        ]["embeddings"]

        # Avoid storing nearly identical embeddings
        if embeddings:

            highest_similarity = max(
                self.cosine_similarity(
                    embedding,
                    existing_embedding
                )
                for existing_embedding
                in embeddings
            )

            # If almost identical, don't store duplicate
            if highest_similarity >= 0.98:
                return

        embeddings.append(embedding)

        # Keep only the latest N embeddings
        if len(embeddings) > self.max_embeddings_per_face:

            embeddings.pop(0)

    # ========================================================
    # FIND BEST MATCH
    # ========================================================

    def find_match(self, embedding):
        """
        Compare an embedding against all stored
        embeddings of all registered faces.

        Returns:

            face_id
            best similarity

        or:

            None
            best similarity
        """

        if not self.faces:
            return None, 0.0

        best_face_id = None
        best_similarity = -1.0

        for face_id, data in self.faces.items():

            face_embeddings = data["embeddings"]

            for stored_embedding in face_embeddings:

                similarity = self.cosine_similarity(
                    embedding,
                    stored_embedding
                )

                if similarity > best_similarity:

                    best_similarity = similarity
                    best_face_id = face_id

        # Check threshold
        if (
            best_face_id is not None
            and best_similarity >= self.similarity_threshold
        ):

            return (
                best_face_id,
                best_similarity
            )

        return (
            None,
            best_similarity
        )

    # ========================================================
    # IDENTIFY OR REGISTER
    # ========================================================

    def identify_or_register(
        self,
        embedding
    ):
        """
        Identify an existing person or register
        a new person.

        Returns:

            face_id
            is_new
            similarity
        """

        # ----------------------------------------------------
        # Search existing registry
        # ----------------------------------------------------

        face_id, similarity = self.find_match(
            embedding
        )

        # ----------------------------------------------------
        # Existing person
        # ----------------------------------------------------

        if face_id is not None:

            # Add this embedding to the person's gallery
            self.add_embedding(
                face_id,
                embedding
            )

            return (
                face_id,
                False,
                similarity
            )

        # ----------------------------------------------------
        # Completely new person
        # ----------------------------------------------------

        face_id = self.register_face(
            embedding
        )

        return (
            face_id,
            True,
            1.0
        )

    # ========================================================
    # GET FACE COUNT
    # ========================================================

    def get_face_count(self):
        """
        Return number of unique registered faces.
        """

        return len(self.faces)

    # ========================================================
    # GET EMBEDDING COUNT
    # ========================================================

    def get_embedding_count(
        self,
        face_id
    ):
        """
        Return number of stored embeddings
        for a particular Face ID.
        """

        if face_id not in self.faces:
            return 0

        return len(
            self.faces[face_id]["embeddings"]
        )