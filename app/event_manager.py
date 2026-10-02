from datetime import datetime

from database import Database
from event_logger import EventLogger
from utils import save_face_crop


class EventManager:

    def __init__(
        self,
        database: Database,
        event_logger: EventLogger,
        max_missing_frames: int = 30
    ):
        self.database = database
        self.event_logger = event_logger
        self.max_missing_frames = max_missing_frames

        # =====================================================
        # Temporary Track ID state
        #
        # track_id -> {
        #     face_id,
        #     last_seen,
        #     last_face_crop
        # }
        # =====================================================

        self.active_tracks = {}

        # =====================================================
        # Persistent Face ID state
        #
        # face_id -> {
        #     active,
        #     last_seen,
        #     current_track_id
        # }
        #
        # This prevents duplicate ENTRY events when
        # ByteTrack changes the temporary track ID.
        # =====================================================

        self.active_faces = {}

    # =========================================================
    # PROCESS RECOGNIZED FACE
    # =========================================================

    def process_face(
        self,
        face_id,
        track_id,
        face_crop=None
    ):
        """
        Process a recognized face.

        Track ID = temporary tracking identity.

        Face ID = persistent visitor identity.

        A new Track ID does NOT automatically mean
        a new ENTRY if the same Face ID is already active.
        """

        current_time = datetime.now().isoformat()

        # =====================================================
        # CASE 1: TRACK ALREADY EXISTS
        # =====================================================

        if track_id in self.active_tracks:

            track_data = self.active_tracks[
                track_id
            ]

            # Reset missing counter
            track_data["last_seen"] = 0

            # Update latest face crop
            if face_crop is not None:

                track_data["last_face_crop"] = (
                    face_crop
                )

            # Keep Face ID mapping updated
            track_data["face_id"] = face_id

            # Update persistent face state
            if face_id in self.active_faces:

                self.active_faces[
                    face_id
                ]["last_seen"] = 0

                self.active_faces[
                    face_id
                ]["current_track_id"] = track_id

            # Mandatory tracking log
            self.event_logger.tracking(
                face_id=face_id,
                track_id=track_id
            )

            return

        # =====================================================
        # CASE 2: NEW TRACK ID
        # =====================================================

        # -----------------------------------------------------
        # Check whether this Face ID is already active.
        # -----------------------------------------------------

        if face_id in self.active_faces:

            face_data = self.active_faces[
                face_id
            ]

            # Same visitor, new temporary track
            print(
                f"[RE-TRACK] "
                f"{face_id} | "
                f"Old Track: "
                f"{face_data['current_track_id']} | "
                f"New Track: {track_id}"
            )

            # Create new temporary track state
            self.active_tracks[
                track_id
            ] = {
                "face_id": face_id,
                "last_seen": 0,
                "last_face_crop": face_crop
            }

            # Update persistent Face ID
            face_data[
                "last_seen"
            ] = 0

            face_data[
                "current_track_id"
            ] = track_id

            # Mandatory tracking log
            self.event_logger.tracking(
                face_id=face_id,
                track_id=track_id
            )

            # IMPORTANT:
            # No ENTRY event here.
            #
            # The visitor is already considered
            # present.

            return

        # =====================================================
        # CASE 3: COMPLETELY NEW FACE
        # =====================================================

        image_path = None

        # -----------------------------------------------------
        # Save ENTRY face crop
        # -----------------------------------------------------

        if face_crop is not None:

            image_path = save_face_crop(
                face_crop,
                face_id,
                "ENTRY"
            )

        # -----------------------------------------------------
        # Save ENTRY to database
        # -----------------------------------------------------

        self.database.save_event(
            face_id=face_id,
            event_type="ENTRY",
            track_id=track_id,
            image_path=image_path
        )

        # -----------------------------------------------------
        # Write ENTRY to events.log
        # -----------------------------------------------------

        self.event_logger.entry(
            face_id=face_id,
            track_id=track_id,
            image_path=image_path
        )

        # -----------------------------------------------------
        # Store temporary track
        # -----------------------------------------------------

        self.active_tracks[
            track_id
        ] = {
            "face_id": face_id,
            "last_seen": 0,
            "last_face_crop": face_crop
        }

        # -----------------------------------------------------
        # Store persistent Face ID
        # -----------------------------------------------------

        self.active_faces[
            face_id
        ] = {
            "active": True,
            "last_seen": 0,
            "current_track_id": track_id
        }

        print(
            f"[ENTRY] "
            f"{face_id} | "
            f"Track ID: {track_id} | "
            f"{current_time}"
        )

    # =========================================================
    # UPDATE MISSING TRACKS
    # =========================================================

    def update_missing_tracks(self):
        """
        Update tracks that are not visible.

        A temporary Track ID can disappear while the same
        persistent Face ID may still be visible under a new
        Track ID.

        Therefore:

            Track disappears
                    ↓
            Do NOT immediately EXIT
                    ↓
            Wait max_missing_frames
                    ↓
            EXIT visitor if still absent
        """

        tracks_to_remove = []

        # =====================================================
        # CHECK TEMPORARY TRACKS
        # =====================================================

        for track_id, track_data in list(
            self.active_tracks.items()
        ):

            # Increase missing counter
            track_data[
                "last_seen"
            ] += 1

            # -------------------------------------------------
            # Track missing
            # -------------------------------------------------

            if (
                track_data["last_seen"]
                >= self.max_missing_frames
            ):

                face_id = track_data[
                    "face_id"
                ]

                # ---------------------------------------------
                # Save EXIT only if Face ID is still associated
                # with this track.
                # ---------------------------------------------

                face_data = self.active_faces.get(
                    face_id
                )

                if face_data is not None:

                    current_track_id = (
                        face_data[
                            "current_track_id"
                        ]
                    )

                    # -----------------------------------------
                    # If the Face ID has already moved to
                    # another Track ID, this old track must
                    # NOT generate EXIT.
                    # -----------------------------------------

                    if current_track_id != track_id:

                        tracks_to_remove.append(
                            track_id
                        )

                        continue

                # =============================================
                # TRUE VISITOR EXIT
                # =============================================

                image_path = None

                last_face_crop = track_data.get(
                    "last_face_crop"
                )

                if last_face_crop is not None:

                    image_path = save_face_crop(
                        last_face_crop,
                        face_id,
                        "EXIT"
                    )

                # ---------------------------------------------
                # Database
                # ---------------------------------------------

                self.database.save_event(
                    face_id=face_id,
                    event_type="EXIT",
                    track_id=track_id,
                    image_path=image_path
                )

                # ---------------------------------------------
                # Event log
                # ---------------------------------------------

                self.event_logger.exit(
                    face_id=face_id,
                    track_id=track_id,
                    image_path=image_path
                )

                current_time = (
                    datetime.now().isoformat()
                )

                print(
                    f"[EXIT] "
                    f"{face_id} | "
                    f"Track ID: {track_id} | "
                    f"{current_time}"
                )

                # ---------------------------------------------
                # Remove persistent Face ID
                # ---------------------------------------------

                if face_id in self.active_faces:

                    del self.active_faces[
                        face_id
                    ]

                tracks_to_remove.append(
                    track_id
                )

        # =====================================================
        # REMOVE FINISHED TRACKS
        # =====================================================

        for track_id in tracks_to_remove:

            if track_id in self.active_tracks:

                del self.active_tracks[
                    track_id
                ]

    # =========================================================
    # FINALIZE ACTIVE VISITORS
    # =========================================================

    def finalize_active_visitors(self):
        """
        Generate EXIT events for visitors who are still active
        when the video or RTSP stream ends.
        """

        if not self.active_faces:

            print(
                "No active visitors to finalize."
            )

            return

        print()
        print(
            "Finalizing active visitors..."
        )

        processed_faces = set()

        # =====================================================
        # PROCESS EACH ACTIVE FACE
        # =====================================================

        for face_id, face_data in list(
            self.active_faces.items()
        ):

            # Safety check to prevent duplicate EXIT
            if face_id in processed_faces:

                continue

            track_id = face_data.get(
                "current_track_id"
            )

            if track_id is None:

                continue

            track_data = self.active_tracks.get(
                track_id
            )

            image_path = None

            # -------------------------------------------------
            # Get latest face crop
            # -------------------------------------------------

            if track_data is not None:

                last_face_crop = track_data.get(
                    "last_face_crop"
                )

                if last_face_crop is not None:

                    image_path = save_face_crop(
                        last_face_crop,
                        face_id,
                        "EXIT"
                    )

            # -------------------------------------------------
            # Save EXIT event to database
            # -------------------------------------------------

            self.database.save_event(
                face_id=face_id,
                event_type="EXIT",
                track_id=track_id,
                image_path=image_path
            )

            # -------------------------------------------------
            # Write EXIT to events.log
            # -------------------------------------------------

            self.event_logger.exit(
                face_id=face_id,
                track_id=track_id,
                image_path=image_path
            )

            print(
                f"[FINAL EXIT] "
                f"{face_id} | "
                f"Track ID: {track_id}"
            )

            processed_faces.add(
                face_id
            )

        # =====================================================
        # CLEAR ACTIVE STATE
        # =====================================================

        self.active_tracks.clear()

        self.active_faces.clear()

    # =========================================================
    # ACTIVE VISITOR COUNT
    # =========================================================

    def get_active_count(self):

        return len(
            self.active_faces
        )

    # =========================================================
    # ACTIVE FACES
    # =========================================================

    def get_active_faces(self):

        return self.active_faces.copy()