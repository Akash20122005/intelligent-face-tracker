from insightface.app import FaceAnalysis


def main():

    print()
    print("========================================")
    print(" InsightFace Model Inspection")
    print("========================================")

    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"]
    )

    app.prepare(
        ctx_id=-1,
        det_size=(320, 320)
    )

    print()
    print("Loaded models:")

    for name, model in app.models.items():

        print(
            f"\nModel name: {name}"
        )

        print(
            f"Class: {type(model).__name__}"
        )

        print(
            f"Has get_feat: "
            f"{hasattr(model, 'get_feat')}"
        )

        print(
            f"Has get: "
            f"{hasattr(model, 'get')}"
        )

    print()
    print("========================================")
    print("Inspection Completed")
    print("========================================")


if __name__ == "__main__":
    main()