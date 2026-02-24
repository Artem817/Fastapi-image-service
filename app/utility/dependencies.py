from fastapi import Request


def get_segmentation_model(request: Request):
    """
    FastAPI dependency that returns the preloaded segmentation model.
    The model is expected to be set on `app.state.segmentation_model` in lifespan.
    """
    return request.app.state.segmentation_model

