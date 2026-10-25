"""Available descriptor metadata."""
from fastapi import APIRouter
from backend.app.config import get_settings
from backend.app.core.descriptors import DESCRIPTORS, MPEG7_PROVIDED_DESCRIPTORS
router = APIRouter()

@router.get("/descriptors")
def descriptors(dataset: str = "objects"):
    names = list(DESCRIPTORS) if dataset == "objects" else sorted(MPEG7_PROVIDED_DESCRIPTORS)
    root = get_settings().data_dir
    vgg_artifacts = (root / "descriptors" / dataset / "vgg16.npy",
                     root / "descriptors" / dataset / "vgg16" / "ids.npy",
                     root / "indexes" / dataset / "vgg16.npz")
    if dataset != "objects" and all(path.exists() for path in vgg_artifacts):
        names.append("vgg16")
    available = []
    for name in names:
        artifacts = (root / "descriptors" / dataset / f"{name}.npy",
                     root / "descriptors" / dataset / name / "ids.npy",
                     root / "indexes" / dataset / f"{name}.npz")
        if all(path.exists() for path in artifacts):
            metric = DESCRIPTORS[name].metric if name in DESCRIPTORS else "l2"
            # The persisted tree metric reflects build-time per-descriptor overrides.
            try:
                import numpy as np
                with np.load(artifacts[2], allow_pickle=False) as index:
                    metric = str(index["metric"].item())
            except (OSError, KeyError, ValueError):
                pass
            available.append({"name": name, "metric": metric,
                              "kind": "deep" if name == "vgg16" else "handcrafted"})
    return {"dataset": dataset, "descriptors": available}
