import numpy as np
import pytest

from backend.app.core.descriptors.mpeg7_loader import load_mpeg7_text


def test_mpeg7_text_loader_streams_float32_and_preserves_explicit_ids(tmp_path):
    source = tmp_path / "ColorLayout.txt"
    source.write_text("1, 2, 3\n4, 5, 6\n", encoding="utf-8")
    output = tmp_path / "descriptors" / "mpeg7_1m" / "ColorLayout.npy"

    matrix = load_mpeg7_text(source, output, delimiter=",",
                             image_ids=np.asarray([42, 17], dtype=np.int64))

    assert isinstance(matrix, np.memmap)
    assert matrix.dtype == np.float32
    assert matrix.tolist() == [[1, 2, 3], [4, 5, 6]]
    ids = np.load(output.parent / "ColorLayout" / "ids.npy")
    assert ids.tolist() == [42, 17]


def test_mpeg7_loader_rejects_implicit_ids(tmp_path):
    source = tmp_path / "values.txt"
    source.write_text("1 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Provide image_ids"):
        load_mpeg7_text(source, tmp_path / "values.npy")
