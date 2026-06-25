from __future__ import annotations

import httpx
import pytest
import respx
from avala import Client
from avala.importers import available_importers, detect_data_type, import_dataset, import_folder

BASE_URL = "https://api.avala.ai/api/v1"
PRESIGN_URL = f"{BASE_URL}/datasets/manual-upload/file-upload-url/"
FINALIZE_URL = f"{BASE_URL}/datasets/manual-upload/"
S3_URL = "https://s3.example.com/upload"


# ── pure logic ──
def test_detect_data_type_image():
    assert detect_data_type([("x/a.jpg", "a.jpg"), ("x/b.png", "b.png")]) == "image"


def test_detect_data_type_mcap():
    assert detect_data_type([("x/run.mcap", "run.mcap")]) == "mcap"


def test_detect_data_type_mixed_errors():
    with pytest.raises(ValueError, match="mixed data types"):
        detect_data_type([("a.jpg", "a.jpg"), ("b.mp4", "b.mp4")])


def test_detect_data_type_unknown_errors():
    with pytest.raises(ValueError, match="could not infer"):
        detect_data_type([("a.txt", "a.txt")])


def test_registry_has_folder():
    assert "folder" in available_importers()


def test_import_dataset_unknown_source_errors():
    with pytest.raises(ValueError, match="unknown importer"):
        import_dataset("nope", Client(api_key="k"))


# ── import_folder (respx: presign + S3 POST + finalize) ──
def _write_files(tmp_path, names):
    for n in names:
        (tmp_path / n).write_bytes(b"x" * 16)
    return tmp_path


def _wire_upload(dataset_json):
    respx.post(PRESIGN_URL).mock(
        return_value=httpx.Response(200, json={"url": S3_URL, "fields": {"key": "k", "Content-Type": "image/jpeg"}})
    )
    s3 = respx.post(S3_URL).mock(return_value=httpx.Response(204))
    respx.post(FINALIZE_URL).mock(return_value=httpx.Response(201, json=dataset_json))
    return s3


@respx.mock
def test_import_folder_auto_detects_and_creates(tmp_path):
    _write_files(tmp_path, ["a.jpg", "b.jpg"])
    s3 = _wire_upload({"uid": "d1", "name": "My Drive", "slug": "my-drive", "data_type": "image", "item_count": 2})

    client = Client(api_key="test-key")
    ds = import_folder(client, source=str(tmp_path), name="My Drive", slug="my-drive")
    client.close()

    assert ds.uid == "d1"
    assert ds.data_type == "image"
    assert s3.call_count == 2  # one S3 upload per file


@respx.mock
def test_import_folder_explicit_data_type(tmp_path):
    _write_files(tmp_path, ["scene.alp"])  # server-indexable LiDAR suffix
    _wire_upload({"uid": "d2", "name": "L", "slug": "l", "data_type": "lidar", "item_count": 1})

    client = Client(api_key="test-key")
    ds = import_folder(client, source=str(tmp_path), name="L", slug="l", data_type="lidar")
    client.close()
    assert ds.data_type == "lidar"


def test_import_folder_rejects_non_indexable_files(tmp_path):
    # .pcd/.bin/.las/.laz are NOT indexed by the server LiDAR filter (only .alp/.alp.gz).
    # Importing them would finalize an empty dataset, so the guard must reject up front.
    _write_files(tmp_path, ["cloud.pcd"])
    with pytest.raises(ValueError, match="indexable"):
        import_folder(Client(api_key="k"), source=str(tmp_path), name="x", slug="x", data_type="lidar")


def test_detect_data_type_ignores_non_indexable():
    # A folder of .pcd files no longer auto-detects as LiDAR (server can't index them).
    with pytest.raises(ValueError, match="could not infer"):
        detect_data_type([("a.pcd", "a.pcd"), ("b.bin", "b.bin")])


def test_detect_data_type_compressed_ply_is_splat():
    assert detect_data_type([("x/a.compressed.ply", "a.compressed.ply")]) == "splat"


@respx.mock
def test_import_dataset_dispatches_to_folder(tmp_path):
    _write_files(tmp_path, ["a.png"])
    _wire_upload({"uid": "d3", "name": "P", "slug": "p", "data_type": "image", "item_count": 1})
    client = Client(api_key="test-key")
    ds = import_dataset("folder", client, source=str(tmp_path), name="P", slug="p")
    client.close()
    assert ds.uid == "d3"


def test_import_folder_empty_errors(tmp_path):
    with pytest.raises(ValueError, match="no files found"):
        import_folder(Client(api_key="k"), source=str(tmp_path), name="x", slug="x")


# ── backbone: datasets.upload_files ──
@respx.mock
def test_upload_files_reports_progress(tmp_path):
    _write_files(tmp_path, ["a.jpg", "b.jpg"])
    respx.post(PRESIGN_URL).mock(
        return_value=httpx.Response(200, json={"url": S3_URL, "fields": {"Content-Type": "image/jpeg"}})
    )
    respx.post(S3_URL).mock(return_value=httpx.Response(204))

    seen = []
    client = Client(api_key="test-key")
    total = client.datasets.upload_files(
        dataset_name="My Drive",
        files=[(str(tmp_path / "a.jpg"), "a.jpg"), (str(tmp_path / "b.jpg"), "b.jpg")],
        workers=2,
        on_progress=lambda rel, n: seen.append(rel),
    )
    client.close()

    assert total == 32  # 2 files x 16 bytes
    assert set(seen) == {"a.jpg", "b.jpg"}


@respx.mock
def test_upload_files_fails_fast(tmp_path):
    _write_files(tmp_path, ["a.jpg"])
    respx.post(PRESIGN_URL).mock(return_value=httpx.Response(500, json={"detail": "boom"}))

    client = Client(api_key="test-key")
    with pytest.raises(Exception):
        client.datasets.upload_files(dataset_name="X", files=[(str(tmp_path / "a.jpg"), "a.jpg")], workers=1)
    client.close()
