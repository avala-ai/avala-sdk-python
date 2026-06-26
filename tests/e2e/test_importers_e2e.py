"""End-to-end importer tests against a REAL Avala server.

Unlike the unit tests (which mock the network boundary with respx), these run each
importer against a live server, wait for server-side indexing, and assert the dataset
actually materialized (status ``created``, expected ``item_count`` / ``data_type``).

The whole module is **skipped** unless ``AVALA_E2E_API_KEY`` is set, so it never runs in
the default CI / ``pytest`` path.

Run against a local server:

    cd server && python manage.py runserver 0.0.0.0:8000     # in another shell

    export AVALA_E2E_API_KEY="avk_..."
    export AVALA_E2E_BASE_URL="http://localhost:8000/api/v1"  # default
    export AVALA_ALLOW_INSECURE_BASE_URL=true                 # required for http://localhost
    pip install -e ".[dev,cli,lerobot,rosbag]"
    pytest tests/e2e -v

Optional, per-importer:

    # cloud (S3) — point at a small bucket of images you control
    export AVALA_E2E_S3_URI="s3://my-test-bucket/sample-images/"
    export AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... AWS_REGION=us-west-2

    # LeRobot — a SMALL public Hugging Face dataset (download can be large/slow)
    export AVALA_E2E_LEROBOT_REPO="lerobot/<small_dataset>"
"""

from __future__ import annotations

import os
import uuid
from io import BytesIO

import pytest

E2E_API_KEY = os.environ.get("AVALA_E2E_API_KEY")
E2E_BASE_URL = os.environ.get("AVALA_E2E_BASE_URL", "http://localhost:8000/api/v1")

pytestmark = pytest.mark.skipif(
    not E2E_API_KEY,
    reason="set AVALA_E2E_API_KEY (and AVALA_E2E_BASE_URL) to run importer E2E against a server",
)

# A generous ceiling — real server-side indexing (download + extract + thumbnail) is slow.
WAIT_TIMEOUT = float(os.environ.get("AVALA_E2E_WAIT_TIMEOUT", "600"))


def _unique(prefix: str) -> str:
    """A collision-proof name/slug so repeated runs don't clash."""
    return f"{prefix}-e2e-{uuid.uuid4().hex[:8]}"


@pytest.fixture(scope="module")
def client():
    from avala import Client

    # The SDK rejects non-HTTPS base URLs unless this is set; localhost http needs it.
    if E2E_BASE_URL.startswith("http://"):
        os.environ.setdefault("AVALA_ALLOW_INSECURE_BASE_URL", "true")
    c = Client(api_key=E2E_API_KEY, base_url=E2E_BASE_URL)
    yield c
    c.close()


def _assert_ingested(dataset, *, data_type: str, min_items: int = 1) -> None:
    assert dataset is not None
    assert dataset.data_type == data_type
    # import_*(wait=True) returns the dataset refreshed after indexing finishes.
    assert (dataset.item_count or 0) >= min_items, f"expected >= {min_items} items, got {dataset.item_count}"


# folder/rosbag use the manual-upload flow, which the server only serves when
# MANUAL_DATASET_UPLOADS_BUCKET_NAME (+ AWS creds) is configured — otherwise the
# presign endpoint 500s. Gate them so the default E2E run (and CI without an upload
# bucket) stays green; set AVALA_E2E_UPLOADS=1 once the server has a bucket wired.
needs_uploads = pytest.mark.skipif(
    not os.environ.get("AVALA_E2E_UPLOADS"),
    reason="needs the manual-upload S3 bucket configured on the server; set AVALA_E2E_UPLOADS=1",
)


# ── folder: a local image directory ────────────────────────────────────────────
@needs_uploads
def test_e2e_folder_images(client, tmp_path):
    pytest.importorskip("PIL")
    from avala.importers import import_folder
    from PIL import Image

    for i in range(3):
        Image.new("RGB", (16, 16), (i * 40, 80, 120)).save(tmp_path / f"frame_{i}.jpg")

    slug = _unique("folder")
    ds = import_folder(client, source=str(tmp_path), name=slug, slug=slug, wait=True, wait_timeout=WAIT_TIMEOUT)
    _assert_ingested(ds, data_type="image", min_items=3)


# ── rosbag: a generated ROS2 bag with one camera topic ──────────────────────────
@needs_uploads
def test_e2e_rosbag(client, tmp_path):
    pytest.importorskip("avala.importers.rosbag")  # skip if the rosbag importer isn't released yet
    pytest.importorskip("rosbags.rosbag2")
    pytest.importorskip("mcap_protobuf.writer")
    import numpy as np
    from avala.importers import import_ros_bag
    from PIL import Image
    import inspect

    from rosbags.rosbag2 import Writer
    from rosbags.typesys import Stores, get_typestore

    ts = get_typestore(Stores.ROS2_HUMBLE)
    Header, Time = ts.types["std_msgs/msg/Header"], ts.types["builtin_interfaces/msg/Time"]
    CompressedImage = ts.types["sensor_msgs/msg/CompressedImage"]

    bag_dir = tmp_path / "bag"
    # rosbags >=0.10 made Writer's `version` kwarg required (older versions don't
    # accept it); pass it only when present so this works across versions.
    wkw = {}
    if "version" in inspect.signature(Writer.__init__).parameters:
        wkw["version"] = getattr(Writer, "VERSION_LATEST", 9)
    writer = Writer(bag_dir, **wkw)
    writer.open()
    try:
        conn = writer.add_connection("/camera/front/compressed", "sensor_msgs/msg/CompressedImage", typestore=ts)
        for i in range(3):
            buf = BytesIO()
            Image.new("RGB", (32, 32), (i * 30, 60, 90)).save(buf, format="JPEG")
            msg = CompressedImage(
                header=Header(stamp=Time(sec=i, nanosec=0), frame_id="front"),
                format="jpeg",
                data=np.frombuffer(buf.getvalue(), dtype=np.uint8),
            )
            writer.write(conn, (i + 1) * 1_000_000_000, bytes(ts.serialize_cdr(msg, "sensor_msgs/msg/CompressedImage")))
    finally:
        writer.close()

    slug = _unique("rosbag")
    ds = import_ros_bag(client, bag=str(bag_dir), name=slug, slug=slug, wait=True, wait_timeout=WAIT_TIMEOUT)
    _assert_ingested(ds, data_type="mcap", min_items=1)


# ── cloud: an existing S3 bucket (zero-copy) ────────────────────────────────────
@pytest.mark.skipif(not os.environ.get("AVALA_E2E_S3_URI"), reason="set AVALA_E2E_S3_URI (+ AWS_* env) for cloud E2E")
def test_e2e_cloud_s3(client):
    from avala.importers import import_cloud

    uri = os.environ["AVALA_E2E_S3_URI"]
    data_type = os.environ.get("AVALA_E2E_S3_DATA_TYPE", "image")
    slug = _unique("cloud-s3")
    ds = import_cloud(
        client,
        uri=uri,
        name=slug,
        slug=slug,
        data_type=data_type,
        region=os.environ.get("AWS_REGION"),
        access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        organization_uid=os.environ.get("AVALA_E2E_ORG_UID"),
        wait=True,
        wait_timeout=WAIT_TIMEOUT,
    )
    _assert_ingested(ds, data_type=data_type, min_items=1)


# ── lerobot: a small public Hugging Face dataset ────────────────────────────────
@pytest.mark.skipif(
    not os.environ.get("AVALA_E2E_LEROBOT_REPO"), reason="set AVALA_E2E_LEROBOT_REPO to a small HF dataset for E2E"
)
def test_e2e_lerobot(client):
    pytest.importorskip("lerobot")
    from avala.importers import import_lerobot

    repo = os.environ["AVALA_E2E_LEROBOT_REPO"]
    episodes = os.environ.get("AVALA_E2E_LEROBOT_EPISODES")
    slug = _unique("lerobot")
    ds = import_lerobot(
        client,
        repo_id=repo,
        name=slug,
        slug=slug,
        episodes=[int(e) for e in episodes.split(",")] if episodes else [0],
        wait=True,
        wait_timeout=WAIT_TIMEOUT,
    )
    _assert_ingested(ds, data_type="mcap", min_items=1)
