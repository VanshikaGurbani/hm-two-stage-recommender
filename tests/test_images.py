import zipfile

from hmrec.images import ImageStore, image_member


def test_member_path_pads_to_ten_digits():
    assert image_member(108775015) == "images/010/0108775015.jpg"
    assert image_member(959461001) == "images/095/0959461001.jpg"


def test_reads_from_zip_without_extracting(tmp_path):
    archive = tmp_path / "h-and-m-personalized-fashion-recommendations.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("images/010/0108775015.jpg", b"jpeg-bytes")
        zf.writestr("articles.csv", "x")
    store = ImageStore(folders=(tmp_path / "none",), archive=archive)
    assert store.available and store.source == "competition zip"
    assert store.get(108775015) == b"jpeg-bytes"
    assert store.get(123) is None
    assert not (tmp_path / "images").exists()


def test_reads_from_extracted_folder(tmp_path):
    d = tmp_path / "images" / "010"
    d.mkdir(parents=True)
    (d / "0108775015.jpg").write_bytes(b"abc")
    store = ImageStore(folders=(tmp_path,))
    assert store.source.startswith("folder") and store.get(108775015) == b"abc"
    assert store.get(1) is None
