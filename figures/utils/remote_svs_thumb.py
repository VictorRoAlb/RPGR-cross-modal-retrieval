"""
remote_svs_thumb.py
=======================
Extracts a WSI's embedded thumbnail directly from GDC without downloading
the full slide -- uses HTTP Range requests (api.gdc.cancer.gov supports
HTTP 206 partial content) through a stdlib-only file-like object that
tifffile can read directly, exploiting the fact that Aperio SVS files store
their associated images (thumbnail/label/macro) as ordinary TIFF pages
inside the same multi-page container as the full-resolution slide.

Verified byte-for-byte identical against a fully downloaded .svs before
being trusted for new cases (see METHOD_PROVENANCE.md / the paper's
qualitative retrieval example). Only reads from GDC; writes nothing there.
"""
import io
import json
import time
import urllib.parse
import urllib.request

import tifffile


class HTTPRangeFile(io.RawIOBase):
    """Read-only file-like object over a URL, using Range requests (stdlib
    urllib only). tifffile only needs read()/seek()/tell() to parse a TIFF's
    IFDs -- it never needs the whole file downloaded sequentially."""

    def __init__(self, url: str):
        self.url = url
        self.pos = 0
        req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
        last_err = None
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    content_range = resp.headers["Content-Range"]  # "bytes 0-0/TOTAL"
                    self.size = int(content_range.split("/")[-1])
                return
            except Exception as e:  # noqa: BLE001 -- deliberate retry on transient network errors
                last_err = e
                time.sleep(1.5 * (attempt + 1))
        raise last_err

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        elif whence == io.SEEK_END:
            self.pos = self.size + offset
        return self.pos

    def tell(self) -> int:
        return self.pos

    def readinto(self, b) -> int:
        n = len(b)
        if n == 0 or self.pos >= self.size:
            return 0
        end = min(self.pos + n - 1, self.size - 1)
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        last_err = None
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=90) as resp:
                    data = resp.read()
                b[: len(data)] = data
                self.pos += len(data)
                return len(data)
            except Exception as e:  # noqa: BLE001 -- deliberate retry on transient network errors
                last_err = e
                time.sleep(1.5 * (attempt + 1))
        raise last_err


def resolve_file_id(file_name: str) -> str | None:
    """Look up a slide's current GDC file_id from its filename. The UUID
    embedded in the filename itself is a legacy identifier and will not
    work directly against most GDC endpoints."""
    filt = f'{{"op":"=","content":{{"field":"file_name","value":"{file_name}"}}}}'
    url = ("https://api.gdc.cancer.gov/files?filters=" + urllib.parse.quote(filt)
           + "&fields=file_id&format=json")
    with urllib.request.urlopen(url, timeout=30) as resp:
        payload = json.loads(resp.read())
    hits = payload["data"]["hits"]
    return hits[0]["file_id"] if hits else None


def get_thumbnail_array(file_id: str):
    """Return (array, width, height) for the slide's embedded Aperio
    thumbnail page, read remotely via HTTP range requests. The thumbnail is
    the smallest non-label/non-macro page in the SVS's page pyramid
    (identified by the absence of "label"/"macro" in its TIFF
    ImageDescription), typically a few hundred pixels on a side."""
    url = f"https://api.gdc.cancer.gov/data/{file_id}"
    f = HTTPRangeFile(url)
    tf = tifffile.TiffFile(f)
    candidates = []
    for page in tf.pages:
        desc = str(getattr(page, "description", "") or "")
        h, w = page.shape[0], page.shape[1]
        is_associated = "label" in desc.lower() or "macro" in desc.lower()
        candidates.append((w * h, w, h, is_associated, page))
    plain = [c for c in candidates if not c[3]]
    plain.sort(key=lambda c: c[0])
    thumb_candidates = [c for c in plain if c[1] < 2000 and c[2] < 2000]
    chosen = thumb_candidates[-1] if thumb_candidates else plain[0]
    arr = chosen[4].asarray()
    return arr, chosen[1], chosen[2]
