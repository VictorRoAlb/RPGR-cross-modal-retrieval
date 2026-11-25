"""
Thumbnails of TCGA WSIs from the public GDC API without downloading the slides.

Aperio SVS files store their thumbnail as an ordinary TIFF page; HTTP Range requests
(api.gdc.cancer.gov supports HTTP 206) let tifffile read only that page. The identity of
each slide is checked against the GDC metadata (file name and case submitter id).

Run with: python -m figures.fetch_thumbnails --config <yaml>
Reads <output_dir>/qualitative_example/rankings.csv and writes <short id>.png (e.g. 46-3766.png)
for the top-5 WSIs of both lists into tcga.thumbnail_dir.
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


def main() -> None:
    import hashlib
    from pathlib import Path

    import numpy as np
    import pandas as pd
    from PIL import Image

    from data.config import config_argument_parser, load_config, output_dir

    cfg = load_config(config_argument_parser(__doc__).parse_args().config)
    out = Path(cfg["tcga"]["thumbnail_dir"])
    out.mkdir(parents=True, exist_ok=True)
    rank = pd.read_csv(output_dir(cfg, "qualitative_example") / "rankings.csv")
    rows = []
    for slide in sorted(rank[rank["rank"] <= 5].slide.unique()):
        short = slide[:12]
        png = out / f"{short[5:]}.png"
        if png.exists():
            continue
        fname = slide + ".svs"
        fid = resolve_file_id(fname)
        assert fid, f"not found on GDC: {fname}"
        filt = json.dumps({"op": "=", "content": {"field": "file_id", "value": fid}})
        url = ("https://api.gdc.cancer.gov/files?filters=" + urllib.parse.quote(filt)
               + "&fields=file_name,cases.submitter_id&format=json")
        meta = json.loads(urllib.request.urlopen(url, timeout=30).read())["data"]["hits"][0]
        assert meta["file_name"] == fname and meta["cases"][0]["submitter_id"] == short
        arr, w, h = get_thumbnail_array(fid)
        Image.fromarray(np.asarray(arr)[..., :3]).save(png)
        rows.append(dict(slide_id=slide, gdc_file_id=fid, thumbnail_px=f"{w}x{h}", png=png.name,
                         png_sha256=hashlib.sha256(png.read_bytes()).hexdigest()))
        print(short, fid, f"{w}x{h}", flush=True)
    pd.DataFrame(rows).to_csv(out / "thumbnails_provenance.csv", index=False)


if __name__ == "__main__":
    main()
