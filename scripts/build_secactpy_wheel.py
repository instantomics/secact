"""Build the SecActpy wheel published as this repository's release asset.

SecActpy 0.3.1 caps numpy below 2, which has no Python 3.13 wheels. This
rewrites only the wheel's metadata to drop that cap and marks the change with a
local version; the package code and signature matrices are upstream's bytes.
"""

import base64
import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = (
    "https://files.pythonhosted.org/packages/py3/s/secactpy/"
    "secactpy-0.3.1-py3-none-any.whl"
)
SHA256 = "8b8b59294178285b3d1ed5f3cd288243eed3d1e4d630ad4b35a201484dea553b"
OLD, NEW = "secactpy-0.3.1.dist-info/", "secactpy-0.3.1+numpy2.dist-info/"
CAP = "Requires-Dist: numpy<2.0.0,>=1.20.0\n"


def record_hash(data):
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
    return f"sha256={digest.decode()}"


def main(output):
    with urllib.request.urlopen(URL, timeout=300) as response:
        wheel = response.read()
    if hashlib.sha256(wheel).hexdigest() != SHA256:
        raise SystemExit("upstream wheel digest mismatch")
    source = Path(output) / "upstream.whl"
    source.write_bytes(wheel)
    target = Path(output) / "secactpy-0.3.1+numpy2-py3-none-any.whl"
    records = []
    with (
        zipfile.ZipFile(source) as old,
        zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as new,
    ):
        for info in old.infolist():
            name = info.filename.replace(OLD, NEW)
            if name == NEW + "RECORD":
                continue
            data = old.read(info)
            if name == NEW + "METADATA":
                text = data.decode()
                if CAP not in text:
                    raise SystemExit("numpy cap not found in upstream metadata")
                text = text.replace(CAP, "Requires-Dist: numpy>=1.20.0\n")
                text = text.replace("Version: 0.3.1\n", "Version: 0.3.1+numpy2\n", 1)
                data = text.encode()
            new.writestr(
                zipfile.ZipInfo(name, info.date_time), data, zipfile.ZIP_DEFLATED
            )
            records.append(f"{name},{record_hash(data)},{len(data)}")
        records.append(f"{NEW}RECORD,,")
        new.writestr(NEW + "RECORD", "\n".join(records) + "\n")
    source.unlink()
    print(hashlib.sha256(target.read_bytes()).hexdigest(), target)


if __name__ == "__main__":
    main(sys.argv[1])
