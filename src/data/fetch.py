"""Fetch SimJEB from Harvard Dataverse and read files straight out of the archives.

The dataset is distributed as five zip archives totalling ~7 GB, which expand to well
over 20 GB. Kaggle's working directory holds about 20 GB, so **the archives are never
fully extracted**: files are read one member at a time, used, and discarded.

That also makes the whole pipeline reproducible from nothing -- a Kaggle account and
an internet connection are the only prerequisites, with no local data to upload.
"""

from __future__ import annotations

import io
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

DATAVERSE = "https://dataverse.harvard.edu/api/access/datafile"

# File ids within the SimJEB record (doi:10.7910/DVN/XFUWJG), with their published
# sizes so a truncated download is caught rather than producing a corrupt archive.
FILES = {
    "metadata":      (4639239, 155_602,       "all_bracket_metadata.tab"),
    "fem_first":     (4640545, 1_051_947_428, "SimJEB_fea_fem_firsthalf.zip"),
    "fem_second":    (4640683, 1_024_622_207, "SimJEB_fea_fem_secondhalf.zip"),
    "vtk":           (4640734, 1_741_938_661, "SimJEB_volmesh_vtk.zip"),
    "csv_first":     (4640686, 1_677_460_769, "SimJEB_simresults_csv_firsthalf.zip"),
    "csv_second":    (4640716, 1_634_994_033, "SimJEB_simresults_csv_secondhalf.zip"),
}


@dataclass
class SimJEBSource:
    """Locates every file for a model, wherever the archives happen to live.

    Handles the split archives transparently: the ``.fem`` decks and result CSVs are
    each spread across two zips with no documented rule for which model is in which,
    so the index is built by looking.
    """

    root: Path
    _members: dict[str, tuple[Path, str]]

    @classmethod
    def open(cls, root: str | Path) -> "SimJEBSource":
        root = Path(root)
        members: dict[str, tuple[Path, str]] = {}
        for path in sorted(root.glob("*.zip")):
            with zipfile.ZipFile(path) as archive:
                for name in archive.namelist():
                    if name.endswith((".fem", ".vtk", ".csv")):
                        members[Path(name).name] = (path, name)
        if not members:
            raise FileNotFoundError(f"no SimJEB archives found in {root}")
        return cls(root=root, _members=members)

    def model_ids(self) -> list[int]:
        """Models with all three required files present."""
        ids = []
        for name in self._members:
            if name.endswith(".vtk"):
                model_id = int(Path(name).stem)
                if f"{model_id}.fem" in self._members and \
                   f"{model_id}field.csv" in self._members:
                    ids.append(model_id)
        return sorted(ids)

    def read(self, filename: str) -> bytes:
        if filename not in self._members:
            raise KeyError(f"{filename} not found in any archive under {self.root}")
        archive_path, member = self._members[filename]
        with zipfile.ZipFile(archive_path) as archive:
            return archive.read(member)

    def extract_model(self, model_id: int, dest: str | Path) -> dict[str, Path]:
        """Write one model's three files to ``dest`` and return their paths.

        The parsers take file paths rather than bytes, so a model is materialised
        briefly and then removed by the caller. Peak disk stays at one model -- a few
        hundred MB -- instead of the 20+ GB a full extraction would need.
        """
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        paths = {}
        for key, filename in (
            ("vtk", f"{model_id}.vtk"),
            ("fem", f"{model_id}.fem"),
            ("csv", f"{model_id}field.csv"),
        ):
            out = dest / filename
            out.write_bytes(self.read(filename))
            paths[key] = out
        return paths


def download(key: str, dest: str | Path, chunk_size: int = 1 << 20,
             verify_size: bool = True) -> Path:
    """Download one Dataverse file, resuming if a partial copy exists.

    Dataverse resets long connections -- four drops were observed pulling the 3.3 GB of
    result CSVs -- so this uses a Range request to continue from whatever is already on
    disk rather than starting over.
    """
    file_id, expected_size, filename = FILES[key]
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / filename

    have = path.stat().st_size if path.exists() else 0
    if verify_size and have == expected_size:
        return path
    if have > expected_size:
        path.unlink()
        have = 0

    request = urllib.request.Request(f"{DATAVERSE}/{file_id}")
    if have:
        request.add_header("Range", f"bytes={have}-")

    with urllib.request.urlopen(request) as response, open(path, "ab") as out:
        shutil.copyfileobj(response, out, chunk_size)

    final = path.stat().st_size
    if verify_size and final != expected_size:
        raise IOError(
            f"{filename}: got {final} bytes, expected {expected_size}. "
            f"Re-run to resume from where it stopped."
        )
    return path


def download_all(dest: str | Path, keys: list[str] | None = None) -> dict[str, Path]:
    """Fetch everything needed to build the dataset. Safe to re-run."""
    keys = keys or list(FILES)
    paths = {}
    for key in keys:
        print(f"fetching {FILES[key][2]} ...", flush=True)
        paths[key] = download(key, dest)
    return paths
