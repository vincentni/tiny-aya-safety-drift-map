"""The MBBQ files the question pairs in four languages are drawn from, at one commit of the public repository.

MBBQ (Neplenbroek, Bisazza and Fernández, 2024) is licensed CC BY 4.0, and its authors ask that it be used to
evaluate models and not to train them. No file of it is kept in this repository: the files are downloaded into a
folder the caller names, in the layout of the MBBQ repository, and every file is checked against the sha256
recorded here before it is read.
"""

from collections.abc import Callable
from pathlib import Path

from driftmap.attribution import bbq, pinned
from driftmap.attribution.pinned import download

REPOSITORY = "Veranep/MBBQ"
COMMIT = "7046cb17bc86292b347a282275d94b78585f3084"
LICENCE = "CC BY 4.0"
README = "README.md"
# the sentence of the README that says what the dataset is for
NOTE = "MBBQ is intended for model evaluation and should *NOT* be used for model training."
CATEGORIES = ("Gender_identity", "Age", "Disability_status", "Physical_appearance", "SES", "Sexual_orientation")
LANGUAGES = ("en", "es", "nl", "tr")
FILES = {
    "LICENSE": "9e5f1b3c610b9c2da5c313bf81d577a7d1acec686bdb0384edefa6df0f90cd94",
    README: "149f96495811f5d921b18299504cf7939c1c5ad31a74d756ed5562a161c2cde0",
    "data/Age_en.jsonl": "15d34d6ba4dc7e8f2fc2362ef9a3a37ce148e04b2edd3e3700eac49e4c10871c",
    "data/Age_es.jsonl": "44995575aa60e19e6db1bc9f262b2aaba07f563d7394eecab6475ad478a191cd",
    "data/Age_nl.jsonl": "bf4e0a636be38c0c0dbbac638baaabcc1e25206e7c785ef9b0077ccb2bfbbfa1",
    "data/Age_tr.jsonl": "98ef9e3b899cee7faca8ad4f56e7976a757738babaedc36063a0c15da4c27bf8",
    "data/Disability_status_en.jsonl": "003424ffe5ebf26302c2cae5af9afd37acd1ff55a86b36412e6bc0b59e1fdc88",
    "data/Disability_status_es.jsonl": "157202c79093cb182957ef316f477fb20d03a9ef6774078b5c35a1b6aac8db3b",
    "data/Disability_status_nl.jsonl": "d0f7bc310cce987d5fff853a922061c0a60d54a5ba6a1d627ca11e21b4112ac5",
    "data/Disability_status_tr.jsonl": "275c68a207531225a5ee62cc7bf644b565c392135f352517c35381aa7913bd16",
    "data/Gender_identity_en.jsonl": "b6e9be11cb0d6c85bb305867e1b74338ff0ec2dfebc02760778a4364130a3d92",
    "data/Gender_identity_es.jsonl": "ba872fd284130a3deb6bdc62ad77002b89c67a4622e3d7cc268f233e0df555f6",
    "data/Gender_identity_nl.jsonl": "d2f5a41d1f6d1fdbb05c4ac4fa887f62bd0c135cc3f2e3756bc7531e7e87903e",
    "data/Gender_identity_tr.jsonl": "1e2564322b1f4bce1f3fbddbf766e9c7871465a5eeac27d8911c47d7d0dffadd",
    "data/Physical_appearance_en.jsonl": "cc6fd2e318e7a7058199537bcf3bdb5ee5629ac8480257d72b8130362f374974",
    "data/Physical_appearance_es.jsonl": "f02c9515d888690b67ad95551d0dfbf1591d59064c5defac8c1ce066bad48304",
    "data/Physical_appearance_nl.jsonl": "a78b3108bb5754cca1208134f5bc57cba1997bf3e95aeb2f962afc26f5cb6240",
    "data/Physical_appearance_tr.jsonl": "f08f77e6f6a282415d14320ab09554dcf8318d8ab83e9f96e8796247c2b0d685",
    "data/SES_en.jsonl": "6603a4d2ac5293b2718b13d439361a9b8a16e6be8f600b150b8ab9c0d432efb9",
    "data/SES_es.jsonl": "7108b3c6809e1557aa366060961d02f30dc6112f2d35863faf745f92fb9ef089",
    "data/SES_nl.jsonl": "19969eb8bbd8f783d40ee42db2bccfda9bf3be132a325b60ef37affdb2ee51a7",
    "data/SES_tr.jsonl": "7db84dfac8ce1031ac7812c51b88b66f79c00b8b0d2646e0ce47035775aa214f",
    "data/Sexual_orientation_en.jsonl": "d61426a4634a8eecc44ca258c3d70b9730dadcefc625bd5890f1207269d7b660",
    "data/Sexual_orientation_es.jsonl": "8e75c1e1931fc288b91c9418b35267429f8bbdd2848e04feeaa531fecccf14d9",
    "data/Sexual_orientation_nl.jsonl": "894e58600a28bb0063c775517daff5054a827ef3a6c4fdd17605a2703edb6e49",
    "data/Sexual_orientation_tr.jsonl": "e4f17323bd73fbba9df7f3957697d86bdfd2774ed579b838772f7a486612001c",
}


def data_file(category: str, language: str) -> str:
    """Path, in the repository, of the rows of one category in one language."""
    return f"data/{category}_{language}.jsonl"


def fetch(directory: Path, get: Callable[[str], bytes] = download) -> dict[str, str]:
    """The sha256 of every MBBQ file under ``directory``, by its path there; a missing file is downloaded first.

    A file that does not have the recorded sha256 stops the run; a downloaded one is then not written. So does a
    README without the sentence on what the dataset is for.
    """

    def url(name: str) -> str:
        return bbq.URL.format(repository=REPOSITORY, commit=COMMIT, name=name)

    found = pinned.fetch(directory, FILES, url, f"MBBQ at {COMMIT[:12]}", get)
    if NOTE not in (directory / README).read_text(encoding="utf-8"):
        raise SystemExit(f"{README} does not hold the sentence on what MBBQ is for")
    return found
