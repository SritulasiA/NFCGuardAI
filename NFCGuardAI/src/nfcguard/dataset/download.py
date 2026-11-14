

from __future__ import annotations
from pathlib import Path
from urllib.request import Request, urlopen
import shutil

DATA_URL = 'https://media.githubusercontent.com/media/Yifeng-Wang/nfc-relayattack/main/NFC_Relay.zip'
DATA_PAGE = 'https://github.com/Yifeng-Wang/nfc-relayattack'


def download_dataset(destination: str | Path, url=DATA_URL, overwrite=False):
    dest = Path(destination)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not overwrite:
        raise FileExistsError(f'{dest} exists; supply --overwrite to replace it.')
    tmp = dest.with_name(dest.name + '.part')
    try:
        req = Request(url, headers={'User-Agent': 'NFCGuardAI Research Software/1.0'})
        with urlopen(req, timeout=120) as response, tmp.open('wb') as stream:
            shutil.copyfileobj(response, stream, length=1024 * 1024)
        # Git LFS pointer (not actual archive) must be rejected.
        import zipfile
        if not zipfile.is_zipfile(tmp):
            raise ValueError('Downloaded bytes are not a ZIP. Git LFS may have returned a pointer. Use git lfs clone, then pass --source.')
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)
    return dest
