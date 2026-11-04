import hashlib
import json
from pathlib import Path, PurePosixPath
from . import OFFICIAL_COMMIT

def verify_bundle(source: str | Path) -> Path:
    source = Path(source).expanduser().resolve()
    try:
        manifest = json.loads((source / 'manifest.json').read_text(encoding='utf-8'))
        if not isinstance(manifest, dict) or manifest.get('revision') != OFFICIAL_COMMIT:
            raise ValueError('unexpected upstream revision')
        files = manifest.get('files')
        if not isinstance(files, dict) or not files:
            raise ValueError('empty source manifest')
        for relative, expected in files.items():
            if not isinstance(relative, str) or '\\' in relative:
                raise ValueError('invalid source path')
            name = PurePosixPath(relative)
            if name.is_absolute() or '..' in name.parts or (not name.parts):
                raise ValueError('source path escapes the bundle')
            path = source.joinpath(*name.parts)
            if not isinstance(expected, str) or len(expected) != 64 or any((part.is_symlink() for part in (path, *path.parents) if part != source)) or (not path.resolve().is_relative_to(source)):
                raise ValueError('invalid source manifest entry')
            with path.open('rb') as handle:
                actual = hashlib.file_digest(handle, 'sha256').hexdigest()
            if actual != expected:
                raise ValueError(f'source checksum differs: {relative}')
    except (OSError, ValueError, TypeError) as exc:
        raise RuntimeError(f'bundled official helpers are missing or modified: {exc}') from exc
    return source
