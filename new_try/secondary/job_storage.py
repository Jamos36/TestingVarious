"""Optional checkpoint storage. Only disk mode writes files."""
import json
import os
from pathlib import Path
import tempfile


class MemoryStorage:
    def __init__(self):
        self.results = {}

    def load(self, key):
        return self.results.get(key)

    def save(self, key, value):
        self.results[key] = value

    def status(self, value):
        pass


class DiskStorage:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def load(self, key):
        path = self.directory / f'{key}.json'
        return path.read_text(encoding='utf-8') if path.exists() else None

    def save(self, key, value):
        # Publish a complete file or leave the previous version intact.
        fd, name = tempfile.mkstemp(dir=self.directory, prefix='.pending-')
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as output:
                output.write(value)
                output.flush()
                os.fsync(output.fileno())
            os.replace(name, self.directory / f'{key}.json')
        finally:
            if os.path.exists(name):
                os.unlink(name)

    def status(self, value):
        self.save('status', json.dumps(value, ensure_ascii=False))


def open_storage(mode, root, job_id):
    if mode == 'memory':
        return MemoryStorage()
    if mode == 'disk':
        return DiskStorage(Path(root) / job_id)
    raise ValueError('TRANSCRIPT_STORAGE must be memory or disk')
