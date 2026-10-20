"""Optional checkpoints using pathlib for filesystem operations."""
import json
from pathlib import Path
from uuid import uuid4


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
        path = self.directory / f"{key}.json"
        try:
            return path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None

    def save(self, key, value):
        destination = self.directory / f"{key}.json"
        temporary = self.directory / f".pending-{uuid4().hex}"

        try:
            # Exclusive creation avoids overwriting another temporary file.
            with temporary.open("x", encoding="utf-8") as output:
                output.write(value)

            # Same-directory replacement publishes the completed file.
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)

    def status(self, value):
        self.save(
            "status",
            json.dumps(value, ensure_ascii=False),
        )


def open_storage(mode, root, job_id):
    if mode == "memory":
        return MemoryStorage()

    if mode == "disk":
        return DiskStorage(Path(root) / job_id)

    raise ValueError("Storage mode must be memory or disk")
