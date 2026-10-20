"""Output delivery can be replaced without changing analysis or model calls."""
import json
from pathlib import Path


def deliver(report, settings):
    rendered = None
    if settings.render_markdown:
        from renderers import markdown
        rendered = markdown(report)
    if settings.output == 'stdout':
        envelope = {'report': report}
        if rendered is not None:
            envelope['markdown'] = rendered
        print(json.dumps(envelope, ensure_ascii=False))
    else:
        # Each file is atomic; the pair is not a database transaction.
        from job_storage import DiskStorage
        writer = DiskStorage(settings.output_dir)
        writer.save('summary', json.dumps(report, indent=2, ensure_ascii=False))
        if rendered is not None:
            path = Path(settings.output_dir) / 'summary.md'
            # Use the same atomic writer, then publish with the intended suffix.
            writer.save('markdown-temp', rendered)
            (Path(settings.output_dir) / 'markdown-temp.json').replace(path)
