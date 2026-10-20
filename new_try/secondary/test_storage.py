import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import URLError

import resumable_adapter as runner
import transcript_length as length
from job_storage import open_storage
from pipeline import Analysis, SYSTEM_PROMPT


def reply():
    return json.dumps(dict(participants=[dict(speaker='A',role=None)],
        summary='Mock section summary.', high_points=[],timeline=[],
        unresolved_questions=[], model_sources=[],legal_review=dict(
        status='insufficient_information',explanation='Mock only; no legal sources.',
        concerns=[],limitations=[])))


class StorageTests(unittest.TestCase):
    def test_memory_does_not_create_directory(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'unused'
            store=open_storage('memory',path,'job')
            store.save('chunk','{}')
            store.status({'status':'running'})
            self.assertEqual(store.load('chunk'),'{}')
            self.assertFalse(path.exists())

    def test_disk_resume_after_failure(self):
        text=''.join(f'[{i:02}:00] A: Marker{i} '+ 'x'*400+'\n' for i in range(40))
        request=dict(system_prompt=SYSTEM_PROMPT,json_schema=Analysis.model_json_schema(),
            user_prompt=json.dumps(dict(transcript=text,source_mode='model',scope='Title 18',
                coverage_complete=False,legal_sources=[])))
        with tempfile.TemporaryDirectory() as root, \
             patch.object(runner,'STORAGE_MODE','disk'), \
             patch.object(runner,'CHECKPOINT_ROOT',root), \
             patch.object(length,'MAX_REQUEST_BYTES',15000), \
             contextlib.redirect_stderr(io.StringIO()):
            calls=[]
            def interrupted(**kw):
                calls.append(kw['user_prompt'])
                if len(calls)==2:
                    raise RuntimeError('Simulated process failure')
                return reply()
            with patch.object(runner.model_adapter,'base_model',side_effect=interrupted):
                with self.assertRaises(RuntimeError):runner.base_model(**request)
            retried=[]
            def resumed(**kw):
                retried.append(kw['user_prompt'])
                return reply()
            with patch.object(runner.model_adapter,'base_model',side_effect=resumed):
                result=json.loads(runner.base_model(**request))
            self.assertNotIn(calls[0],retried)
            self.assertEqual(retried[0],calls[1])
            self.assertEqual(result['legal_review']['status'],'insufficient_information')
            # A third run uses only checkpoints, including the formerly failed chunk.
            with patch.object(runner.model_adapter,'base_model',side_effect=AssertionError('Unexpected call')):
                runner.base_model(**request)
            status=json.loads(next(Path(root).glob('*/status.json')).read_text())
            self.assertEqual(status['completed_chunks'],status['total_chunks'])
            self.assertEqual(status['status'],'analysis_completed')
            self.assertGreater(status['checkpoint_bytes'],0)

    def test_transient_retry_in_memory(self):
        request=dict(system_prompt=SYSTEM_PROMPT,json_schema=Analysis.model_json_schema(),
            user_prompt=json.dumps(dict(transcript='[00:00] A: Mock.',source_mode='model',scope='Title 18',
                coverage_complete=False,legal_sources=[])))
        with patch.object(runner,'STORAGE_MODE','memory'), \
             patch.object(runner.model_adapter,'base_model',side_effect=[URLError('offline'),reply()]) as call, \
             patch.object(runner.time,'sleep'), contextlib.redirect_stderr(io.StringIO()):
            runner.base_model(**request)
            self.assertEqual(call.call_count,2)


if __name__=='__main__':unittest.main()
