import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import URLError

from config import Settings, load_settings
from output import deliver
from service import run
from test_pipeline import FIXTURE, ROOT


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.text=(ROOT/'mock_interview.txt').read_text()

    def fake(self, **kwargs):
        return json.dumps(FIXTURE)

    def test_minimal_no_storage_or_progress_or_markdown(self):
        s=Settings(chunking=False,storage='none',progress=False,render_markdown=False)
        with tempfile.TemporaryDirectory() as root, patch('job_storage.open_storage') as unused:
            report=run(self.text,s,model_call=self.fake)
            output=io.StringIO()
            with contextlib.redirect_stdout(output):deliver(report,s)
            self.assertNotIn('markdown',json.loads(output.getvalue()))
            unused.assert_not_called()

    def test_file_output_and_optional_markdown(self):
        with tempfile.TemporaryDirectory() as root:
            s=Settings(output='files',output_dir=root,progress=False)
            report=run(self.text,s,model_call=self.fake)
            deliver(report,s)
            self.assertTrue((Path(root)/'summary.json').exists())
            self.assertIn('Interview summary',(Path(root)/'summary.md').read_text())

    def test_settings_and_configs(self):
        for file in (ROOT/'configs').glob('*.json'):load_settings(file)
        for values in ({'unknown':True},{'chunking':'false'},{'retry_attempts':0},
                       {'adapter':'transcript_length:base_model'}):
            with self.subTest(values=values),self.assertRaises(ValueError):Settings(**values)

    def test_oversize_rejected_without_call(self):
        s=Settings(chunking=False,max_request_bytes=100,progress=False)
        with self.assertRaises(ValueError):run(self.text,s,model_call=lambda **kw:self.fail('called'))

    def test_retry_toggle(self):
        count=[]
        def flaky(**kw):
            count.append(1)
            if len(count)==1:raise URLError('offline')
            return self.fake(**kw)
        with patch('retry_policy.time.sleep'):
            run(self.text,Settings(retry_attempts=2,progress=False),model_call=flaky)
        self.assertEqual(len(count),2)

    def test_sources_independent_of_storage(self):
        f=copy.deepcopy(FIXTURE)
        f['legal_review'].update(status='none_identified',limitations=[],
            explanation='No potential violation identified in this screening against the reviewed references.')
        source=dict(source_id='s',citation='TEST ONLY',source_url=None,version=None,text='Dummy text')
        for storage in ('none','memory','disk'):
            with tempfile.TemporaryDirectory() as root:
                settings=Settings(source_mode='supplied',coverage_complete=True,
                    storage=storage,checkpoint_dir=root,progress=False)
                report=run(self.text,settings,sources=[source],model_call=lambda **kw:json.dumps(f))
                self.assertEqual(report['analysis']['legal_review']['status'],'none_identified')

    def test_checkpoint_reuse_and_revision_invalidation(self):
        calls=[]
        def model(**kw):calls.append(1);return self.fake(**kw)
        with tempfile.TemporaryDirectory() as root:
            s=Settings(storage='disk',checkpoint_dir=root,progress=False)
            first=run(self.text,s,model_call=model)
            second=run(self.text,s,model_call=model)
            self.assertEqual(len(calls),1)
            self.assertEqual(first['job_id'],second['job_id'])
            changed=run(self.text,s.model_copy(update={'job_revision':'2'}),model_call=model)
            self.assertEqual(len(calls),2)
            self.assertNotEqual(first['job_id'],changed['job_id'])

    def test_prompt_and_progress_injection(self):
        seen=[];events=[]
        def model(**kw):seen.append(kw['system_prompt']);return self.fake(**kw)
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'prompt.txt';path.write_text('Custom policy')
            report=run(self.text,Settings(prompt_path=str(path)),model_call=model,progress_sink=events.append)
            self.assertEqual(seen,['Custom policy'])
            self.assertEqual(events[-1]['status'],'analysis_completed')
            self.assertEqual(events[-1]['job_id'],report['job_id'])

class OptionalModuleTests(unittest.TestCase):
    def test_minimal_cli_without_optional_modules(self):
        import os
        import shutil
        import subprocess
        import sys
        with tempfile.TemporaryDirectory() as root:
            target=Path(root)
            omitted={'transcript_length.py','resumable_adapter.py','job_storage.py',
                     'retry_policy.py','renderers.py'}
            for file in ROOT.iterdir():
                if file.is_file() and file.name not in omitted and not file.name.startswith('test_'):
                    shutil.copy(file,target/file.name)
            (target/'fake_adapter.py').write_text(
                'def base_model(**kwargs):\n    return '+repr(json.dumps(FIXTURE))+'\n')
            (target/'minimal.json').write_text(json.dumps(dict(adapter='fake_adapter:base_model',
                chunking=False, storage='none',retry_attempts=1,progress=False,render_markdown=False)))
            result=subprocess.run([sys.executable,'pipeline.py','mock_interview.txt',
                '--config','minimal.json'],cwd=target,capture_output=True,text=True,check=True)
            report=json.loads(result.stdout)
            self.assertIn('report',report)
            self.assertNotIn('markdown',report)
            self.assertEqual(result.stderr,'')


if __name__=='__main__':unittest.main()
