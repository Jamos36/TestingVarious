import copy
import json
import unittest
from unittest.mock import patch

import transcript_length as length
from pipeline import Analysis, SYSTEM_PROMPT
from test_pipeline import FIXTURE


def payload(text):
    return dict(transcript=text, source_mode='model', scope='U.S. Code Title 18',
                coverage_complete=False, legal_sources=[])


class LengthTests(unittest.TestCase):
    def test_turn_coverage_and_overlap(self):
        text = ''.join(f'[{i:02}:00] A: Turn {i}.\n' for i in range(12))
        chunks = length.split_transcript(text, lambda s: len(s) <= 100)
        self.assertGreater(len(chunks), 1)
        for line in text.splitlines(keepends=True):
            self.assertTrue(any(line in c for c in chunks))
        self.assertTrue(all(len(c) <= 100 for c in chunks))

    def test_oversized_turn_rejected(self):
        with self.assertRaises(ValueError):
            length.split_transcript('[00:00] A: '+ 'x'*200, lambda s: len(s) <= 100)

    def test_short_passes_through(self):
        with patch.object(length, 'call_model', return_value='unchanged') as model:
            self.assertEqual(length.base_model(system_prompt=SYSTEM_PROMPT,
                user_prompt=json.dumps(payload('short')), json_schema=Analysis.model_json_schema()), 'unchanged')
            self.assertEqual(model.call_count, 1)

    def test_chunked_calls_and_citation_namespacing(self):
        text = ''.join(f'[{i:02}:00] A: Marker{i} '+ 'x'*400+'\n' for i in range(40))
        seen = []
        def fake(**kwargs):
            self.assertLessEqual(len(length.encode(kwargs).encode()), 15000)
            p = json.loads(kwargs['user_prompt'])
            seen.append(p['transcript'])
            f = copy.deepcopy(FIXTURE)
            f.update(participants=[dict(speaker='A',role=None)], summary='Section summary',
                     high_points=[], timeline=[], unresolved_questions=[])
            f['model_sources'] = [dict(source_id='same-id',citation='TEST ONLY',
                origin='model_memory',text=None,source_url=None,version=None)]
            f['legal_review'].update(status='potential_concern', concerns=[dict(
                description='Test only',source_id='same-id', legal_quote=None,
                evidence=[dict(speaker='A',timestamp=None,quote=p['transcript'].splitlines()[0])],
                element_analysis='Test only',missing_facts=[],counterevidence=[])])
            return json.dumps(f)
        with patch.object(length,'MAX_REQUEST_BYTES',15000), patch.object(length,'call_model',side_effect=fake):
            result=json.loads(length.base_model(system_prompt=SYSTEM_PROMPT,
                user_prompt=json.dumps(payload(text)),json_schema=Analysis.model_json_schema()))
        self.assertGreater(len(seen),1)
        self.assertEqual(len(result['legal_review']['concerns']),len(seen))
        ids=[s['source_id'] for s in result['model_sources']]
        self.assertEqual(len(ids),len(set(ids)))
        for line in text.splitlines():
            self.assertTrue(any(line in c for c in seen))


if __name__ == '__main__':
    unittest.main()
