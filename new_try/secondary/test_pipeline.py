import copy
import json
from pathlib import Path
import unittest

from pipeline import summarize, markdown

ROOT = Path(__file__).parent
FIXTURE = {
    "model_sources": [],
    "participants": [{"speaker": "Interviewer", "role": "Interviewer"},
                     {"speaker": "Witness A", "role": None},
                     {"speaker": "Witness B", "role": None}],
    "summary": "Witness A reported two boxes missing from a shelf. Witness B said they moved two boxes to an overflow room and emailed the team. Witness A had not seen the email and did not allege theft. The overflow log and inventory list remain to be reconciled.",
    "high_points": [{"summary": "Witness A explicitly qualified the initial report.", "evidence": [
        {"speaker": "Witness A", "timestamp": "00:00:25", "quote": "I cannot say anything was stolen."}]}],
    "timeline": [{"summary": "Witness A counted boxes on Tuesday and noticed two missing from the shelf.", "evidence": [
        {"speaker": "Witness A", "timestamp": "00:00:08", "quote": "I counted the boxes on Tuesday. Two were missing from the shelf."}]},
        {"summary": "Witness B explained the move and proposed reconciliation.", "evidence": [
        {"speaker": "Witness B", "timestamp": "00:00:17", "quote": "I moved two boxes to the overflow room. I emailed the team."},
        {"speaker": "Witness B", "timestamp": "00:00:39", "quote": "We should reconcile the overflow log with the inventory list."}]}],
    "unresolved_questions": ["Do the overflow log and inventory list reconcile?"],
    "legal_review": {"status": "insufficient_information",
                     "explanation": "No Title 18 source excerpts were provided, so a legal screening conclusion cannot be reached.",
                     "concerns": [], "limitations": ["Legal library retrieval is not connected."]}
}


def run_fixture(fixture=None, sources=None, complete=False):
    reply = FIXTURE if fixture is None else fixture
    def fake_model(**kwargs):
        assert "json_schema" in kwargs
        return json.dumps(reply)
    return summarize((ROOT / 'mock_interview.txt').read_text(), fake_model,
                     sources=sources or [], coverage_complete=complete,
                     scope="U.S. Code Title 18")


class PipelineTests(unittest.TestCase):
    def test_summary_and_render(self):
        a = run_fixture()
        md = markdown({"input_file": "mock_interview.txt", "analysis": a, "legal_sources": []})
        self.assertIn("## Legal screening", md)
        self.assertIn("00:00:25", md)
        self.assertEqual(len(a['participants']), 3)

    def test_reject_invented_quote(self):
        f = copy.deepcopy(FIXTURE)
        f['high_points'][0]['evidence'][0]['quote'] = 'Fabricated confession'
        with self.assertRaises(ValueError):
            run_fixture(f)

    def test_reject_clean_without_sources_or_coverage(self):
        f = copy.deepcopy(FIXTURE)
        f['legal_review']['status'] = 'none_identified'
        with self.assertRaises(ValueError):
            run_fixture(f, complete=True)
        s = [dict(source_id='test', citation='TEST ONLY', source_url=None, version=None, text='Dummy source for validation testing.')]
        with self.assertRaises(ValueError):
            run_fixture(f, sources=s)
        self.assertEqual(run_fixture(f, sources=s, complete=True)['legal_review']['status'], 'none_identified')

    def test_reject_wrong_speaker_or_timestamp(self):
        for key, value in [('speaker', 'Witness B'), ('timestamp', '00:00:08')]:
            f = copy.deepcopy(FIXTURE)
            f['high_points'][0]['evidence'][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                run_fixture(f)

    def test_reject_unsupported_legal_citation(self):
        f = copy.deepcopy(FIXTURE)
        f['legal_review']['status'] = 'potential_concern'
        f['legal_review']['concerns'] = [dict(description='Test only', source_id='absent',
            legal_quote='Invented statute', evidence=f['high_points'][0]['evidence'],
            element_analysis='Test', missing_facts=[], counterevidence=[])]
        with self.assertRaises(ValueError):
            run_fixture(f)

    def test_reject_malformed_response_and_extra_keys(self):
        with self.assertRaises(ValueError):
            summarize('text', lambda **kw: 'not JSON', sources=[], scope='Title 18')
        f = copy.deepcopy(FIXTURE)
        f['unexpected'] = True
        with self.assertRaises(ValueError):
            run_fixture(f)


class ModelModeTests(unittest.TestCase):
    def reply(self, origin="model_memory"):
        f = copy.deepcopy(FIXTURE)
        f['model_sources'] = [dict(source_id='ref1', citation='TEST ONLY reference',
            origin=origin, source_url=None, version=None,
            text='Dummy retrieved text.' if origin == 'model_rag' else None)]
        f['legal_review']['status'] = 'none_identified'
        f['legal_review']['explanation'] = 'No potential violation identified in this mock screening.'
        return f

    def run_model(self, reply):
        def fake_model(**kwargs):
            assert json.loads(kwargs['user_prompt'])['source_mode'] == 'model'
            return json.dumps(reply)
        return summarize((ROOT/'mock_interview.txt').read_text(), fake_model, source_mode='model')

    def test_memory_mode_without_source_file(self):
        a = self.run_model(self.reply())
        self.assertTrue(any('unverified model memory' in x for x in a['legal_review']['limitations']))

    def test_rag_mode_records_sources(self):
        a = self.run_model(self.reply('model_rag'))
        md = markdown(dict(input_file='mock', legal_sources=[], analysis=a))
        self.assertIn('model\\_rag', md)

    def test_cannot_claim_retrieval_from_memory(self):
        f = self.reply()
        f['model_sources'][0]['text'] = 'Pretended quote'
        with self.assertRaises(ValueError):
            self.run_model(f)

    def test_negative_requires_references(self):
        f = self.reply()
        f['model_sources'] = []
        with self.assertRaises(ValueError):
            self.run_model(f)

    def test_memory_concern_has_no_legal_quote(self):
        f = self.reply()
        f['legal_review']['status'] = 'potential_concern'
        c = dict(description='Test fixture only', source_id='ref1', legal_quote=None,
                 evidence=f['high_points'][0]['evidence'], element_analysis='Test only',
                 missing_facts=[], counterevidence=[])
        f['legal_review']['concerns'] = [c]
        self.run_model(f)
        c['legal_quote'] = 'Invented legal quote'
        with self.assertRaises(ValueError):
            self.run_model(f)


if __name__ == '__main__':
    unittest.main()
