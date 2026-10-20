import io
import json
import unittest
from unittest.mock import patch

import model_adapter as adapter


class AdapterTests(unittest.TestCase):
    def invoke(self, result):
        response = io.BytesIO(json.dumps(result).encode())
        with patch.object(adapter, 'urlopen', return_value=response) as call:
            text = adapter.base_model(system_prompt='Instructions',
                user_prompt='{"source_mode":"model","transcript":"Mock"}',
                json_schema={'type': 'object'})
            payload = json.loads(call.call_args.args[0].data)
            return text, payload

    def test_prompts_schema_and_json_text(self):
        text, payload = self.invoke({'choices':[{'finish_reason':'stop',
            'message':{'content':'{"summary":"Mock"}'}}]})
        self.assertEqual(json.loads(text)['summary'], 'Mock')
        self.assertEqual(payload['messages'][0]['content'], 'Instructions')
        self.assertEqual(payload['response_format']['json_schema']['schema'], {'type':'object'})

    def test_schema_prompt_fallback(self):
        with patch.object(adapter, 'USE_STRUCTURED_OUTPUT', False):
            _, payload = self.invoke({'choices':[{'finish_reason':'stop',
                'message':{'content':'{}'}}]})
        self.assertNotIn('response_format', payload)
        self.assertIn('Required JSON Schema', payload['messages'][0]['content'])

    def test_reject_truncation_and_refusal(self):
        for finish, message in [('length', {'content':'{'}),
                                ('stop', {'refusal':'Cannot comply'})]:
            with self.subTest(finish=finish), self.assertRaises(RuntimeError):
                self.invoke({'choices':[{'finish_reason':finish, 'message':message}]})


if __name__ == '__main__':
    unittest.main()
