import contextlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from ..common import read_jsonl, write_jsonl
from .run import discover, main
from .configure_h200 import make_config, save_config
from .audit_prompts import inspect_prompt


class DownstreamTests(unittest.TestCase):
    def test_prompt_audit_distinguishes_formatting_from_repetition(self):
        flags = inspect_prompt('Instructions: def count(n): return n', '\ndef count(n):\n    return n\n', 'count')
        self.assertFalse(flags['preserved_exact'])
        self.assertTrue(flags['preserved_whitespace_normalized'])
        self.assertTrue(flags['entry_point_present'])
        repeated = inspect_prompt(('Please solve this task carefully and return correct code. ' * 20),
                                  'original problem', 'count')
        self.assertTrue(repeated['repetition_warning'])
        self.assertFalse(repeated['entry_point_present'])

    def test_h200_config_records_identity_and_protects_existing_experiment(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'server-config.json'
            cfg = make_config('model', 'commit', 'sha256:image', 16384, 0.7, 4, 'h200-v1')
            save_config(path, cfg)
            save_config(path, cfg)
            self.assertEqual(json.loads(path.read_text())['server_metadata']['model_revision'], 'commit')
            changed = make_config('model', 'different-commit', 'sha256:image', 16384, 0.7, 4, 'h200-v1')
            with self.assertRaises(ValueError):
                save_config(path, changed)

    def fixture(self, root):
        tasks = root / 'tasks.jsonl'
        write_jsonl(tasks, [{'task_id': 't', 'split': 'validation', 'problem': 'PROBLEM'}])
        inputs = root / 'runs'
        for variant in ('original', 'static', 'base', 'sft'):
            write_jsonl(inputs / 'qwen-best-s42' / f'validation-{variant}-prompts.jsonl', [{
                'task_id': 't', 'split': 'validation', 'variant': variant, 'prompt': 'PROBLEM',
                'target_label': 'target', 'preserves_problem_verbatim': True, 'prompt_chars': 7}])
        config = root / 'config.json'
        config.write_text(json.dumps({'experiment_id': 'test', 'model': 'fake-model',
                                     'server_metadata': {'revision': 'commit123'}}))
        return tasks, inputs, config

    def test_real_http_dedup_and_resume(self):
        calls = []

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                payload = {'data': [{'id': 'fake-model'}]}
                self.respond(payload)

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                calls.append(body)
                self.respond({'choices': [{'message': {'content': '```python\nprint(1)\n```'},
                                           'finish_reason': 'stop'}], 'usage': {'completion_tokens': 4}})

            def respond(self, payload):
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(payload).encode())

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                tasks, inputs, config = self.fixture(root)
                argv = ['run', '--tasks', str(tasks), '--input-root', str(inputs), '--config', str(config),
                        '--output-root', str(root / 'out'), '--base-url', f'http://127.0.0.1:{server.server_port}/v1']
                for _ in range(2):
                    with patch('sys.argv', argv), contextlib.redirect_stdout(io.StringIO()):
                        main()
                self.assertEqual(len(calls), 1)  # Four identical requests, then a complete resume.
                self.assertEqual(calls[0]['temperature'], 0)
                self.assertEqual(calls[0]['seed'], 42)
                outputs = list((root / 'out').rglob('*-code.jsonl'))
                self.assertEqual(len(outputs), 4)
                self.assertEqual(read_jsonl(outputs[0])[0]['code'], 'print(1)')
                self.assertEqual(read_jsonl(outputs[0])[0]['codegen_server_metadata'], {'revision': 'commit123'})
                # A different generator revision cannot silently reuse old responses.
                cfg = json.loads(config.read_text())
                cfg['server_metadata']['revision'] = 'different'
                config.write_text(json.dumps(cfg))
                with patch('sys.argv', argv), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        main()
                self.assertEqual(len(calls), 1)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_coverage_audit_and_invalid_prompt_retention(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tasks, inputs, _ = self.fixture(root)
            path = inputs / 'qwen-best-s42/validation-sft-prompts.jsonl'
            row = read_jsonl(path)[0]
            row.update(prompt='broken prompt', truncated=True)
            write_jsonl(path, [row])
            (inputs / 'qwen-best-s42/validation-base-prompts.jsonl').unlink()
            files, audit = discover(inputs, tasks, 'validation')
            self.assertEqual(len(files), 3)
            self.assertEqual(audit['problem_not_preserved'], 1)
            self.assertEqual(audit['truncated'], 1)
            self.assertEqual(audit['coverage_issues'][0]['reason'], 'missing_file')
            self.assertEqual(files[-1][-1][0]['prompt'], 'broken prompt')


if __name__ == '__main__':
    unittest.main()
