import tempfile
import unittest
import os
from unittest.mock import patch
from pathlib import Path

from .common import configure_gpu, conditioning, load_splits, write_jsonl
from .codegen import extract_code
from .train import encode_example
from .prepare_experiments import prepare, select_pool
from .common import model_load_kwargs
from types import SimpleNamespace


class FakeTokenizer:
    eos_token_id = 999

    def apply_chat_template(self, messages, **kwargs):
        return messages[0]['content'] + '<assistant>'

    def encode(self, text, **kwargs):
        return list(text.encode())


class PipelineTests(unittest.TestCase):
    def test_variants_preserve_splits_and_ignore_test_scores(self):
        from .common import read_jsonl
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'source'
            tasks = [{'task_id': tid, 'split': split, 'problem': 'PROBLEM'}
                     for tid, split in [('a', 'train'), ('b', 'validation'), ('c', 'test')]]
            write_jsonl(source / 'tasks.jsonl', tasks)
            (source / 'splits.json').write_text('{}')
            anchor = {'task_id': 'a', 'split': 'train', 'target_model': 'target',
                      'comparison_group_id': 'g', 'input': 'PROBLEM', 'output': 'PROBLEM carefully'}
            write_jsonl(source / 'sft/train.jsonl', [anchor])
            write_jsonl(source / 'sft/validation.jsonl', [{**anchor, 'task_id': 'b', 'split': 'validation'}])
            write_jsonl(source / 'runs.jsonl', [{'run_id': 'r', 'target_model': 'target',
                        'comparison_group_id': 'g', 'optimizer_method': 'APE'}])
            write_jsonl(source / 'candidates.jsonl', [{'task_id': 'a', 'candidate_id': 'candidate',
                        'run_id': 'r', 'prompt': 'PROBLEM carefully'}])
            write_jsonl(source / 'filtering_report.jsonl', [])
            dev = {'candidate_id': 'candidate', 'evaluation_split': 'dev', 'score': 0.8, 'evaluation_id': 'd'}
            heldout = {**dev, 'evaluation_split': 'test', 'evaluation_id': 't', 'score': 0.0}
            write_jsonl(source / 'evaluations.jsonl', [dev, heldout])
            prepare(source, root / 'v1', 'target')
            write_jsonl(source / 'evaluations.jsonl', [dev, {**heldout, 'score': 1.0}])
            prepare(source, root / 'v2', 'target')
            for strategy in ('best', 'random', 'topk'):
                _, parts = load_splits(root / 'v1' / strategy)
                self.assertEqual({r['task_id'] for r in parts['train']}, {'a'})
                self.assertEqual(read_jsonl(root / 'v1' / strategy / 'sft/train.jsonl'),
                                 read_jsonl(root / 'v2' / strategy / 'sft/train.jsonl'))

    def test_lora_without_quantization(self):
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_bf16_supported=lambda: True),
                                     bfloat16='bf16', float16='fp16')
        kwargs, bf16 = model_load_kwargs({'mode': 'lora', 'require_bf16': True}, fake_torch)
        self.assertTrue(bf16)
        self.assertNotIn('quantization_config', kwargs)
        self.assertEqual(kwargs['torch_dtype'], 'bf16')

    def test_require_bf16(self):
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_bf16_supported=lambda: False))
        with self.assertRaises(RuntimeError):
            model_load_kwargs({'mode': 'lora', 'require_bf16': True}, fake_torch)

    def test_supervision_selection(self):
        pool = [{'task_id': 't', 'comparison_group_id': 'g', 'source_candidate_id': str(i),
                 'label_dev_score': score, 'input': 'PROBLEM', 'output': text}
                for i, (score, text) in enumerate([(1.0, 'PROBLEM think carefully first'),
                    (0.95, 'PROBLEM inspect constraints and limits'), (0.2, 'PROBLEM use a simple approach')])]
        self.assertEqual(select_pool(pool, 'best', 42), [pool[0]])
        self.assertEqual(select_pool(pool, 'topk', 42), pool[:2])
        self.assertEqual(select_pool(pool, 'random', 42), select_pool(pool[::-1], 'random', 42))

    def test_default_gpu(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(configure_gpu(), '3')
            self.assertEqual(os.environ['CUDA_DEVICE_ORDER'], 'PCI_BUS_ID')

    def test_container_gpu_remapping(self):
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0'}, clear=True):
            self.assertEqual(configure_gpu(), '0')

    def test_reject_multiple_gpus_and_processes(self):
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0,3'}, clear=True):
            with self.assertRaises(ValueError):
                configure_gpu()
        with patch.dict(os.environ, {'WORLD_SIZE': '2'}, clear=True):
            with self.assertRaises(RuntimeError):
                configure_gpu()

    def test_completion_loss_and_overflow(self):
        row = {'input': 'problem', 'target_model': 'target', 'instruction': 'rewrite', 'output': 'answer'}
        encoded = encode_example(FakeTokenizer(), row, 1000)
        self.assertEqual(encoded['labels'][-7:], list(b'answer') + [999])
        self.assertTrue(all(x == -100 for x in encoded['labels'][:-7]))
        self.assertEqual(len(encoded['labels']), len(encoded['input_ids']))
        self.assertIsNone(encode_example(FakeTokenizer(), row, 10))

    def test_split_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_jsonl(root / 'tasks.jsonl', [{'task_id': 't', 'split': 'test'}])
            write_jsonl(root / 'sft/train.jsonl', [{'task_id': 't', 'split': 'train', 'input': 'x', 'output': 'x'}])
            write_jsonl(root / 'sft/validation.jsonl', [])
            with self.assertRaisesRegex(ValueError, 'Split leakage'):
                load_splits(root)

    def test_template(self):
        self.assertIn('Target code generation model: target\n\nProgramming problem:\nproblem',
                      conditioning('problem', 'target'))

    def test_code_extraction(self):
        self.assertEqual(extract_code('```python\ndef f():\n    return 1\n```'), 'def f():\n    return 1')


if __name__ == '__main__':
    unittest.main()
