"""Generate held-out prompts with original/static/base/SFT variants."""
import argparse
import json
import time
from pathlib import Path

from .common import configure_gpu, model_load_kwargs, read_jsonl, render, write_jsonl


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--data', default='CodePromptOpt/data')
    p.add_argument('--output', required=True)
    p.add_argument('--variant', choices=['original', 'static', 'base', 'sft'], required=True)
    p.add_argument('--adapter')
    p.add_argument('--split', choices=['validation', 'test'], default='test')
    p.add_argument('--max-new-tokens', type=int, default=2048)
    args = p.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding='utf-8'))
    tasks = [t for t in read_jsonl(Path(args.data) / 'tasks.jsonl') if t['split'] == args.split]
    if args.variant == 'sft' and not args.adapter:
        p.error('--adapter is required for SFT')
    if args.variant != 'sft' and args.adapter:
        p.error('--adapter is only allowed with --variant sft')
    model = tokenizer = None
    if args.variant in ('base', 'sft'):
        visible_gpu = configure_gpu()
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
        from peft import PeftModel
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA GPU required')
        if torch.cuda.device_count() != 1:
            raise RuntimeError('Expected exactly one visible GPU; check CUDA_VISIBLE_DEVICES')
        print(f'CUDA_VISIBLE_DEVICES={visible_gpu}; using cuda:0: {torch.cuda.get_device_name(0)}')
        set_seed(cfg['seed'])
        kwargs, _ = model_load_kwargs(cfg, torch)
        tokenizer = AutoTokenizer.from_pretrained(args.adapter or cfg['model'], revision=None if args.adapter else cfg['revision'])
        model = AutoModelForCausalLM.from_pretrained(cfg['model'], revision=cfg['revision'],
            **kwargs)
        if args.adapter:
            metadata = json.loads((Path(args.adapter) / 'adapter_config.json').read_text(encoding='utf-8'))
            if metadata['base_model_name_or_path'] != cfg['model']:
                raise ValueError('Adapter and base model differ')
            model = PeftModel.from_pretrained(model, args.adapter)
        model.eval()
    rows = []
    for task in tasks:
        start = time.perf_counter()
        truncated = False
        if args.variant == 'original':
            prompt = task['problem']
        elif args.variant == 'static':
            prompt = ('Write a correct Python solution. Respect the interface and all constraints. '
                      'Check boundary cases. Return only executable Python code.\n\n' + task['problem'])
        else:
            inputs = tokenizer(render(tokenizer, task['problem'], cfg['target_label']),
                               add_special_tokens=False, return_tensors='pt').to(model.device)
            limit = getattr(model.config, 'max_position_embeddings', 32768)
            if inputs.input_ids.shape[1] + args.max_new_tokens > limit:
                raise ValueError(f'Context budget exceeded: {task["task_id"]}')
            with torch.inference_mode():
                output = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                    do_sample=False, pad_token_id=tokenizer.eos_token_id)
            tokens = output[0, inputs.input_ids.shape[1]:]
            truncated = len(tokens) == args.max_new_tokens and tokens[-1].item() != tokenizer.eos_token_id
            prompt = tokenizer.decode(tokens, skip_special_tokens=True)
        rows.append({'task_id': task['task_id'], 'split': args.split, 'variant': args.variant,
                     'optimizer': cfg['model'], 'adapter': args.adapter, 'target_label': cfg['target_label'],
                     'prompt': prompt, 'preserves_problem_verbatim': task['problem'] in prompt,
                     'truncated': truncated, 'prompt_chars': len(prompt),
                     'latency_seconds': time.perf_counter() - start, 'seed': cfg['seed']})
        write_jsonl(args.output, rows)


if __name__ == '__main__':
    main()
