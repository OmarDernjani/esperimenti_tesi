"""LoRA/QLoRA with explicit completion-only labels, without silent truncation."""
import argparse
import json
from pathlib import Path

from .common import configure_gpu, digest, load_splits, model_load_kwargs, render, write_jsonl


def encode_example(tokenizer, row, max_length):
    prefix = tokenizer.encode(render(tokenizer, row['input'], row['target_model'], row['instruction']),
                              add_special_tokens=False)
    completion = tokenizer.encode(row['output'], add_special_tokens=False) + [tokenizer.eos_token_id]
    if len(prefix) + len(completion) > max_length:
        return None
    return {'input_ids': prefix + completion, 'attention_mask': [1] * (len(prefix) + len(completion)),
            'labels': [-100] * len(prefix) + completion}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--data', default='CodePromptOpt/data')
    parser.add_argument('--output', required=True)
    parser.add_argument('--audit-only', action='store_true', help='No model download or GPU needed')
    parser.add_argument('--prepare-only', action='store_true', help='Token audit without loading model')
    parser.add_argument('--resume', default=None)
    parser.add_argument('--max-steps', type=int, default=-1, help='Positive value for a smoke run')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'adapter').exists() and not args.audit_only and not args.prepare_only:
        raise ValueError('Adapter already exists: use a new output directory')
    tasks, parts = load_splits(args.data)
    report = {'config': config, 'task_counts': {s: sum(t['split'] == s for t in tasks)
              for s in ('train', 'validation', 'test')}, 'files': {name: digest(Path(args.data) / name)
              for name in ('tasks.jsonl', 'sft/train.jsonl', 'sft/validation.jsonl')}, 'splits': {}}
    for split, rows in parts.items():
        parts[split] = [r for r in rows if r['target_model'] == config['target_label']]
        report['splits'][split] = {'selected': len(parts[split]),
                                 'tasks': len({r['task_id'] for r in parts[split]})}
        if not parts[split]:
            raise ValueError(f'No samples for {split} / {config["target_label"]}')
    (out / 'audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    if args.audit_only:
        print(json.dumps(report, indent=2))
        return
    report['cuda_visible_devices'] = configure_gpu()
    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              Trainer, TrainingArguments, EarlyStoppingCallback, set_seed)
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    set_seed(config['seed'])
    tokenizer = AutoTokenizer.from_pretrained(config['model'], revision=config['revision'])
    tokenizer.pad_token = tokenizer.eos_token
    encoded, excluded = {}, []
    for split, rows in parts.items():
        encoded[split] = []
        for row in rows:
            sample = encode_example(tokenizer, row, config['max_length'])
            if sample is None:
                excluded.append({'split': split, 'sample_id': row['sample_id'], 'reason': 'over_max_length'})
            else:
                encoded[split].append(sample)
        report['splits'][split]['retained'] = len(encoded[split])
        if not encoded[split]:
            raise ValueError(f'All {split} samples exceed max_length')
    report['tokenizer_commit'] = tokenizer.init_kwargs.get('_commit_hash')
    write_jsonl(out / 'excluded.jsonl', excluded)
    (out / 'audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    if args.prepare_only:
        return
    if not torch.cuda.is_available():
        raise RuntimeError('QLoRA requires a CUDA GPU; use --audit-only for CPU checks')
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Expected exactly one visible GPU; check CUDA_VISIBLE_DEVICES')
    print(f'CUDA_VISIBLE_DEVICES={report["cuda_visible_devices"]}; using cuda:0: {torch.cuda.get_device_name(0)}')
    kwargs, bf16 = model_load_kwargs(config, torch)
    model = AutoModelForCausalLM.from_pretrained(
        config['model'], revision=config['revision'], **kwargs)
    report['model_commit'] = getattr(model.config, '_commit_hash', None)
    (out / 'audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    model.config.use_cache = False
    if config.get('mode', 'qlora') == 'qlora':
        model = prepare_model_for_kbit_training(model, gradient_checkpointing_kwargs={'use_reentrant': False})
    else:
        model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=config['lora_rank'], lora_alpha=config['lora_alpha'],
        lora_dropout=config['lora_dropout'], target_modules=config.get('target_modules', 'all-linear'),
        task_type='CAUSAL_LM', bias='none'))
    model.print_trainable_parameters()

    # Give every task the same probability regardless of the number of Top-k prompts
    # retained after token filtering. Sampling remains reproducible through Trainer's seed.
    from collections import Counter
    retained_ids = {}
    excluded_ids = {r['sample_id'] for r in excluded}
    for split, rows in parts.items():
        retained_ids[split] = [r['task_id'] for r in rows if r['sample_id'] not in excluded_ids]
    counts = Counter(retained_ids['train'])

    class TaskBalancedTrainer(Trainer):
        def _get_train_sampler(self):
            if not config.get('balance_tasks', False):
                return super()._get_train_sampler()
            generator = torch.Generator().manual_seed(config['seed'])
            return torch.utils.data.WeightedRandomSampler(
                [1.0 / counts[tid] for tid in retained_ids['train']],
                len(counts) * config.get('samples_per_task_per_epoch', 1),
                replacement=True, generator=generator)

    def collate(rows):
        length = max(len(r['input_ids']) for r in rows)
        return {key: torch.tensor([r[key] + [pad] * (length - len(r[key])) for r in rows])
                for key, pad in (('input_ids', tokenizer.pad_token_id), ('attention_mask', 0), ('labels', -100))}

    trainer = TaskBalancedTrainer(model=model, train_dataset=encoded['train'], eval_dataset=encoded['validation'],
        data_collator=collate, callbacks=[EarlyStoppingCallback(config['early_stopping_patience'])],
        args=TrainingArguments(output_dir=str(out / 'checkpoints'), learning_rate=config['learning_rate'],
            num_train_epochs=config['epochs'], max_steps=args.max_steps,
            per_device_train_batch_size=config.get('batch_size', 1), per_device_eval_batch_size=1,
            gradient_accumulation_steps=config['gradient_accumulation_steps'],
            gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
            bf16=bf16, fp16=not bf16, eval_strategy='epoch', save_strategy='epoch',
            load_best_model_at_end=True, metric_for_best_model='eval_loss', greater_is_better=False,
            label_names=['labels'],
            save_total_limit=2, logging_steps=1, warmup_ratio=0.05, weight_decay=0.01,
            lr_scheduler_type='cosine', seed=config['seed'], report_to='none'))
    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(str(out / 'adapter'))
    tokenizer.save_pretrained(out / 'adapter')
    (out / 'experiment.json').write_text(json.dumps({**config, 'revision': report['model_commit'] or config['revision'],
        'smoke_run': args.max_steps > 0}, indent=2), encoding='utf-8')
    import subprocess
    import sys
    environment = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)
    (out / 'environment.txt').write_text(environment, encoding='utf-8')
    (out / 'history.json').write_text(json.dumps(trainer.state.log_history, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
