"""Single-GPU BF16 LoRA DPO, optionally initialized from an SFT adapter."""
import argparse
import json
from pathlib import Path
import re

from ..common import configure_gpu, digest, model_load_kwargs, read_jsonl, render, write_jsonl


def trainer_class():
    from trl import DPOTrainer

    class AuditedDPOTrainer(DPOTrainer):
        # TRL 0.17 always tokenizes raw strings in _prepare_dataset. Our rows
        # already include the exact no-thinking prefix and EOS checked below.
        def _prepare_dataset(self, dataset, processing_class, args, dataset_name):
            required = {'prompt_input_ids', 'chosen_input_ids', 'rejected_input_ids'}
            if not required.issubset(dataset.column_names):
                raise ValueError('Expected audited tokenized preference pairs')
            return dataset

    return AuditedDPOTrainer


def tokenize_pairs(tokenizer, rows, max_length):
    kept, excluded = [], []
    for row in rows:
        prompt = render(tokenizer, row['input'], row['target_model'], row['instruction'])
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        choices = {k: tokenizer.encode(row[k], add_special_tokens=False) + [tokenizer.eos_token_id]
                   for k in ('chosen', 'rejected')}
        if any(len(ids) + len(v) > max_length for v in choices.values()):
            excluded.append({'sample_id': row['sample_id'], 'reason': 'over_max_length'})
        else:
            kept.append({'prompt_input_ids': ids, 'chosen_input_ids': choices['chosen'],
                         'rejected_input_ids': choices['rejected']})
    return kept, excluded


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--data', required=True, help='Prepared DPO directory')
    p.add_argument('--output', required=True)
    p.add_argument('--init-adapter', help='Optional SFT adapter: both policy and frozen reference start here')
    p.add_argument('--seed', type=int)
    p.add_argument('--max-steps', type=int, default=-1)
    p.add_argument('--prepare-only', action='store_true', help='Tokenizer audit; no model weights')
    args = p.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding='utf-8'))
    if args.seed is not None:
        cfg['seed'] = args.seed
    out, data = Path(args.output), Path(args.data)
    if not args.prepare_only and out.exists() and any(out.iterdir()):
        raise ValueError('Run output already exists; choose a fresh directory (no silent overwrite)')
    audit = json.loads((data / 'audit.json').read_text(encoding='utf-8'))
    for key in ('target_label', 'minimum_margin', 'pairs_per_task'):
        if audit['config'][key] != cfg[key]:
            raise ValueError(f'Prepared dataset/config mismatch: {key}')
    for split in ('train', 'validation'):
        if digest(data / f'{split}.jsonl') != audit['files'][f'{split}.jsonl']:
            raise ValueError(f'Prepared data changed: {split}')
    visible = configure_gpu()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from peft import LoraConfig, PeftModel
    from datasets import Dataset
    from trl import DPOConfig
    set_seed(cfg['seed'])
    if args.init_adapter:
        initial_meta = json.loads((Path(args.init_adapter) / 'adapter_config.json').read_text(encoding='utf-8'))
        if initial_meta['base_model_name_or_path'] != cfg['model']:
            raise ValueError('SFT adapter belongs to a different model')
        initial_audit = Path(args.init_adapter).parent / 'audit.json'
        if not initial_audit.is_file():
            raise ValueError('SFT audit.json is needed to verify the original base revision')
        initial_commit = json.loads(initial_audit.read_text(encoding='utf-8')).get('model_commit')
        if not initial_commit:
            raise ValueError('SFT audit lacks model_commit')
        if cfg['revision'] != 'main' and cfg['revision'] != initial_commit:
            raise ValueError('DPO config and SFT initialization use different base revisions')
        cfg['revision'] = initial_commit
    if not re.fullmatch(r'[a-fA-F0-9]{40}', cfg['revision']):
        from huggingface_hub import HfApi
        cfg['revision'] = HfApi().model_info(cfg['model'], revision=cfg['revision']).sha
    tokenizer = AutoTokenizer.from_pretrained(cfg['model'], revision=cfg['revision'])
    tokenizer.pad_token = tokenizer.eos_token
    encoded, excluded = {}, []
    for split in ('train', 'validation'):
        encoded[split], dropped = tokenize_pairs(tokenizer, read_jsonl(data / f'{split}.jsonl'), cfg['max_length'])
        excluded.extend(dict(r, split=split) for r in dropped)
        if not encoded[split]:
            raise ValueError(f'No pairs fit context in {split}')
    report = {'config': cfg, 'initial_adapter': args.init_adapter, 'cuda_visible_devices': visible,
              'dataset_audit': audit, 'retained': {s: len(v) for s, v in encoded.items()},
              'tokenizer_commit': cfg['revision']}
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / 'excluded.jsonl', excluded)
    (out / 'audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (out / 'config.json').write_text(json.dumps(cfg, indent=2), encoding='utf-8')
    if args.prepare_only:
        print(json.dumps(report['retained'], indent=2))
        return
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one CUDA GPU required')
    print(f'CUDA_VISIBLE_DEVICES={visible}; using cuda:0: {torch.cuda.get_device_name(0)}')
    kwargs, bf16 = model_load_kwargs(cfg, torch)
    # Reuse the exact tokenizer snapshot for weights and later inference.
    cfg['revision'] = report['tokenizer_commit'] or cfg['revision']
    model = AutoModelForCausalLM.from_pretrained(cfg['model'], revision=cfg['revision'], **kwargs)
    model.config.use_cache = False
    adapter_names = {}
    peft_config = LoraConfig(r=cfg['lora_rank'], lora_alpha=cfg['lora_alpha'],
                            lora_dropout=cfg['lora_dropout'], target_modules=cfg['target_modules'],
                            task_type='CAUSAL_LM', bias='none')
    if args.init_adapter:
        meta = json.loads((Path(args.init_adapter) / 'adapter_config.json').read_text(encoding='utf-8'))
        if meta['base_model_name_or_path'] != cfg['model']:
            raise ValueError('SFT adapter belongs to a different model')
        model = PeftModel.from_pretrained(model, args.init_adapter, adapter_name='default', is_trainable=True)
        model.load_adapter(args.init_adapter, adapter_name='reference', is_trainable=False)
        model.set_adapter('default')
        adapter_names = {'model_adapter_name': 'default', 'ref_adapter_name': 'reference'}
        report['initial_adapter_sha256'] = digest(Path(args.init_adapter) / 'adapter_model.safetensors')
        peft_config = None
    # Pretokenized datasets bypass TRL chat rendering/tokenization. All lengths
    # were checked including EOS; no prompt or completion is silently truncated.
    training = DPOConfig(output_dir=str(out / 'checkpoints'), beta=cfg['beta'], loss_type='sigmoid',
        learning_rate=cfg['learning_rate'], num_train_epochs=cfg['epochs'], max_steps=args.max_steps,
        per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=cfg['gradient_accumulation_steps'],
        gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
        bf16=bf16, fp16=not bf16, optim='adamw_torch', warmup_ratio=0.1,
        max_length=cfg['max_length'], max_prompt_length=None, max_completion_length=None,
        precompute_ref_log_probs=True, precompute_ref_batch_size=1,
        eval_strategy='epoch', save_strategy='no', logging_steps=1, report_to='none',
        seed=cfg['seed'], data_seed=cfg['seed'], **adapter_names)
    trainer = trainer_class()(model=model, ref_model=None, peft_config=peft_config, args=training,
        processing_class=tokenizer, train_dataset=Dataset.from_list(encoded['train']),
        eval_dataset=Dataset.from_list(encoded['validation']))
    metrics = {'before': trainer.evaluate()}
    result = trainer.train()
    metrics.update(train=result.metrics, after=trainer.evaluate())
    # Only save the policy adapter, never the reference or a full base-model copy.
    trainer.model.save_pretrained(out / 'adapter', selected_adapters=['default'])
    tokenizer.save_pretrained(out / 'adapter')
    (out / 'history.json').write_text(json.dumps(trainer.state.log_history, indent=2), encoding='utf-8')
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    report['model_commit'] = getattr(model.config, '_commit_hash', None)
    (out / 'audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (out / 'config.json').write_text(json.dumps(cfg, indent=2), encoding='utf-8')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
