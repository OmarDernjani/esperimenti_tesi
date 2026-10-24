"""Plan/run H200 experiments sequentially inside a single GPU container."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
import uuid

from .prepare_experiments import prepare


def archive_run(root, run, stage):
    """Preserve old adapter, checkpoints and prompts before a fresh training."""
    root = Path(root).resolve()
    run = Path(run)
    expected_parent = root / ('smoke' if stage == 'smoke' else 'runs')
    if run.is_symlink() or run.resolve().parent != expected_parent:
        raise ValueError(f'Refusing to archive unexpected run path: {run}')
    if not run.exists():
        return None
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8]
    destination = root / 'archive' / stamp / stage / run.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    run.rename(destination)
    print(f'Previous run archived: {destination}', flush=True)
    return destination


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--matrix', default='CodePromptOpt/finetuning/configs/h200_matrix.json')
    p.add_argument('--data', default='CodePromptOpt/data')
    p.add_argument('--output', default='CodePromptOpt/finetuning/outputs/h200')
    p.add_argument('--models', nargs='+')
    p.add_argument('--seeds', nargs='+', type=int)
    p.add_argument('--strategies', nargs='+', choices=['best', 'random', 'topk'])
    p.add_argument('--stage', choices=['plan', 'prepare', 'smoke', 'train', 'generate'], default='plan')
    p.add_argument('--split', choices=['validation', 'test'], default='validation')
    p.add_argument('--pin-revisions', action='store_true', help='Resolve HF commits during planning (network required)')
    p.add_argument('--retrain', action='store_true', help='Archive selected old runs and train from scratch (train/smoke only)')
    args = p.parse_args()
    if args.retrain and args.stage not in ('train', 'smoke'):
        p.error('--retrain is only supported for train and smoke')
    matrix = json.loads(Path(args.matrix).read_text(encoding='utf-8'))
    models, seeds = args.models or matrix['default_models'], args.seeds or matrix['seeds']
    strategies = args.strategies or matrix['strategies']
    unknown = set(models) - matrix['models'].keys()
    if unknown:
        p.error(f'Unknown models: {sorted(unknown)}')
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    pins_file = root / 'revisions.json'
    pins = json.loads(pins_file.read_text(encoding='utf-8')) if pins_file.exists() else {}
    if args.pin_revisions:
        if args.stage != 'plan':
            p.error('--pin-revisions is only supported with --stage plan')
        from huggingface_hub import HfApi
        api = HfApi()
        for name in models:
            model_id = matrix['models'][name]['model']
            if model_id not in pins:
                revision = matrix['models'][name].get('revision', matrix['defaults']['revision'])
                pins[model_id] = api.model_info(model_id, revision=revision).sha
        pins_file.write_text(json.dumps(pins, indent=2), encoding='utf-8')
    if args.stage != 'plan' and any(matrix['models'][m]['model'] not in pins for m in models):
        p.error('Resolve model commits first: plan --pin-revisions (use a fresh output if already planned without pins)')
    target = matrix['defaults']['target_label']
    datasets = root / 'datasets'
    # Never rewrite the prepared datasets during training or evaluation.
    if args.stage == 'plan':
        prepare(args.data, datasets, target)
    elif not (datasets / 'manifest.json').is_file():
        p.error('Run --stage plan before any GPU stage')
    plan = []
    for model in models:
        for strategy in strategies:
            for seed in seeds:
                run = root / 'runs' / f'{model}-{strategy}-s{seed}'
                cfg = {**matrix['defaults'], **matrix['models'][model], 'seed': seed,
                       'supervision': strategy}
                cfg['revision'] = pins.get(cfg['model'], cfg['revision'])
                cfg_path = root / 'configs' / f'{model}-{strategy}-s{seed}.json'
                cfg_path.parent.mkdir(parents=True, exist_ok=True)
                if cfg_path.exists() and json.loads(cfg_path.read_text(encoding='utf-8')) != cfg:
                    raise ValueError(f'Config changed for existing experiment: {cfg_path}; use a new output directory')
                cfg_path.write_text(json.dumps(cfg, indent=2), encoding='utf-8')
                command = [sys.executable, '-m', 'CodePromptOpt.finetuning.train', '--config', str(cfg_path),
                           '--data', str(datasets / strategy), '--output', str(run)]
                plan.append({'run': str(run), 'config': str(cfg_path), 'command': command})
                if args.stage == 'plan':
                    continue
                if args.stage == 'generate':
                    trained_cfg = run / 'experiment.json'
                    if not trained_cfg.exists():
                        raise ValueError(f'Missing trained experiment: {run}')
                    if json.loads(trained_cfg.read_text(encoding='utf-8')).get('smoke_run'):
                        raise ValueError('Smoke adapters cannot be used for final evaluation')
                    for variant in ('original', 'static', 'base', 'sft'):
                        dest = run / f'{args.split}-{variant}-prompts.jsonl'
                        if dest.exists():
                            raise ValueError(f'Refusing to overwrite prompts: {dest}')
                        gen = [sys.executable, '-m', 'CodePromptOpt.finetuning.generate', '--config', str(trained_cfg),
                               '--data', args.data, '--split', args.split, '--variant', variant, '--output', str(dest)]
                        if variant == 'sft':
                            gen += ['--adapter', str(run / 'adapter')]
                        subprocess.run(gen, check=True)
                elif args.stage == 'prepare':
                    subprocess.run(command + ['--prepare-only'], check=True)
                elif args.stage == 'smoke':
                    command[-1] = str(root / 'smoke' / run.name)
                    if args.retrain:
                        archive_run(root, Path(command[-1]), 'smoke')
                    subprocess.run(command + ['--max-steps', '2'], check=True)
                elif args.retrain:
                    archive_run(root, run, 'train')
                    subprocess.run(command, check=True)
                elif (run / 'adapter' / 'adapter_config.json').exists() and (run / 'experiment.json').exists():
                    print(f'Skip completed: {run}', flush=True)
                else:
                    if (run / 'checkpoints').exists():
                        raise ValueError(f'Incomplete run {run}: resume explicitly with train --resume or use a new output')
                    subprocess.run(command, check=True)
    if args.stage == 'plan':
        (root / 'plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    print(f'{len(plan)} experiments selected; stage={args.stage}', flush=True)


if __name__ == '__main__':
    main()
