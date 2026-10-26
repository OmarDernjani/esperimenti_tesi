"""Run individual DPO stages inside the GPU container."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

from ..common import digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['tokens', 'smoke', 'train', 'generate'])
    p.add_argument('--root', required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--init-adapter')
    p.add_argument('--run-name')
    p.add_argument('--split', choices=['validation', 'test'], default='validation')
    args = p.parse_args()
    name = args.run_name or f'qwen3_8b-dpo-s{args.seed}'
    if not re.fullmatch(r'[a-zA-Z0-9_-][a-zA-Z0-9_.-]*', name):
        p.error('Invalid run name')
    root = Path(args.root)
    run = root / 'runs' / name

    def execute(module, *argv):
        subprocess.run([sys.executable, '-m', module, *map(str, argv)], check=True)

    if args.stage != 'generate':
        out = run if args.stage == 'train' else root / args.stage / name
        command = ['--config', args.config, '--data', root / 'data', '--output', out, '--seed', args.seed]
        if args.init_adapter:
            command += ['--init-adapter', args.init_adapter]
        if args.stage == 'tokens':
            command += ['--prepare-only']
        elif args.stage == 'smoke':
            command += ['--max-steps', 2]
        execute('CodePromptOpt.finetuning.dpo.train', *command)
        return
    if args.init_adapter:
        p.error('Generate reads initialization metadata from the saved run; omit --init-adapter')
    if not (run / 'metrics.json').is_file():
        raise ValueError('Training is not complete: metrics.json missing')
    audit = json.loads((run / 'audit.json').read_text(encoding='utf-8'))
    if audit['initial_adapter'] and digest(Path(audit['initial_adapter']) / 'adapter_model.safetensors') != audit['initial_adapter_sha256']:
        raise ValueError('Initial SFT adapter changed after training')
    for variant in ('original', 'static', 'base', 'sft'):
        output = run / f'{args.split}-{variant}-prompts.jsonl'
        if output.exists():
            raise ValueError(f'Output exists (may be partial); archive before retry: {output}')
    for variant in ('original', 'static', 'base', 'sft'):
        command = ['--config', run / 'config.json', '--output', run / f'{args.split}-{variant}-prompts.jsonl',
                   '--variant', variant, '--split', args.split]
        if variant == 'sft':
            command += ['--adapter', run / 'adapter']
        execute('CodePromptOpt.finetuning.generate', *command)
    if audit['initial_adapter']:
        # The same initialization is the causal baseline for SFT -> DPO.
        baseline = root / 'runs' / (name + '-initial-sft')
        baseline.mkdir(parents=True, exist_ok=True)
        for variant in ('original', 'static', 'base', 'sft'):
            output = baseline / f'{args.split}-{variant}-prompts.jsonl'
            if output.exists():
                raise ValueError(f'Initial SFT output exists: {output}')
            command = ['--config', run / 'config.json', '--output', output,
                       '--variant', variant, '--split', args.split]
            if variant == 'sft':
                command += ['--adapter', audit['initial_adapter']]
            execute('CodePromptOpt.finetuning.generate', *command)


if __name__ == '__main__':
    main()
