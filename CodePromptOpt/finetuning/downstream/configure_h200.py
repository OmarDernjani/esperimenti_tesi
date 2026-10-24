"""Persist exact code-generator identity before collecting responses."""
import argparse
import json
from pathlib import Path

from .run import atomic_json


def make_config(model, revision, image, context, gpu_util, max_seqs, experiment):
    return {'experiment_id': experiment, 'model': model,
            'parameters': {'temperature': 0, 'top_p': 1, 'max_tokens': 2048, 'seed': 42},
            'extra_body': {}, 'server_metadata': {'model_revision': revision,
            'image_id': image, 'dtype': 'bfloat16', 'max_model_len': context,
            'gpu_memory_utilization': gpu_util, 'max_num_seqs': max_seqs, 'hardware_profile': 'h200'}}


def save_config(path, cfg):
    path = Path(path)
    if path.exists() and json.loads(path.read_text(encoding='utf-8')) != cfg:
        raise ValueError('Server identity/settings changed: choose a new CPO_CODEGEN_EXPERIMENT in h200.env')
    atomic_json(path, cfg)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--revision', required=True)
    p.add_argument('--image', required=True)
    p.add_argument('--context', required=True, type=int)
    p.add_argument('--gpu-util', required=True, type=float)
    p.add_argument('--max-seqs', required=True, type=int)
    p.add_argument('--experiment', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    save_config(args.output, make_config(args.model, args.revision, args.image, args.context,
                                        args.gpu_util, args.max_seqs, args.experiment))
    print(f'Code-generator configuration: {args.output}')


if __name__ == '__main__':
    main()
