"""Run annotation/selection requests against an explicitly configured local API."""
import argparse
import json
from pathlib import Path

from ..finetuning.common import read_jsonl
from ..finetuning.downstream.run import atomic_json, atomic_jsonl, request_json
from .core import fingerprint


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--model-revision', required=True, help='Actual served revision, recorded as declared metadata')
    p.add_argument('--base-url', default='http://127.0.0.1:8000/v1')
    p.add_argument('--max-tokens', type=int, default=2048)
    p.add_argument('--timeout', type=float, default=600)
    p.add_argument('--server-config', help='Recorded serving identity/settings, included in protocol fingerprint')
    a = p.parse_args()
    if a.max_tokens < 1 or a.timeout <= 0:
        p.error('Positive token budget and timeout required')
    requests = read_jsonl(a.input)
    if len({r['id'] for r in requests}) != len(requests):
        p.error('Duplicate request IDs')
    protocol = {'model': a.model, 'declared_model_revision': a.model_revision,
                'temperature': 0, 'top_p': 1, 'seed': 42, 'max_tokens': a.max_tokens,
                'input_hash': fingerprint(requests), 'base_url': a.base_url.rstrip('/')}
    if a.server_config:
        server = json.loads(Path(a.server_config).read_text(encoding='utf-8'))
        if server['model'] != a.model or server['server_metadata']['model_revision'] != a.model_revision:
            p.error('Server config does not match selected model/revision')
        protocol['server_config'] = server
    manifest = Path(a.output + '.manifest.json')
    if manifest.exists() and json.loads(manifest.read_text(encoding='utf-8')) != protocol:
        p.error('Inference protocol changed; choose a new output')
    if Path(a.output).exists() and not manifest.exists():
        p.error('Existing output has no protocol manifest')
    atomic_json(manifest, protocol)
    completed = read_jsonl(a.output) if Path(a.output).exists() else []
    done = {r['id'] for r in completed}
    if len(done) != len(completed) or done - {r['id'] for r in requests}:
        p.error('Invalid existing response IDs')
    for r in requests:
        if r['id'] in done:
            continue
        raw = request_json(protocol['base_url'] + '/chat/completions', {
            'model': a.model, 'messages': r['messages'], 'temperature': 0,
            'top_p': 1, 'seed': 42, 'max_tokens': a.max_tokens,
            'response_format': {'type': 'json_object'}}, timeout=a.timeout)
        choice = raw['choices'][0]
        if choice.get('finish_reason') != 'stop':
            raise ValueError('Incomplete JSON response for ' + r['id'])
        response = json.loads(choice['message']['content'])
        completed.append({**{k: v for k, v in r.items() if k != 'messages'},
                          'response': response, 'raw_response': raw, 'metadata': protocol})
        atomic_jsonl(a.output, completed)
        print(f'Completed {len(completed)}/{len(requests)}', flush=True)


if __name__ == '__main__':
    main()
