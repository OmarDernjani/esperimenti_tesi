"""Send prompts to one OpenAI-compatible code generator (Ollama/vLLM/etc.)."""
import argparse
import json
import os
import re
import time
import urllib.request

from .common import read_jsonl, write_jsonl


def extract_code(text):
    blocks = re.findall(r'```(?:python|py)?\s*\n(.*?)```', text, flags=re.S)
    return blocks[0].strip() if blocks else text.strip()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prompts', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--base-url', required=True, help='e.g. http://localhost:11434/v1')
    p.add_argument('--model', required=True, help='Exact runtime model ID, not historical dataset label')
    p.add_argument('--max-tokens', type=int, default=2048)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    rows = []
    for row in read_jsonl(args.prompts):
        body = {'model': args.model, 'messages': [
            {'role': 'system', 'content': 'Return only the complete executable Python solution, without Markdown.'},
            {'role': 'user', 'content': row['prompt']}], 'temperature': 0,
            'max_tokens': args.max_tokens, 'seed': args.seed}
        req = urllib.request.Request(args.base_url.rstrip('/') + '/chat/completions',
            data=json.dumps(body).encode(), headers={'Content-Type': 'application/json',
            'Authorization': 'Bearer ' + os.environ.get('CODEGEN_API_KEY', 'local')})
        start = time.perf_counter()
        with urllib.request.urlopen(req, timeout=600) as response:
            result = json.load(response)
        choice = result['choices'][0]
        rows.append({**row, 'code': extract_code(choice['message']['content']),
                     'codegen_model': args.model, 'codegen_parameters': {'temperature': 0,
                     'max_tokens': args.max_tokens, 'seed': args.seed},
                     'finish_reason': choice.get('finish_reason'), 'usage': result.get('usage'),
                     'codegen_latency_seconds': time.perf_counter() - start})
        write_jsonl(args.output, rows)


if __name__ == '__main__':
    main()
