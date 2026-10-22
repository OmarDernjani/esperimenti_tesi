import hashlib
import json
import os
from pathlib import Path

INSTRUCTION = ('Optimize the instruction for the target code generation model. '
               'Preserve the complete original programming problem, including its constraints and requirements. '
               'Produce only the optimized prompt, without solving the problem.')


def configure_gpu():
    """Restrict CUDA before importing torch; containers may remap host GPU 3 to 0."""
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise RuntimeError('Run a single process; distributed training is not supported')
    os.environ.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')
    device = os.environ.setdefault('CUDA_VISIBLE_DEVICES', '3').strip()
    if not device or ',' in device or device == '-1':
        raise ValueError('CUDA_VISIBLE_DEVICES must select exactly one GPU (default: 3)')
    return device


def read_jsonl(path):
    with open(path, encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')


def conditioning(problem, target, instruction=INSTRUCTION):
    return f'{instruction}\n\nTarget code generation model: {target}\n\nProgramming problem:\n{problem}'


def render(tokenizer, problem, target, instruction=INSTRUCTION):
    return tokenizer.apply_chat_template(
        [{'role': 'user', 'content': conditioning(problem, target, instruction)}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def model_load_kwargs(config, torch):
    """Use the same base precision for training and optimizer inference."""
    mode = config.get('mode', 'qlora')
    if mode not in ('lora', 'qlora'):
        raise ValueError('mode must be lora or qlora')
    bf16 = torch.cuda.is_bf16_supported()
    if config.get('require_bf16', False) and not bf16:
        raise RuntimeError('This experiment requires a GPU supporting BF16')
    dtype = torch.bfloat16 if bf16 else torch.float16
    kwargs = dict(device_map={'': 0}, torch_dtype=dtype, attn_implementation='sdpa')
    if mode == 'qlora':
        from transformers import BitsAndBytesConfig
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype)
    return kwargs, bf16


def load_splits(data):
    data = Path(data)
    tasks = read_jsonl(data / 'tasks.jsonl')
    ids = {row['task_id']: row['split'] for row in tasks}
    if len(ids) != len(tasks):
        raise ValueError('Duplicate task IDs')
    parts = {}
    for split in ('train', 'validation'):
        parts[split] = read_jsonl(data / 'sft' / f'{split}.jsonl')
        for row in parts[split]:
            if ids.get(row['task_id']) != split or row['split'] != split:
                raise ValueError(f'Split leakage: {row["task_id"]}')
            if row['input'] not in row['output']:
                raise ValueError('Training output does not preserve original problem')
    return tasks, parts
