"""Validate exported artifacts against canonical relations and original source bytes."""
import argparse
import hashlib
import json
from pathlib import Path

from .build import extract, training_data, validate


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def validate_directory(directory, input_dir):
    directory, input_dir = Path(directory), Path(input_dir)
    manifest = json.loads((directory / 'build_manifest.json').read_text(encoding='utf-8'))
    split_file = json.loads((directory / 'splits.json').read_text(encoding='utf-8'))
    splits = {key: split_file[key] for key in ('train', 'validation', 'test')}
    data = {}
    for key in ('tasks', 'runs', 'candidates', 'evaluations'):
        id_key = key[:-1] + '_id'
        rows = load_jsonl(directory / f'{key}.jsonl')
        data[key] = {row[id_key]: row for row in rows}
        if len(data[key]) != len(rows):
            raise ValueError(f'Duplicate canonical IDs in {key}')
    data['filtering'] = load_jsonl(directory / 'filtering_report.jsonl')
    sft = sum((load_jsonl(directory / 'sft' / f'{split}.jsonl') for split in ('train', 'validation')), [])
    dpo = sum((load_jsonl(directory / 'dpo' / f'{split}.jsonl') for split in ('train', 'validation')), [])
    result = validate(data, splits, sft, dpo, manifest['min_dpo_score_margin'],
                      manifest['max_dpo_pairs_per_task_target'], manifest['quality_filtering'])
    for item in manifest['input_sources']:
        source = input_dir / item['source_file']
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if digest != item['sha256']:
            result['errors'].append('Source checksum mismatch: ' + item['source_file'])
    hashes = json.loads((directory / 'artifact_hashes.json').read_text(encoding='utf-8'))
    for filename, digest in hashes.items():
        path = directory / filename
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            result['errors'].append('Artifact checksum mismatch: ' + filename)
    # Regeneration from the inputs independently verifies every pointer-backed
    # canonical record, including exact prompt/code text, score fields and aliases.
    regenerated = extract(input_dir)
    membership = {tid: split for split, tids in splits.items() for tid in tids}
    for key in ('tasks', 'runs', 'candidates', 'evaluations'):
        if set(data[key]) != set(regenerated[key]):
            result['errors'].append('Canonical ID mismatch against sources: ' + key)
            continue
        for identifier, observed in data[key].items():
            expected = dict(regenerated[key][identifier])
            tid = expected.get('task_id') or regenerated['candidates'][expected['candidate_id']]['task_id']
            expected['split'] = membership.get(tid)
            if observed != expected:
                result['errors'].append('Canonical value/source provenance mismatch: ' + identifier)
            result['assertions'] += 1
    # Recompute labels to verify best-dev selection and deterministic pair sampling,
    # rather than merely checking that the selected score exists somewhere.
    expected_sft, expected_dpo, _, _ = training_data(
        regenerated, splits, manifest['min_dpo_score_margin'], manifest['max_dpo_pairs_per_task_target'],
        manifest['prompt_deduplication'], manifest['quality_filtering'])
    if sorted(sft, key=lambda s: s['sample_id']) != sorted(expected_sft, key=lambda s: s['sample_id']):
        result['errors'].append('SFT records differ from dev-only selection from original sources')
    if sorted(dpo, key=lambda s: s['sample_id']) != sorted(expected_dpo, key=lambda s: s['sample_id']):
        result['errors'].append('DPO records differ from dev-only construction from original sources')
    result['passed'] = not result['errors']
    result['source_provenance_verified'] = True
    result['artifact_checksums_verified'] = not any('checksum mismatch' in e for e in result['errors'])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', default='CodePromptOpt/data')
    parser.add_argument('--input-dir', default='dataset_code_prompt')
    args = parser.parse_args()
    result = validate_directory(args.data_dir, args.input_dir)
    print(json.dumps(result, sort_keys=True, indent=2))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
