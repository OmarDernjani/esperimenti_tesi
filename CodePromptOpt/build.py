"""Build CodePromptOpt without importing or executing experimental code."""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import re
import statistics

INSTRUCTION = ('Optimize the instruction for the target code generation model. '
               'Preserve the complete original programming problem, including its constraints and requirements. '
               'Produce only the optimized prompt, without solving the problem.')
METHODS = ('zero_shot', 'human_prompt', 'baseline', 'APE', 'APO')
MODEL_LABELS = {'qwen3.5': 'qwen3.5', 'qwen3.5-coder': 'qwen3.5-coder'}


def uid(prefix, *parts):
    data = json.dumps(parts, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return prefix + '_' + hashlib.sha256(data.encode('utf-8')).hexdigest()[:24]


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2,
                               allow_nan=False) + '\n', encoding='utf-8')


def jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='\n') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + '\n')


def documents(text):
    """Support JSON arrays, JSONL, and concatenated JSON documents explicitly."""
    decoder, pos, index = json.JSONDecoder(), 0, 0
    while pos < len(text):
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if pos == len(text):
            break
        value, pos = decoder.raw_decode(text, pos)
        for row_index, row in enumerate(value if isinstance(value, list) else [value]):
            yield index, row_index, row, f'/{row_index}' if isinstance(value, list) else ''
        index += 1


def model_configuration(path):
    """Names identify target first, optimizer second; exact runtime metadata is absent."""
    stem = path.stem
    if stem.endswith('_self'):
        target = optimizer = stem[:-5]
    elif '+' in stem:
        target, optimizer = stem.split('+', 1)
    else:
        target = optimizer = None
    return MODEL_LABELS.get(target), MODEL_LABELS.get(optimizer)


def entry_point(problem):
    # Only read the original function signature, never infer from a completion.
    try:
        module = ast.parse(problem)
        functions = [n.name for n in module.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        return functions[0] if len(functions) == 1 else None
    except SyntaxError:
        return None


def source_ref(file, document, pointer):
    return {'source_file': file, 'source_document': document, 'source_pointer': pointer}


def score_valid(score):
    return isinstance(score, (int, float)) and not isinstance(score, bool) and math.isfinite(score) and 0 <= score <= 1


def extract(input_dir):
    tasks, runs, candidates, evaluations = {}, {}, {}, {}
    inventory, issues, archive, schemas = [], [], [], defaultdict(set)
    counts = Counter()
    seen_rows = {}
    for path in sorted(input_dir.rglob('*')):
        if path.suffix.lower() not in ('.json', '.jsonl') or not path.is_file():
            continue
        relative = path.relative_to(input_dir).as_posix()
        raw = path.read_bytes()
        parsed = list(documents(raw.decode('utf-8-sig')))
        inventory.append({'source_file': relative, 'sha256': hashlib.sha256(raw).hexdigest(),
                          'bytes': len(raw), 'documents': max((d for d, _, _, _ in parsed), default=-1) + 1,
                          'rows': len(parsed)})
        target, optimizer = model_configuration(path)
        for document, row_index, row, pointer in parsed:
            ref = source_ref(relative, document, pointer)
            if not isinstance(row, dict):
                issues.append({**ref, 'reason': 'non_object_record'})
                continue
            if 'question' not in row:
                if 'problem' in row and isinstance(row.get('prompts'), list):
                    archive.append((row, ref))
                else:
                    issues.append({**ref, 'reason': 'unrecognized_record_schema'})
                continue
            problem = row['question']
            if not isinstance(problem, str) or not problem.strip():
                issues.append({**ref, 'reason': 'missing_or_empty_problem'})
                continue
            benchmark = 'HumanEval+' if str(row.get('task_id', '')).startswith('HumanEval/') else 'APPS'
            task_id = uid('task', benchmark, problem)
            task = tasks.setdefault(task_id, {
                'task_id': task_id, 'benchmark': benchmark, 'source_id': row.get('task_id'),
                'problem': problem, 'difficulty': row.get('difficulty'), 'language': 'python',
                'interface': 'call-based' if benchmark == 'HumanEval+' else None,
                'entry_point': entry_point(problem) if benchmark == 'HumanEval+' else None,
                'source_records': [], 'local_problem_indices': [],
            })
            task['source_records'].append(ref)
            if row.get('problem_idx') not in task['local_problem_indices']:
                task['local_problem_indices'].append(row.get('problem_idx'))
            digest = uid('record', relative, row)
            if digest in seen_rows:
                counts['duplicate_task_configuration_records'] += 1
                issues.append({**ref, 'reason': 'identical_duplicate_record', 'original': seen_rows[digest]})
                continue
            seen_rows[digest] = ref
            counts['task_configurations'] += 1
            group = uid('conditions', relative, document, row_index, task_id)
            for method in METHODS:
                body = row.get(method)
                if not isinstance(body, dict):
                    issues.append({**ref, 'method': method, 'reason': 'missing_method'})
                    continue
                schemas[method].update(body)
                run_id = uid('run', group, method)
                run = {
                    'run_id': run_id, 'task_id': task_id, 'optimizer_method': method,
                    'optimizer_model': optimizer if method in ('APE', 'APO', 'baseline') else None,
                    'target_model': target, 'model_quantization': None, 'experiment_version': None,
                    'decoding_parameters': None, 'seed': None, 'comparison_group_id': group,
                    'comparison_evidence': 'same stored task record; main.py shares dev_io across APE/APO',
                    'configuration_provenance': 'filename labels; target first, optimizer second; runtime metadata not persisted',
                    'n_dev': body.get('n_dev', row.get('n_dev')),
                    'n_test': body.get('n_test', row.get('n_test')),
                    'source_file': relative, 'source_document': document,
                    'source_pointer': ref['source_pointer'] + '/' + method,
                }
                runs[run_id] = run
                by_prompt = {}
                observation_registry = defaultdict(list)

                def observe(item, pointer, iteration=None, primary=False, selected=False):
                    """Deduplicate candidates by exact prompt, evaluations by documented copies.

                    Iteration lists are authoritative observation events. Final pools and
                    best summaries are copies; beam entries enrich pool observations with code.
                    Distinct primary occurrences stay distinct even when completions coincide.
                    """
                    if not isinstance(item, dict):
                        issues.append({**ref, 'method': method, 'reason': 'non_object_candidate', 'pointer': pointer})
                        return
                    prompt = item.get('prompt')
                    if not isinstance(prompt, str):
                        issues.append({**ref, 'method': method, 'reason': 'missing_prompt', 'pointer': pointer})
                        return
                    counts['candidate_mentions'] += 1
                    candidate_id = uid('candidate', run_id, prompt)
                    cref = source_ref(relative, document, pointer)
                    if candidate_id not in by_prompt:
                        candidate = {
                            'candidate_id': candidate_id, 'run_id': run_id, 'task_id': task_id,
                            'iteration': iteration, 'iterations_seen': [], 'prompt': prompt,
                            'parent_candidate_id': None, 'feedback': item.get('critique', item.get('feedback')),
                            'is_selected_final': False, 'in_final_pool': False,
                            'source_file': relative, 'source_records': [], 'candidate_sources': [],
                        }
                        by_prompt[candidate_id] = candidate
                        candidates[candidate_id] = candidate
                    candidate = by_prompt[candidate_id]
                    candidate['source_records'].append(cref)
                    if iteration is not None and iteration not in candidate['iterations_seen']:
                        candidate['iterations_seen'].append(iteration)
                    if candidate['iteration'] is None and iteration is not None:
                        candidate['iteration'] = iteration
                    candidate['is_selected_final'] |= selected
                    candidate['in_final_pool'] |= '/final_' in pointer
                    if item.get('source') and item['source'] not in candidate['candidate_sources']:
                        candidate['candidate_sources'].append(item['source'])
                    for split, key in (('dev', 'dev_score'), ('test', 'test_score')):
                        if key not in item or item[key] is None:
                            continue
                        score = item[key]
                        counts[split + '_score_mentions'] += 1
                        if not score_valid(score):
                            issues.append({**cref, 'candidate_id': candidate_id, 'reason': 'invalid_score', 'field': key,
                                           'value': str(score)})
                            continue
                        code = item.get('code')
                        summary = pointer.endswith('/best_prompt')
                        score_pointer = pointer.rsplit('/', 1)[0] + '/best_' + split if summary else pointer + '/' + key
                        code_pointer = pointer.rsplit('/', 1)[0] + '/best_code' if summary else pointer + '/code'
                        eref = {**cref, 'score_pointer': score_pointer,
                                'generated_code_pointer': code_pointer if code is not None else None}
                        registry = observation_registry[(candidate_id, split)]
                        matches = [e for e in registry if e['score'] == score and
                                   (code is None or e['generated_code'] is None or e['generated_code'] == code)]
                        if '/beam/' in pointer:
                            matches = [e for e in matches if e['observed_iteration'] == iteration]
                        # Reused elitist copies are matched by code. Actual newly generated
                        # occurrences remain observations, not additional candidates.
                        if matches and not primary:
                            evaluation = matches[-1]
                            if evaluation['generated_code'] is None and code is not None:
                                evaluation['generated_code'] = code
                            evaluation['source_records'].append(eref)
                            continue
                        evaluation_id = uid('evaluation', candidate_id, split, document, pointer, len(registry))
                        evaluation = {
                            'evaluation_id': evaluation_id, 'candidate_id': candidate_id,
                            'evaluation_split': split, 'score': score, 'passed_tests': None,
                            'total_tests': run['n_' + split], 'generated_code': code,
                            'execution_status': None, 'seed': None, 'test_suite_id': None,
                            'comparison_group_id': group, 'source_records': [eref],
                            'observed_iteration': iteration,
                        }
                        registry.append(evaluation)
                        evaluations[evaluation_id] = evaluation

                base = run['source_pointer']
                if method in ('zero_shot', 'human_prompt', 'baseline'):
                    prompt = problem if method == 'zero_shot' else body.get('prompt', body.get('enriched_prompt'))
                    observe({**body, 'prompt': prompt}, base, selected=True, primary=True)
                    continue
                # APE logs only newly generated proposals per round. APO logs full pools
                # without code, and beam copies with code. No parents or critiques persist.
                for i, iteration in enumerate(body.get('iterations', [])):
                    number = iteration.get('round', iteration.get('iter'))
                    schemas[method + '/iteration'].update(iteration)
                    primary_key = 'candidates' if method == 'APE' else 'all_candidates'
                    for j, item in enumerate(iteration.get(primary_key, [])):
                        schemas[method + '/' + primary_key].update(item)
                        observe(item, f'{base}/iterations/{i}/{primary_key}/{j}', number, primary=True)
                    if method == 'APO':
                        for j, item in enumerate(iteration.get('beam', [])):
                            schemas['APO/beam'].update(item)
                            observe(item, f'{base}/iterations/{i}/beam/{j}', number)
                pool_key = 'final_population' if method == 'APE' else 'final_pool'
                for j, item in enumerate(body.get(pool_key, [])):
                    observe(item, f'{base}/{pool_key}/{j}', 0 if item.get('source') == 'init' else None,
                            selected=item.get('prompt') == body.get('best_prompt'))
                if 'best_prompt' in body:
                    observe({'prompt': body['best_prompt'], 'code': body.get('best_code'),
                             'dev_score': body.get('best_dev'), 'test_score': body.get('best_test')},
                            base + '/best_prompt', selected=True)
    # Unscored curated best prompts are preserved without inventing model or score labels.
    problem_tasks = {task['problem']: tid for tid, task in tasks.items()}
    source_id_tasks = {task['source_id']: tid for tid, task in tasks.items() if task['source_id'] is not None}
    matching = defaultdict(list)
    for candidate in candidates.values():
        matching[(candidate['task_id'], candidate['prompt'])].append(candidate)
    for row, ref in archive:
        tid = problem_tasks.get(row['problem'])
        if tid is None and row.get('id') in source_id_tasks:
            possible = source_id_tasks[row['id']]
            if tasks[possible]['problem'].strip() == row['problem'].strip():
                tid = possible
                issues.append({**ref, 'reason': 'archive_problem_outer_whitespace_variant',
                               'task_id': tid, 'archive_id': row.get('id')})
        if tid is None:
            issues.append({**ref, 'reason': 'archive_problem_not_matched', 'archive_id': row.get('id')})
            continue
        run_id = uid('run', ref, tid, 'unscored_archive')
        for index, prompt in enumerate(row['prompts']):
            counts['archive_prompt_mentions'] += 1
            pref = {**ref, 'source_pointer': ref['source_pointer'] + f'/prompts/{index}'}
            if not isinstance(prompt, str):
                issues.append({**pref, 'reason': 'missing_archive_prompt'})
                continue
            # The archive lacks run identity: exact-text links are matches, not asserted
            # originating run IDs. Already represented prompts need no phantom runs.
            if (tid, prompt) in matching:
                for candidate in matching[(tid, prompt)]:
                    candidate.setdefault('unscored_archive_matches', []).append(pref)
                counts['archive_matched_prompt_mentions'] += 1
                continue
            runs.setdefault(run_id, {
                'run_id': run_id, 'task_id': tid, 'optimizer_method': 'unscored_archive',
                'optimizer_model': None, 'target_model': None, 'model_quantization': None,
                'experiment_version': None, 'decoding_parameters': None, 'seed': None,
                'comparison_group_id': None, 'comparison_evidence': None,
                'configuration_provenance': 'unscored curated archive; originating run unknown',
                'n_dev': None, 'n_test': None, **ref,
            })
            cid = uid('candidate', run_id, prompt)
            if cid not in candidates:
                candidates[cid] = {
                    'candidate_id': cid, 'run_id': run_id, 'task_id': tid, 'iteration': None,
                    'iterations_seen': [], 'prompt': prompt, 'parent_candidate_id': None, 'feedback': None,
                    'is_selected_final': False, 'in_final_pool': False,
                    'source_file': ref['source_file'], 'source_records': [], 'candidate_sources': [],
                }
            candidates[cid]['source_records'].append(pref)
    return {'tasks': tasks, 'runs': runs, 'candidates': candidates, 'evaluations': evaluations,
            'inventory': inventory, 'source_issues': issues, 'source_counts': dict(counts),
            'schemas': {k: sorted(v) for k, v in schemas.items()}}


def quality_reasons(prompt, problem):
    reasons = []
    if not prompt.strip() or '\x00' in prompt:
        reasons.append('empty_or_malformed_prompt')
    if problem not in prompt:
        reasons.append('original_problem_not_preserved_verbatim')
    # Only scan additions: original HumanEval signatures/examples are legitimate.
    extra = prompt.replace(problem, '', 1) if problem in prompt else prompt
    if re.search(r'(?m)^\s*(?:def|class)\s+\w+.*:|```(?:python|py)\s*\n[\s\S]*?\b(?:return|print|for|while)\b', extra):
        reasons.append('possible_complete_solution')
    if re.search(r'(?i)(?:hidden|held[- ]out|dev(?:elopment)?|unit)\s+tests?\s*(?:[:=]|failed|failure)|'
                 r'expected\s*[:=].{0,100}actual\s*[:=]|(?:critique|gradient|failure_str)\s*:', extra):
        reasons.append('possible_evaluation_feedback_leak')
    if re.search(r'(?i)(?:ignore|disregard|change|relax|override)\s+(?:the\s+)?(?:original\s+)?'
                 r'(?:constraints?|requirements?|problem)|instead\s+of\s+(?:the\s+)?(?:required|specified)', extra):
        reasons.append('possible_requirement_override')
    if prompt == problem:
        reasons.append('unoptimized_original_problem')
    return reasons


def split_tasks(tasks, seed, ratios):
    """Integer allocation preserves global counts and stratifies each benchmark/difficulty."""
    labels = ('train', 'validation', 'test')
    total = len(tasks)
    exact = [total * value / sum(ratios) for value in ratios]
    desired = [math.floor(value) for value in exact]
    for i in sorted(range(3), key=lambda i: (-(exact[i] - desired[i]), i))[:total - sum(desired)]:
        desired[i] += 1
    strata = defaultdict(list)
    for tid, task in sorted(tasks.items()):
        strata[(task['benchmark'], task['difficulty'] or 'unknown')].append(tid)
    allocation = {key: [math.floor(len(ids) * value / total) if total else 0 for value in desired]
                  for key, ids in strata.items()}
    remaining = [desired[i] - sum(counts[i] for counts in allocation.values()) for i in range(3)]
    for key in sorted(strata):
        while sum(allocation[key]) < len(strata[key]):
            choices = [i for i in range(3) if remaining[i] > 0]
            i = max(choices, key=lambda i: (len(strata[key]) * desired[i] / total - allocation[key][i],
                                             remaining[i], -i))
            allocation[key][i] += 1
            remaining[i] -= 1
    result = {label: [] for label in labels}
    rng = random.Random(seed)
    for key in sorted(strata):
        ids = list(strata[key])
        rng.shuffle(ids)
        offset = 0
        for i, label in enumerate(labels):
            count = allocation[key][i]
            result[label].extend(ids[offset:offset + count])
            offset += count
    return {label: sorted(ids) for label, ids in result.items()}


def choose_diverse_pairs(pairs, limit):
    """Prefer unused endpoints, then large margins, with deterministic tie breaks."""
    selected, usage, seen_text = [], Counter(), set()
    while pairs and len(selected) < limit:
        pairs.sort(key=lambda p: (usage[p[0]['candidate_id']] + usage[p[1]['candidate_id']],
                                  -abs(p[0]['score'] - p[1]['score']),
                                  p[0]['candidate_id'], p[1]['candidate_id']))
        chosen, rejected = pairs.pop(0)
        text_pair = (chosen['prompt'], rejected['prompt'])
        if text_pair in seen_text:
            continue
        seen_text.add(text_pair)
        selected.append((chosen, rejected))
        usage[chosen['candidate_id']] += 1
        usage[rejected['candidate_id']] += 1
    return selected


def training_data(data, splits, margin=.20, max_pairs=5, dedup='exact', quality='strict'):
    membership = {tid: label for label, ids in splits.items() for tid in ids}
    dev = defaultdict(list)
    for evaluation in data['evaluations'].values():
        if evaluation['evaluation_split'] == 'dev':
            dev[evaluation['candidate_id']].append(evaluation)
    pools, decisions = defaultdict(list), []
    group_dev_counts = defaultdict(set)
    for run in data['runs'].values():
        if run['optimizer_method'] in ('APE', 'APO'):
            group_dev_counts[run['comparison_group_id']].add(run['n_dev'])
    for cid, candidate in sorted(data['candidates'].items()):
        run = data['runs'][candidate['run_id']]
        task = data['tasks'][candidate['task_id']]
        reasons = quality_reasons(candidate['prompt'], task['problem'])
        structural = {'empty_or_malformed_prompt', 'original_problem_not_preserved_verbatim', 'unoptimized_original_problem'}
        blocked = reasons if quality == 'strict' else [r for r in reasons if r in structural]
        if not dev[cid]:
            blocked = blocked + ['missing_dev_score']
        if not run['target_model']:
            blocked = blocked + ['unknown_target_model']
        if run['optimizer_method'] not in ('APE', 'APO'):
            blocked = blocked + ['not_observed_optimized_method']
        if dev[cid] and len({e['score'] for e in dev[cid]}) > 1:
            blocked = blocked + ['conflicting_repeated_dev_scores']
        if dev[cid] and (not isinstance(run['n_dev'], int) or run['n_dev'] <= 0):
            blocked = blocked + ['missing_dev_test_count']
        if len(group_dev_counts[run['comparison_group_id']]) > 1:
            blocked = blocked + ['inconsistent_dev_counts_within_record']
        if dev[cid] and len({e['score'] for e in dev[cid]}) == 1 and dev[cid][0]['score'] <= 0:
            # Zero-scoring prompts can be DPO rejected examples, never SFT labels.
            reasons.append('zero_dev_score_not_sft_label')
        if blocked or reasons:
            decisions.append({'candidate_id': cid, 'task_id': candidate['task_id'],
                              'reasons': sorted(set(blocked + reasons)), 'excluded_from_training': bool(blocked),
                              'split': membership[candidate['task_id']],
                              'source_records': candidate['source_records']})
        if blocked:
            continue
        observed = dev[cid][0]
        group = (candidate['task_id'], run['target_model'], run['comparison_group_id'])
        pools[group].append({'candidate_id': cid, 'prompt': candidate['prompt'], 'score': observed['score'],
                             'evaluation_ids': sorted(e['evaluation_id'] for e in dev[cid]),
                             'comparison_group_id': run['comparison_group_id'],
                             'quality_flags': reasons})
    sft, dpo, group_reports = [], [], []
    pairs_by_task_model = defaultdict(list)
    for (tid, model, group), pool in sorted(pools.items()):
        # Never rank different source-file dev suites solely because sizes match.
        pool.sort(key=lambda c: (len(c['prompt']), c['candidate_id']))
        prompt_scores = defaultdict(set)
        for candidate in pool:
            prompt_key = candidate['prompt'] if dedup == 'exact' else ' '.join(candidate['prompt'].split())
            prompt_scores[prompt_key].add(candidate['score'])
        unique, seen = [], set()
        for candidate in pool:
            prompt_key = candidate['prompt'] if dedup == 'exact' else ' '.join(candidate['prompt'].split())
            if len(prompt_scores[prompt_key]) > 1:
                decisions.append({'candidate_id': candidate['candidate_id'], 'task_id': tid,
                                  'reasons': ['conflicting_prompt_dev_scores_across_runs'], 'excluded_from_training': True,
                                  'split': membership[tid], 'source_records': data['candidates'][candidate['candidate_id']]['source_records']})
                continue
            if prompt_key in seen:
                decisions.append({'candidate_id': candidate['candidate_id'], 'task_id': tid,
                                  'reasons': ['duplicate_prompt_in_comparable_pool'], 'excluded_from_training': True,
                                  'split': membership[tid], 'source_records': data['candidates'][candidate['candidate_id']]['source_records']})
            else:
                unique.append(candidate)
                seen.add(prompt_key)
        group_reports.append({'task_id': tid, 'target_model': model, 'comparison_group_id': group,
                              'eligible_candidates': len(unique), 'split': membership[tid]})
        if membership[tid] == 'test':
            # Internal test labels are never selected or paired.
            continue
        positive = [candidate for candidate in unique if candidate['score'] > 0]
        if positive:
            best = min(positive, key=lambda c: (-c['score'], len(c['prompt']), c['candidate_id']))
            sft.append({
                'sample_id': uid('sft', tid, model, group, best['candidate_id']),
                'task_id': tid, 'target_model': model, 'instruction': INSTRUCTION,
                'input': data['tasks'][tid]['problem'], 'output': best['prompt'],
                'source_candidate_id': best['candidate_id'], 'label_dev_score': best['score'],
                'label_evaluation_ids': best['evaluation_ids'], 'comparison_group_id': group,
                'quality_flags': best['quality_flags'], 'split': membership[tid],
            })
        for i, left in enumerate(unique):
            for right in unique[i + 1:]:
                chosen, rejected = (left, right) if left['score'] > right['score'] else (right, left)
                if chosen['prompt'] == rejected['prompt'] or chosen['score'] <= rejected['score']:
                    continue
                if chosen['score'] - rejected['score'] + 1e-12 < margin:
                    continue
                pairs_by_task_model[(tid, model)].append((chosen, rejected))
    # Multiple unprovably equivalent suites get separate SFT labels; exact same
    # (task, target, output) labels are retained once, without ranking across suites.
    sft_unique, seen_sft = [], set()
    for sample in sorted(sft, key=lambda s: s['sample_id']):
        key = (sample['task_id'], sample['target_model'], sample['output'])
        if key in seen_sft:
            decisions.append({'sample_id': sample['sample_id'], 'candidate_id': sample['source_candidate_id'],
                              'task_id': sample['task_id'], 'reasons': ['duplicate_sft_output_across_conditions'],
                              'excluded_from_training': True, 'split': sample['split']})
        else:
            seen_sft.add(key)
            sft_unique.append(sample)
    for (tid, model), pairs in sorted(pairs_by_task_model.items()):
        for chosen, rejected in choose_diverse_pairs(pairs, max_pairs):
            dpo.append({
                'sample_id': uid('dpo', tid, model, chosen['candidate_id'], rejected['candidate_id']),
                'task_id': tid, 'target_model': model, 'instruction': INSTRUCTION,
                'input': data['tasks'][tid]['problem'], 'chosen': chosen['prompt'], 'rejected': rejected['prompt'],
                'chosen_candidate_id': chosen['candidate_id'], 'rejected_candidate_id': rejected['candidate_id'],
                'chosen_dev_score': chosen['score'], 'rejected_dev_score': rejected['score'],
                'chosen_evaluation_ids': chosen['evaluation_ids'], 'rejected_evaluation_ids': rejected['evaluation_ids'],
                'score_margin': chosen['score'] - rejected['score'],
                'comparison_group_id': chosen['comparison_group_id'], 'split': membership[tid],
                'quality_flags': sorted(set(chosen['quality_flags'] + rejected['quality_flags'])),
            })
    return sft_unique, sorted(dpo, key=lambda s: s['sample_id']), decisions, group_reports


def validate(data, splits, sft, dpo, margin=.20, max_pairs=5, quality='strict'):
    errors, checks = [], Counter()

    def check(condition, message):
        checks['assertions'] += 1
        if not condition:
            errors.append(message)

    tasks, runs, candidates, evaluations = (data[key] for key in ('tasks', 'runs', 'candidates', 'evaluations'))
    membership = {}
    for label, tids in splits.items():
        for tid in tids:
            check(tid not in membership, f'duplicate task split: {tid}')
            check(tid in tasks, f'unknown split task: {tid}')
            membership[tid] = label
    check(set(membership) == set(tasks), 'split task coverage')
    for rid, run in runs.items():
        check(run['task_id'] in tasks, f'run task reference: {rid}')
        check(bool(run['source_file']), f'run source provenance: {rid}')
    for cid, candidate in candidates.items():
        check(candidate['run_id'] in runs and candidate['task_id'] in tasks, f'candidate reference: {cid}')
        check(runs[candidate['run_id']]['task_id'] == candidate['task_id'], f'candidate/run task mismatch: {cid}')
        check(bool(candidate['source_records']), f'candidate provenance: {cid}')
        # Empty raw observations remain in canonical data, but must be reported.
        if not candidate['prompt'].strip() or '\x00' in candidate['prompt']:
            check(any(x.get('candidate_id') == cid and 'empty_or_malformed_prompt' in x['reasons']
                      for x in data.get('filtering', [])), f'unreported malformed raw candidate: {cid}')
    for eid, evaluation in evaluations.items():
        check(evaluation['candidate_id'] in candidates, f'evaluation reference: {eid}')
        check(score_valid(evaluation['score']), f'score out of bounds: {eid}')
        check(evaluation['evaluation_split'] in ('dev', 'test'), f'invalid evaluation split: {eid}')
        check(bool(evaluation['source_records']), f'evaluation provenance: {eid}')

    def verify_label(sample, cid, prompt, score, ids):
        check(cid in candidates, f'label candidate missing: {cid}')
        if cid not in candidates:
            return
        candidate = candidates[cid]
        run = runs[candidate['run_id']]
        check(candidate['task_id'] == sample['task_id'], f'label task mismatch: {cid}')
        check(candidate['prompt'] == prompt, f'label prompt mismatch: {cid}')
        check(run['target_model'] == sample['target_model'], f'label target mismatch: {cid}')
        check(run['comparison_group_id'] == sample['comparison_group_id'], f'label conditions mismatch: {cid}')
        check(bool(ids), f'missing label evaluation IDs: {cid}')
        for eid in ids:
            check(eid in evaluations, f'label observation absent: {eid}')
            if eid in evaluations:
                e = evaluations[eid]
                check(e['evaluation_split'] == 'dev' and e['candidate_id'] == cid and e['score'] == score,
                      f'non-dev or incorrect label evidence: {eid}')
                check(e['comparison_group_id'] == sample['comparison_group_id'], f'noncomparable evaluation: {eid}')
        reasons = quality_reasons(prompt, tasks[sample['task_id']]['problem'])
        if quality == 'strict':
            check(not reasons, f'invalid training prompt: {cid}: {reasons}')
        else:
            check(not set(reasons) & {'empty_or_malformed_prompt', 'original_problem_not_preserved_verbatim',
                                     'unoptimized_original_problem'}, f'incomplete training prompt: {cid}')

    for rows, kind in ((sft, 'sft'), (dpo, 'dpo')):
        seen = set()
        for sample in rows:
            check(sample['sample_id'] not in seen, f'duplicate {kind} sample')
            seen.add(sample['sample_id'])
            check(sample['split'] in ('train', 'validation') and membership[sample['task_id']] == sample['split'],
                  f'leaked internal test task: {sample["sample_id"]}')
            check(sample['instruction'] == INSTRUCTION and sample['input'] == tasks[sample['task_id']]['problem'],
                  f'inconsistent training input: {sample["sample_id"]}')
            if kind == 'sft':
                verify_label(sample, sample['source_candidate_id'], sample['output'], sample['label_dev_score'],
                             sample['label_evaluation_ids'])
                check(sample['label_dev_score'] > 0, 'zero-scoring SFT label')
            else:
                for prefix in ('chosen', 'rejected'):
                    verify_label(sample, sample[prefix + '_candidate_id'], sample[prefix],
                                 sample[prefix + '_dev_score'], sample[prefix + '_evaluation_ids'])
                gap = sample['chosen_dev_score'] - sample['rejected_dev_score']
                check(gap > 0 and gap + 1e-12 >= margin and abs(gap - sample['score_margin']) < 1e-12,
                      'invalid preference margin')
                check(sample['chosen'] != sample['rejected'], 'identical preference prompts')
    check(len({(s['task_id'], s['target_model'], s['output']) for s in sft}) == len(sft), 'duplicate SFT text')
    check(len({(s['task_id'], s['target_model'], s['chosen'], s['rejected']) for s in dpo}) == len(dpo),
          'duplicate DPO text')
    pair_counts = Counter((s['task_id'], s['target_model']) for s in dpo)
    check(all(n <= max_pairs for n in pair_counts.values()), 'DPO pair cap exceeded')
    return {'passed': not errors, 'assertions': checks['assertions'], 'errors': errors,
            'raw_invalid_prompt_candidate_ids': sorted(c['candidate_id'] for c in candidates.values()
                                                     if not c['prompt'].strip() or '\x00' in c['prompt']),
            'scope': 'valid training prompts and canonical references/provenance; raw invalid observations retained and flagged; semantic equivalence not proved'}


def distribution(values):
    if not values:
        return {'count': 0, 'min': None, 'max': None, 'mean': None, 'median': None, 'histogram': {}}
    return {'count': len(values), 'min': min(values), 'max': max(values),
            'mean': statistics.mean(values), 'median': statistics.median(values),
            'histogram': dict(sorted(Counter(format(x, '.6g') for x in values).items()))}


def stats(data, splits, sft, dpo, decisions, groups):
    candidates, evaluations, runs, tasks = (list(data[k].values()) for k in ('candidates', 'evaluations', 'runs', 'tasks'))
    dev = [e['score'] for e in evaluations if e['evaluation_split'] == 'dev']
    runmap = data['runs']
    improvements = set()
    by_group = defaultdict(list)
    for e in evaluations:
        if e['evaluation_split'] == 'dev':
            c = data['candidates'][e['candidate_id']]
            by_group[(c['task_id'], runmap[c['run_id']]['comparison_group_id'])].append(e['score'])
    for (tid, _), scores in by_group.items():
        if max(scores) > min(scores):
            improvements.add(tid)
    # This is within-search spread, never a claim over zero-shot: baselines lack dev scores.
    filtering_counts = Counter(reason for d in decisions for reason in d['reasons'])
    removed = {d['candidate_id'] for d in decisions if d.get('excluded_from_training') and d.get('candidate_id')}
    counts = {
        'tasks': len(tasks), 'tasks_by_benchmark': dict(Counter(t['benchmark'] for t in tasks)),
        'tasks_by_difficulty': dict(Counter((t['benchmark'] + '/' + (t['difficulty'] or 'unknown')) for t in tasks)),
        'task_configurations': data['source_counts'].get('task_configurations', 0),
        'experimental_runs': sum(r['optimizer_method'] != 'unscored_archive' for r in runs),
        'unscored_archive_runs': sum(r['optimizer_method'] == 'unscored_archive' for r in runs),
        'optimizer_target_configurations': len({(r['optimizer_model'], r['target_model']) for r in runs if r['target_model'] and r['optimizer_model']}),
        'candidates': len(candidates), 'candidate_mentions': data['source_counts'].get('candidate_mentions', 0),
        'candidate_prompts_by_method': dict(Counter(runmap[c['run_id']]['optimizer_method'] for c in candidates)),
        'unique_prompt_texts': len({c['prompt'] for c in candidates}),
        'unique_task_prompt_texts': len({(c['task_id'], c['prompt']) for c in candidates}),
        'dev_evaluations': len(dev), 'test_evaluations': sum(e['evaluation_split'] == 'test' for e in evaluations),
        'evaluations_with_generated_code': sum(e['generated_code'] is not None for e in evaluations),
        'distinct_candidate_code_completions': len({(e['candidate_id'], e['generated_code']) for e in evaluations if e['generated_code'] is not None}),
        'sft_examples': len(sft), 'dpo_pairs': len(dpo),
        'sft_by_split': dict(Counter(s['split'] for s in sft)), 'dpo_by_split': dict(Counter(s['split'] for s in dpo)),
        'development_scores': distribution(dev), 'preference_margins': distribution([s['score_margin'] for s in dpo]),
        'prompt_length_characters': distribution([len(c['prompt']) for c in candidates]),
        'prompt_length_words': distribution([len(c['prompt'].split()) for c in candidates]),
        'tasks_with_positive_observed_within_search_spread': len(improvements),
        'percentage_tasks_with_positive_observed_within_search_spread': 100 * len(improvements) / len(tasks) if tasks else 0,
        'positive_improvement_over_zero_shot': None,
        'improvement_note': 'Zero-shot/human/baseline have test scores only. Dev improvement over them cannot be measured; within-search spread is max-minus-min in the same comparable group, not causal improvement.',
        'unique_candidates_excluded_from_training': len(removed),
        'filter_rule_counts_nonexclusive': dict(filtering_counts),
        'missing_data': {
            'candidate_parent_ids': sum(c['parent_candidate_id'] is None for c in candidates),
            'candidate_feedback': sum(c['feedback'] is None for c in candidates),
            'candidates_without_dev': len({c['candidate_id'] for c in candidates} - {e['candidate_id'] for e in evaluations if e['evaluation_split'] == 'dev'}),
            'evaluations_without_generated_code': sum(e['generated_code'] is None for e in evaluations),
            'evaluations_without_suite_id': sum(e['test_suite_id'] is None for e in evaluations),
            'evaluations_without_passed_tests': sum(e['passed_tests'] is None for e in evaluations),
            'evaluations_without_execution_status': sum(e['execution_status'] is None for e in evaluations),
            'runs_without_seed': sum(r['seed'] is None for r in runs),
            'runs_without_decoding_parameters': sum(r['decoding_parameters'] is None for r in runs),
            'runs_without_exact_model_quantization': sum(r['model_quantization'] is None for r in runs),
            'tasks_without_original_benchmark_source_id': sum(t['source_id'] is None for t in tasks),
            'tasks_without_interface': sum(t['interface'] is None for t in tasks),
        },
        'splits': {label: {'tasks': len(ids), 'benchmarks': dict(Counter(data['tasks'][tid]['benchmark'] for tid in ids)),
                           'strata': dict(Counter(data['tasks'][tid]['benchmark'] + '/' + (data['tasks'][tid]['difficulty'] or 'unknown') for tid in ids))}
                   for label, ids in splits.items()},
        'comparable_groups': len(groups), 'source_counts': data['source_counts'],
    }
    return counts


def reports(output, data, statistics_data, validation):
    # Reports live alongside the canonical files so custom builds remain self-contained.
    audit = ['# CodePromptOpt data audit', '',
             'Source files below are the only experimental data inputs. Original experiments and algorithms are unchanged.', '',
             '| Source | Documents | Rows | SHA-256 |', '|---|---:|---:|---|']
    audit += [f"| {s['source_file']} | {s['documents']} | {s['rows']} | {s['sha256']} |" for s in data['inventory']]
    audit += ['', '## Counts', '', f"- Unique tasks: {statistics_data['tasks']}.",
              f"- Task/configuration records: {statistics_data['task_configurations']}.",
              f"- Experimental method runs: {statistics_data['experimental_runs']}.",
              f"- Canonical candidates: {statistics_data['candidates']}; distinct exact prompt strings: {statistics_data['unique_prompt_texts']}.",
              f"- Observed dev evaluations: {statistics_data['dev_evaluations']}; test evaluations: {statistics_data['test_evaluations']}.",
              f"- Distinct (candidate, code) completions: {statistics_data['distinct_candidate_code_completions']}.",
              '', '## Verified mapping and limitations', '',
              '`question` is preserved byte-for-byte as decoded Unicode text. HumanEval source IDs are saved; APPS problem_idx is a local sampling index, not an APPS dataset ID. Task IDs hash benchmark and original problem.',
              '', 'APE iterations contain new proposals with code and dev_score. Final populations repeat surviving proposals and add test_score; best summaries repeat the selected observation. APO all_candidates preserves prompts/dev scores/source kind, beam preserves code for selected members, and final_pool preserves code/test scores for the final pool. Intermediate discarded APO completions and initial APO prompts from searches that continued are not fully logged.',
              '', 'Candidates are exact-text deduplicated within a method run. Primary iteration occurrences are separate observed evaluations; beam, final, and best copies merge into compatible existing observations. Missing code is enriched only by a matching copied observation. Counts do not treat saved aliases as independent evaluations.',
              '', 'No parent IDs or persisted critiques exist in observed candidate schemas; source=gradient/paraphrase does not identify parents. Lineage cannot be reconstructed reliably. All parent_candidate_id and unavailable feedback remain null.',
              '', 'Comparable dev pools combine APE and APO only within the same stored task/configuration record. main.py passes the same dev_io to both methods, and both call evaluate_code. Exact test_suite_id is unavailable. Equal test counts across different files do not establish identical suites; cross-file rankings and pairs are prohibited.',
              '', 'SFT emits one best positive-scoring observed prompt per task/target/comparison group. Multiple groups for the same target remain separate; duplicate identical outputs are retained once using sample_id order, never a cross-suite score ranking.',
              '', 'Filename model labels identify the target first and optimizer second. qwen3.5-coder is the filename label, not a verified Ollama model identifier. Exact quantization, generation seeds, decoding settings, and historical executable versions were not persisted in these source files and are null. Current code defaults are not asserted as historical runtime settings.',
              '', 'Zero-shot, human prompt and baseline have held-out test scores only and cannot supply SFT labels or DPO preferences. best_prompts.json is an unscored archive: exact task/prompt matches link to existing candidates. HumanEval archive problems may differ only by outer whitespace; a matching benchmark task ID plus stripped-text equality establishes identity, while the canonical original problem remains unchanged. Unmatched prompts remain in archival runs with unknown target and no fabricated evaluations.',
              '', 'Concatenated JSON documents are explicitly parsed; identical repeated records in one source file are collapsed and reported in source_issues.jsonl. Nonidentical repeats remain separate conditions.',
              '', 'Quality filtering requires the exact full original problem to occur in the prompt. Paraphrases may be semantically correct but cannot be certified automatically and are excluded rather than repaired. Additional code, evaluation feedback and requirement overrides are screened heuristically; passing is not proof of semantic equivalence. Manual review remains necessary.',
              '', 'All unavailable fields use null. passed_tests is not reconstructed by multiplying rounded scores. Suite contents and reference solutions are never added to training inputs. Language is python as established by the experimental generator/evaluator. APPS interfaces cannot be recovered reliably and remain null.',
              '', '## Actual source schemas', '', '```json', json.dumps(data['schemas'], indent=2, sort_keys=True), '```',
              '', '## Missing data', '', '```json', json.dumps(statistics_data['missing_data'], indent=2, sort_keys=True), '```']
    (output / 'DATA_AUDIT.md').write_text('\n'.join(audit) + '\n', encoding='utf-8')
    lines = ['# CodePromptOpt dataset statistics', '',
             f"Tasks: **{statistics_data['tasks']}**. Candidates: **{statistics_data['candidates']}**. SFT examples: **{statistics_data['sft_examples']}**. DPO pairs: **{statistics_data['dpo_pairs']}**.",
             '', f"Validation passed: **{validation['passed']}** ({validation['assertions']} assertions).",
             '', 'Filtering counts overlap; they are candidate decisions, not synthetic preference counts. Canonical raw observations are retained. Test tasks have no emitted SFT/DPO labels.',
             '', '```json', json.dumps(statistics_data, ensure_ascii=False, sort_keys=True, indent=2), '```']
    (output / 'DATASET_STATISTICS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def build(args):
    input_dir, output = Path(args.input_dir).resolve(), Path(args.output_dir).resolve()
    if input_dir == output or input_dir in output.parents or output in input_dir.parents:
        raise ValueError('Input and output directories must not overlap.')
    data = extract(input_dir)
    splits = split_tasks(data['tasks'], args.seed, args.split_ratios)
    membership = {tid: label for label, ids in splits.items() for tid in ids}
    sft, dpo, decisions, groups = training_data(data, splits, args.min_margin, args.max_pairs, args.deduplication, args.quality_filtering)
    data['filtering'] = decisions
    result = validate(data, splits, sft, dpo, args.min_margin, args.max_pairs, args.quality_filtering)
    if not result['passed']:
        output.mkdir(parents=True, exist_ok=True)
        dump(output / 'validation.json', result)
        raise ValueError(f'Dataset validation failed: {result["errors"][:10]}')
    settings = {'pipeline_version': '1.0.0', 'seed': args.seed, 'split_ratios': list(args.split_ratios),
                'min_dpo_score_margin': args.min_margin, 'max_dpo_pairs_per_task_target': args.max_pairs,
                'prompt_deduplication': args.deduplication, 'quality_filtering': args.quality_filtering,
                'input_sources': data['inventory'], 'comparison_policy': 'same stored task record only',
                'sft_zero_score_labels': False}
    for key in ('tasks', 'runs', 'candidates', 'evaluations'):
        records = sorted(data[key].values(), key=lambda r: r[key[:-1] + '_id' if key != 'evaluations' else 'evaluation_id'])
        for record in records:
            tid = record.get('task_id')
            if tid is None:
                tid = data['candidates'][record['candidate_id']]['task_id']
            record['split'] = membership[tid]
        jsonl(output / f'{key}.jsonl', records)
    for label in ('train', 'validation'):
        jsonl(output / 'sft' / f'{label}.jsonl', [s for s in sft if s['split'] == label])
        jsonl(output / 'dpo' / f'{label}.jsonl', [s for s in dpo if s['split'] == label])
    jsonl(output / 'filtering_report.jsonl', decisions)
    jsonl(output / 'source_issues.jsonl', data['source_issues'])
    dump(output / 'splits.json', {'seed': args.seed, 'ratios': list(args.split_ratios), **splits})
    dump(output / 'build_manifest.json', settings)
    dump(output / 'comparable_groups.json', groups)
    dump(output / 'validation.json', result)
    statistics_data = stats(data, splits, sft, dpo, decisions, groups)
    dump(output / 'statistics.json', statistics_data)
    reports(output, data, statistics_data, result)
    if output.parent == Path(__file__).resolve().parent:
        for name in ('DATA_AUDIT.md', 'DATASET_STATISTICS.md'):
            (output.parent / name).write_bytes((output / name).read_bytes())
    artifacts = sorted(p for p in output.rglob('*') if p.is_file() and p.name != 'artifact_hashes.json')
    dump(output / 'artifact_hashes.json', {p.relative_to(output).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in artifacts})
    return statistics_data


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir', default='dataset_code_prompt')
    p.add_argument('--output-dir', default='CodePromptOpt/data')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--split-ratios', type=float, nargs=3, default=(70, 10, 10), metavar=('TRAIN', 'VALIDATION', 'TEST'))
    p.add_argument('--min-margin', type=float, default=.20)
    p.add_argument('--max-pairs', type=int, default=5)
    p.add_argument('--deduplication', choices=('exact', 'whitespace'), default='exact',
                   help='Canonical prompts always use exact text; optional whitespace dedup applies to training pools only.')
    p.add_argument('--quality-filtering', choices=('strict', 'flag'), default='strict',
                   help='flag retains heuristic warnings; complete original problem and dev comparability remain mandatory.')
    return p


def main():
    p = parser()
    args = p.parse_args()
    if any(x < 0 or not math.isfinite(x) for x in args.split_ratios) or sum(args.split_ratios) <= 0:
        p.error('Split ratios must be finite, nonnegative and sum to a positive number.')
    if not math.isfinite(args.min_margin) or not 0 <= args.min_margin <= 1 or args.max_pairs < 0:
        p.error('Margin must be in [0, 1]; max-pairs must be nonnegative.')
    result = build(args)
    print(json.dumps({k: result[k] for k in ('tasks', 'experimental_runs', 'candidates', 'sft_examples', 'dpo_pairs', 'sft_by_split', 'dpo_by_split')}, indent=2))


if __name__ == '__main__':
    main()
