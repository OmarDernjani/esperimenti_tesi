"""Bounded diagnostics for execution INSIDE the Podman test sandbox only."""
import json
from pathlib import Path
import runpy
import subprocess
import sys
import threading
import time

LIMIT = 32768


def classify(returncode, stderr):
    if returncode == 0:
        return 'passed'
    for marker, label in (('SyntaxError', 'syntax_error'), ('IndentationError', 'syntax_error'),
                          ('TimeoutExpired', 'inner_timeout'), ('MemoryError', 'memory_error'),
                          ('ModuleNotFoundError', 'missing_import'), ('AttributeError', 'interface_or_attribute_error'),
                          ('NameError', 'undefined_name'), ('AssertionError', 'assertion_failed')):
        if marker in stderr:
            return label
    return 'runtime_error'


def main():
    if sys.argv[1] == '--worker':
        # Existing APPS harnesses capture child stderr before raising. Surface that
        # detail without changing candidate code, assertions or expected values.
        original_run = subprocess.run

        def traced_run(*args, **kwargs):
            try:
                return original_run(*args, **kwargs)
            except subprocess.CalledProcessError as error:
                message = error.stderr or ''
                if isinstance(message, bytes):
                    message = message.decode(errors='replace')
                sys.stderr.write('\nAPPS CHILD STDERR:\n' + message[-LIMIT:] + '\n')
                raise

        subprocess.run = traced_run
        sys.path.insert(0, str(Path.cwd()))
        script = sys.argv[2]
        sys.argv = [script]
        runpy.run_path(script, run_name='__main__')
        return
    start = time.perf_counter()
    process = subprocess.Popen([sys.executable, '-B', __file__, '--worker', sys.argv[1]],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    captures = {}

    def drain(stream, name):
        tail, total = b'', 0
        while True:
            block = stream.read(8192)
            if not block:
                break
            total += len(block)
            tail = (tail + block)[-LIMIT:]
        stream.close()
        captures[name] = {'tail': tail.decode(errors='replace'), 'bytes': total, 'truncated': total > LIMIT}

    threads = [threading.Thread(target=drain, args=(stream, name), daemon=True)
               for stream, name in ((process.stdout, 'stdout'), (process.stderr, 'stderr'))]
    for thread in threads:
        thread.start()
    code = process.wait()
    for thread in threads:
        thread.join()
    print(json.dumps({'child_returncode': code, 'category': classify(code, captures['stderr']['tail']),
                      'elapsed_seconds': time.perf_counter() - start, **captures}), flush=True)
    raise SystemExit(code if 0 <= code <= 255 else 128 - code)


if __name__ == '__main__':
    main()
