"""Read stored Garg experiment results in the Jupyter analysis environment."""
import hashlib
import json

from icl_lab.runtime import ROOT, project_path


def read_json(path, default=None):
    path = project_path(path, 'JSON input')
    return json.loads(path.read_text()) if path.is_file() else default


def read_jsonl(path):
    path = project_path(path, 'JSONL input')
    if not path.is_file():
        return []
    text = path.read_text()
    rows, lines = [], text.splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index == len(lines) - 1 and not text.endswith('\n'):
                print('Последняя строка ещё дописывается:', path.name)
            else:
                raise
    return rows


def sha256_file(path):
    path = project_path(path, 'hash input', must_exist=True)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def current_code_hash():
    # Тот же порядок и формат, что storage.code_hash; torch не импортируется.
    files = []
    def visit(directory):
        directory = project_path(directory, 'source directory', must_exist=True)
        for candidate in sorted(directory.iterdir()):
            if candidate.name == '__pycache__':
                continue
            path = project_path(candidate, 'source input')
            if path.is_dir():
                visit(path)
            elif path.is_file():
                files.append(path)
    visit(ROOT / 'src')
    content = ''.join(str(p.relative_to(ROOT)) + sha256_file(p) for p in sorted(files))
    return hashlib.sha256(content.encode()).hexdigest()


def validation_artifacts(run):
    base = project_path(run / 'evaluations', 'evaluation directory')
    if not base.is_dir():
        return []
    paths = []
    for candidate in base.iterdir():
        # Другие split не открываем, даже если они когда-либо появятся рядом.
        if not candidate.name.startswith('validation-'):
            continue
        directory = project_path(candidate, 'validation evaluation directory')
        if directory.is_dir():
            path = project_path(directory / 'summary.json', 'validation summary')
            if path.is_file():
                paths.append(path)
    return sorted(paths, key=lambda p: (p.stat().st_mtime_ns, str(p)))
