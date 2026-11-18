"""Conservative local include closure; all preprocessor branches are retained."""
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
ROOTS = ('flash_fwd_launch_template.h', 'flash_prepare_scheduler.cu')
INCLUDE = re.compile(r'^\s*#\s*include\s*[<"]([^">]+)[">]', re.M)


def audit(directory):
    all_files = {p.name for p in directory.iterdir() if p.is_file()}
    seen = set()
    pending = list(ROOTS)
    missing = []
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        path = directory / name
        if not path.is_file():
            missing.append(name)
            continue
        seen.add(name)
        for included in INCLUDE.findall(path.read_text()):
            if included in all_files or (('/' not in included) and
                    included.endswith(('.h', '.hpp', '.cuh')) and
                    included not in ('assert.h', 'stdint.h', 'stdlib.h',
                                     'cuda.h', 'cuda_runtime.h', 'cuda_fp16.h',
                                     'cuda_bf16.h', 'cuda_fp8.h')):
                pending.append(included)
    return dict(directory=str(directory.relative_to(ROOT)),
                roots=list(ROOTS), reachable=sorted(seen),
                unreachable=sorted(all_files-seen), missing=sorted(set(missing)))


def main():
    results = [audit(d) for d in sorted(
        (ROOT/'operator/bfla/_csrc').glob('attention_d*')) if d.is_dir()]
    if len(results) != 2:
        raise SystemExit('expected both D256 and D512 forward header trees')
    print(json.dumps(results, indent=2))
    if any(r['missing'] or r['unreachable'] for r in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
