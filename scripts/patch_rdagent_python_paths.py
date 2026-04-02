import os
from pathlib import Path


FILES = {
    Path("/Users/apple/miniconda3/envs/qlib_env/lib/python3.10/site-packages/rdagent/scenarios/qlib/experiment/utils.py"): {
        'entry=f"python generate.py"': 'entry=f"{os.getenv(\'FACTOR_CoSTEER_PYTHON_BIN\', \'python\')} generate.py"',
    },
    Path("/Users/apple/miniconda3/envs/qlib_env/lib/python3.10/site-packages/rdagent/scenarios/qlib/experiment/workspace.py"): {
        'entry="python read_exp_res.py"': 'entry=f"{os.getenv(\'FACTOR_CoSTEER_PYTHON_BIN\', \'python\')} read_exp_res.py"',
    },
    Path("/Users/apple/miniconda3/envs/qlib_env/lib/python3.10/site-packages/rdagent/scenarios/shared/get_runtime_info.py"): {
        'entry=f"python {fname}"': 'entry=f"{os.getenv(\'FACTOR_CoSTEER_PYTHON_BIN\', \'python\')} {fname}"',
        'entry="python -m coverage --version || echo MISSING"': 'entry=f"{os.getenv(\'FACTOR_CoSTEER_PYTHON_BIN\', \'python\')} -m coverage --version || echo MISSING"',
    },
}


def main() -> None:
    for path, replacements in FILES.items():
        text = path.read_text()
        original = text
        for old, new in replacements.items():
            if old in text:
                text = text.replace(old, new)
        if text != original:
            path.write_text(text)
        print(path)


if __name__ == "__main__":
    main()