#!/usr/bin/env python3
"""Install rdagent factor_data_template from repo into the installed rdagent package.

This script copies the contents of `rdagent_workspace/factor_data_template`
to the installed `rdagent` package's `scenarios/qlib/experiment/factor_data_template`
so helper containers that mount the package path will always see the template
and HDF files.
"""
import shutil
from pathlib import Path
import sys


def main() -> int:
    repo_template = Path("rdagent_workspace") / "factor_data_template"
    if not repo_template.exists():
        print(f"Repository template not found: {repo_template}")
        return 2

    # locate installed rdagent package
    try:
        import rdagent
    except Exception as e:
        print(f"Failed to import rdagent: {e}")
        return 3

    # determine rdagent package path robustly (supports namespace packages)
    rdagent_path = None
    if getattr(rdagent, "__file__", None):
        rdagent_path = Path(rdagent.__file__).resolve().parent
    else:
        import importlib.util

        spec = importlib.util.find_spec("rdagent")
        if spec is None:
            print("Could not find rdagent spec")
            return 5
        if spec.submodule_search_locations:
            # package directory
            rdagent_path = Path(list(spec.submodule_search_locations)[0]).resolve()
        else:
            # fallback to origin
            if spec.origin:
                rdagent_path = Path(spec.origin).resolve().parent

    if rdagent_path is None:
        print("Unable to determine rdagent package path")
        return 6
    target = rdagent_path / "scenarios" / "qlib" / "experiment" / "factor_data_template"

    print(f"Repo template: {repo_template}")
    print(f"Detected rdagent package path: {rdagent_path}")
    print(f"Target template dir: {target}")

    target.mkdir(parents=True, exist_ok=True)

    copied = []
    for p in repo_template.iterdir():
        if p.is_file():
            dst = target / p.name
            try:
                shutil.copy2(p, dst)
                # ensure generate.py is executable
                if p.name.endswith("generate.py"):
                    dst.chmod(0o755)
                copied.append(dst)
            except Exception as e:
                print(f"Failed to copy {p} -> {dst}: {e}")

    if copied:
        print("Copied files:")
        for c in copied:
            print(" -", c)
        return 0
    else:
        print("No files copied (template directory possibly empty)")
        return 4


if __name__ == "__main__":
    sys.exit(main())
