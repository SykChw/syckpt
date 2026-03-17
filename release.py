#!/usr/bin/env python3

import sys
import re
import subprocess
from pathlib import Path

def run(cmd):
    print(">", " ".join(cmd))
    subprocess.run(cmd, check=True)

def update_pyproject(version):
    path = Path("pyproject.toml")
    if not path.exists():
        return

    text = path.read_text()
    text = re.sub(r'version\s*=\s*"[^\"]+"', f'version = "{version}"', text)
    path.write_text(text)
    print("Updated pyproject.toml")


def update_setup(version):
    path = Path("setup.py")
    if not path.exists():
        return

    text = path.read_text()
    text = re.sub(r'version\s*=\s*[\'"][^\'"]+[\'"]', f'version="{version}"', text)
    path.write_text(text)
    print("Updated setup.py")


def main():
    if len(sys.argv) != 2:
        print("Usage: python release.py <version>")
        sys.exit(1)


    version = sys.argv[1]

    update_pyproject(version)
    update_setup(version)

    run(["git", "add", "-A"])
    run(["git", "commit", "-m", f"v{version}"])
    run(["git", "tag", f"v{version}"])
    run(["git", "push"])
    run(["git", "push", "--tags"])

    print(f"\n✅ Release v{version} pushed. GitHub Actions should publish to PyPI.")


if __name__ == "__main__":
    main()