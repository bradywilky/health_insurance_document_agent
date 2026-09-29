"""Create the transfer text bundle from tracked and non-ignored source files. Requires Git."""
import argparse
import json
from pathlib import Path
import subprocess

if __package__:
    from .restore_repo import MAGIC, SEPARATOR, END, digest, validate_path
else:
    from restore_repo import MAGIC, SEPARATOR, END, digest, validate_path


def export_bundle(root, output):
    root = Path(root).resolve()
    output = Path(output).resolve()
    names = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=root
    ).decode("utf-8").split("\0")
    entries = []
    for name in sorted(set(filter(None, names))):
        # Always exclude the canonical bundle, even when exporting to another location.
        if name.casefold() == "transfer/repository-bundle.txt":
            continue
        source = root.joinpath(*validate_path(name))
        if source.resolve() == output or not source.exists():
            continue
        if source.is_symlink() or not source.resolve().is_relative_to(root):
            raise ValueError(f"Linked file not supported: {name}")
        raw = source.read_bytes()
        text = raw.decode("utf-8").replace("\r\n", "\n")
        if "\0" in text:
            raise ValueError(f"Binary file not supported: {name}")
        entries.append((name, text))
    chunks = [MAGIC]
    for name, text in entries:
        chunks += [SEPARATOR, json.dumps({"path":name, "characters":len(text), "sha256":digest(text)}) + "\n",
                   text, "\n"]
    chunks += [END]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes("".join(chunks).encode("utf-8"))
    return len(entries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, help="Default: <root>/transfer/repository-bundle.txt")
    args = parser.parse_args()
    output = args.output or args.root / "transfer" / "repository-bundle.txt"
    count = export_bundle(args.root, output)
    print(f"Bundled {count} files into {output}")


if __name__ == "__main__":
    main()
