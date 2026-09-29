"""Restore a text bundle using only Python 3.11+ and this standalone script."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

MAGIC = "REPOSITORY-TEXT-BUNDLE-V1\n"
SEPARATOR = "===== REPOSITORY FILE =====\n"
END = "===== END REPOSITORY BUNDLE =====\n"


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate_path(name):
    if not isinstance(name, str) or not name:
        raise ValueError("Missing file path")
    parts = name.split("/")
    reserved = {"CON", "PRN", "AUX", "NUL"} | {f"{p}{i}" for p in ("COM", "LPT") for i in range(1, 10)}
    if PurePosixPath(name).is_absolute() or any(
        part in {"", ".", ".."} or part.endswith((" ", ".")) or
        any(ord(c) < 32 or c in '<>:"|?*\\' for c in part) or
        part.split(".")[0].upper() in reserved
        for part in parts
    ):
        raise ValueError(f"Unsafe path: {name!r}")
    if any(part.lower() in {".git", ".aws", ".venv", "node_modules", "__pycache__", "runs", "data", "outputs"}
           for part in parts):
        raise ValueError(f"Excluded directory: {name}")
    base = parts[-1].lower()
    if ((base == ".env" or base.startswith(".env.")) and base != ".env.example") or base.startswith("credentials"):
        raise ValueError(f"Excluded private file: {name}")
    if Path(base).suffix in {".pem", ".key", ".p12", ".pfx"}:
        raise ValueError(f"Excluded key file: {name}")
    return parts


def parse_bundle(text):
    # Clipboard/editors may normalize line endings. Exported files use LF in this repository.
    if not text.startswith(MAGIC):
        raise ValueError("Missing bundle header; paste the complete file without Markdown fences")
    offset, entries, seen = len(MAGIC), [], set()
    while not text.startswith(END, offset):
        if not text.startswith(SEPARATOR, offset):
            raise ValueError("Missing file separator or incomplete bundle")
        offset += len(SEPARATOR)
        line_end = text.index("\n", offset)
        header = json.loads(text[offset:line_end])
        name, size = header["path"], header["characters"]
        validate_path(name)
        if name.casefold() in seen:
            raise ValueError(f"Duplicate file: {name}")
        seen.add(name.casefold())
        if type(size) is not int or size < 0:
            raise ValueError("Invalid content length")
        offset = line_end + 1
        content = text[offset:offset + size]
        if len(content) != size or digest(content) != header["sha256"]:
            raise ValueError(f"Content changed or truncated: {name}")
        offset += size
        if text[offset:offset+1] != "\n":
            raise ValueError(f"Missing content boundary: {name}")
        offset += 1
        entries.append((name, content))
    if text[offset+len(END):].strip():
        raise ValueError("Unexpected text after the bundle")
    for name in seen:
        parts = name.split("/")
        if any("/".join(parts[:i]) in seen for i in range(1, len(parts))):
            raise ValueError("A file path conflicts with a parent directory")
    return entries


def restore_bundle(bundle, destination):
    # Validate every file before creating the destination. Never overwrite existing work.
    text = Path(bundle).read_bytes().decode("utf-8-sig").replace("\r\n", "\n")
    entries = parse_bundle(text)
    destination = Path(destination).absolute()
    if destination.exists() or destination.is_symlink():
        raise ValueError("Destination already exists; choose a new folder")
    destination.mkdir(parents=True)
    for name, content in entries:
        target = destination.joinpath(*validate_path(name))
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(content.encode("utf-8"))
    return len(entries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    count = restore_bundle(args.bundle, args.destination)
    print(f"Restored {count} files to {args.destination}")


if __name__ == "__main__":
    main()
