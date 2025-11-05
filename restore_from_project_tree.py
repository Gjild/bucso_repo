#!/usr/bin/env python3

"""
Restore files and directories from a JSON produced by `project_tree.py`.

Features:
- Recreates directories, regular files, and symlinks (if present in JSON).
- Decodes file contents from "utf-8" (text) or "base64" (binary) per node["encoding"].
- Safety checks for truncated contents; skip by default unless --allow-truncated.
- Optional hash verification when JSON includes node["sha256"] and you pass --verify-hash.
- Dry-run mode to preview actions.
- Can force-overwrite existing files/symlinks.

Usage:
  python restore_from_project_tree.py project_tree_out.json --dest ./restored \
      [--dry-run] [--force] [--allow-truncated] [--verify-hash]

Notes:
- The JSON's node["path"] is treated as a POSIX-style relative path. We recreate it under --dest.
- If your JSON was created with `--max-content-bytes` and some nodes show "content_truncated": true,
  you will need to re-run the original export with a higher limit to fully restore those files,
  or use --allow-truncated to write the partial contents (not recommended).
"""
import argparse
import base64
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable

def iter_nodes(tree: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """Yield all nodes in the tree (preorder)."""
    stack = [tree]
    while stack:
        node = stack.pop()
        yield node
        if node.get("type") == "directory":
            for child in reversed(node.get("children", []) or []):
                stack.append(child)

def ensure_dir(path: Path, dry_run: bool) -> None:
    if dry_run:
        print(f"[DRY] mkdir -p {path}")
        return
    path.mkdir(parents=True, exist_ok=True)

def write_file(path: Path, data: bytes, force: bool, dry_run: bool) -> None:
    if path.exists():
        if path.is_dir():
            raise IsADirectoryError(f"Refusing to overwrite directory with file: {path}")
        if not force:
            raise FileExistsError(f"File exists (use --force to overwrite): {path}")
        if dry_run:
            print(f"[DRY] overwrite {path}")
            return
        # ensure parent dir exists
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)
        return
    # new file
    if dry_run:
        print(f"[DRY] write {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)

def create_symlink(path: Path, target: str, force: bool, dry_run: bool) -> None:
    if path.exists() or path.is_symlink():
        if not force:
            raise FileExistsError(f"Symlink/file exists (use --force to overwrite): {path}")
        if dry_run:
            print(f"[DRY] remove existing and symlink {path} -> {target}")
            return
        try:
            if path.is_dir() and not path.is_symlink():
                # remove dir carefully
                raise IsADirectoryError(f"Refusing to replace directory with symlink: {path}")
            path.unlink()
        except FileNotFoundError:
            pass
    if dry_run:
        print(f"[DRY] ln -s {target} {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(target, path)

def decode_contents(node: Dict[str, Any]) -> bytes:
    if "contents" not in node:
        raise ValueError("Missing 'contents' for file node")
    enc = node.get("encoding", "utf-8")
    if enc == "base64":
        try:
            return base64.b64decode(node["contents"])
        except Exception as e:
            raise ValueError(f"Base64 decode failed for {node.get('path')}: {e}")
    # default to utf-8
    text = node["contents"]
    if not isinstance(text, str):
        raise ValueError(f"Unexpected non-string text contents for {node.get('path')}")
    return text.encode("utf-8")

def verify_hash(data: bytes, node: Dict[str, Any]) -> None:
    algo = node.get("sha256")
    if not algo:
        return
    h = hashlib.sha256()
    h.update(data)
    digest = h.hexdigest()
    if digest != algo:
        raise ValueError(f"SHA256 mismatch for {node.get('path')}: got {digest}, expected {algo}")

def restore(tree: Dict[str, Any], dest: Path, *, dry_run: bool, force: bool, allow_truncated: bool, verify_hashes: bool) -> None:
    # sanity
    if tree.get("type") != "directory":
        raise ValueError("Root node must be a directory")
    # Root itself maps to dest; children use their relative 'path'
    ensure_dir(dest, dry_run=dry_run)

    for node in iter_nodes(tree):
        ntype = node.get("type")
        rel = node.get("path")
        if rel is None:
            # Root may have path "." or missing; skip
            continue
        # Normalize rel path as relative
        rel_path = Path(rel)
        while rel_path.is_absolute():
            # guard: do not allow absolute paths
            rel_path = Path(*rel_path.parts[1:])
        out_path = (dest / rel_path).resolve()

        # confinement: must stay under dest
        if dest.resolve() not in out_path.parents and out_path != dest.resolve():
            raise ValueError(f"Refusing to write outside destination: {out_path}")

        if ntype == "directory":
            if rel == ".":
                # already created dest
                continue
            ensure_dir(out_path, dry_run=dry_run)
        elif ntype == "file":
            if node.get("content_truncated") and not allow_truncated:
                print(f"[SKIP] {rel} has content_truncated=true (use --allow-truncated to write partial)")
                continue
            if node.get("encoding_error"):
                print(f"[WARN] {rel} had encoding_error=true in export; restoring empty or base64 as present.")
            data = decode_contents(node)
            if verify_hashes:
                verify_hash(data, node)
            write_file(out_path, data, force=force, dry_run=dry_run)
        elif ntype == "symlink":
            target = node.get("target", "")
            if not isinstance(target, str):
                print(f"[WARN] {rel} symlink has non-string target; skipping")
                continue
            try:
                create_symlink(out_path, target, force=force, dry_run=dry_run)
            except (NotImplementedError, OSError) as e:
                print(f"[WARN] Could not create symlink {rel} -> {target}: {e}")
        else:
            # Unknown node type; ignore
            print(f"[INFO] Skipping unknown node type for {rel}: {ntype}")

def main() -> None:
    ap = argparse.ArgumentParser(description="Restore a codebase from project_tree_out.json")
    ap.add_argument("json_path", help="Path to project_tree_out.json")
    ap.add_argument("--dest", default="./restored", help="Destination directory to rebuild into")
    ap.add_argument("--dry-run", action="store_true", help="Print actions without writing")
    ap.add_argument("--force", action="store_true", help="Overwrite existing files/symlinks")
    ap.add_argument("--allow-truncated", action="store_true", help="Write files even if content_truncated=true")
    ap.add_argument("--verify-hash", action="store_true", help="Verify SHA256 hashes when present in JSON")
    args = ap.parse_args()

    json_path = Path(args.json_path)
    dest = Path(args.dest)

    with json_path.open("r", encoding="utf-8") as f:
        tree = json.load(f)

    restore(
        tree=tree,
        dest=dest,
        dry_run=args.dry_run,
        force=args.force,
        allow_truncated=args.allow_truncated,
        verify_hashes=args.verify_hash,
    )

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)