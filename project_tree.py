#!/usr/bin/env python3
# file: project_tree.py
from __future__ import annotations

import argparse
import base64
import fnmatch
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Set, Any

Tree = Dict[str, Union[str, List["Tree"], bool, int]]

# ---------------------------
# Utilities / normalization
# ---------------------------

def parse_csv(values: List[str]) -> List[str]:
    out: List[str] = []
    for v in values or []:
        parts = [p.strip() for p in v.split(",") if p.strip()]
        out.extend(parts)
    return out

def normalize_exts(exts: List[str]) -> List[str]:
    norm = []
    for e in exts:
        e = e.lower()
        if not e.startswith("."):
            e = "." + e
        norm.append(e)
    return norm

def to_posix(s: str) -> str:
    """Normalize path string to POSIX-style separators for consistent globbing."""
    if os.sep != "/":
        s = s.replace(os.sep, "/")
    # Also normalize possible Windows altsep, just in case
    if os.altsep and os.altsep != "/":
        s = s.replace(os.altsep, "/")
    return s

def normalize_patterns(patterns: List[str], case_insensitive: bool) -> List[str]:
    """
    Normalize a list of glob patterns for fnmatch:
    - Convert separators to POSIX (/)
    - Optionally lowercase for case-insensitive compares
    Note: Python's fnmatch treats '/' as a regular character, so patterns like
    '**/__pycache__' or 'docs/_build' will work with our normalized POSIX paths.
    """
    pats = [to_posix(p) for p in patterns]
    if case_insensitive:
        pats = [p.lower() for p in pats]
    return pats

def matches_any(path: str, patterns: List[str], case_insensitive: bool=False) -> bool:
    """Glob-match path (POSIX-style) against any pattern."""
    p = to_posix(path)
    if case_insensitive:
        p = p.lower()
    return any(fnmatch.fnmatch(p, pat) for pat in patterns)

# ---------------------------
# Exclusion helpers
# ---------------------------

def should_skip_dir(
    rel_dir: str,
    name: str,
    exclude_dirs: List[str],
    case_insensitive: bool,
) -> bool:
    """
    Exclude if either:
      - the *basename* matches a pattern (e.g., '.git', '.venv', 'build*')
      - the *relative path* matches a pattern (e.g., 'docs/_build', 'src/*/generated')
    """
    if not exclude_dirs:
        return False
    base_match = matches_any(name, exclude_dirs, case_insensitive)
    rel_path = to_posix(str(Path(rel_dir) / name)) if rel_dir else name
    rel_match = matches_any(rel_path, exclude_dirs, case_insensitive)
    return base_match or rel_match

def should_skip_file(
    rel_file: str,
    name: str,
    exts: List[str],
    exclude_files: List[str],
    case_insensitive: bool,
) -> bool:
    if exts:
        ext = Path(name).suffix.lower()
        if ext in exts:
            return True
    rel_posix = to_posix(rel_file) if rel_file else name
    if exclude_files and (matches_any(name, exclude_files, case_insensitive) or matches_any(rel_posix, exclude_files, case_insensitive)):
        return True
    return False

# ---------------------------
# File content helpers
# ---------------------------

def detect_binary(sample: bytes) -> bool:
    """Very simple heuristic: if NUL byte present, assume binary."""
    return b"\x00" in sample

def read_file_payload(
    path: Path,
    mode: str,
    max_bytes: int,
    want_hash: str,
) -> Dict[str, Any]:
    """
    Read up to max_bytes+1 to detect truncation.
    Returns dict with:
      - contents
      - encoding: "utf-8" | "base64"
      - content_truncated: bool
      - size: int (best-effort; os.stat)
      - encoding_error: bool (only when mode='text' and decode fails)
      - sha256: hex digest (if requested and read fully or partially)
    """
    payload: Dict[str, Any] = {}
    try:
        st = path.stat()
        payload["size"] = int(getattr(st, "st_size", 0))
    except OSError:
        payload["size"] = 0

    # Read bytes up to limit + 1 (to know if truncated)
    try:
        with path.open("rb") as f:
            data = f.read(max_bytes + 1 if max_bytes > 0 else None)
    except OSError:
        # Unreadable file
        payload.update({
            "contents": "",
            "encoding": "utf-8",
            "content_truncated": False,
            "encoding_error": True,
        })
        if want_hash == "sha256":
            payload["sha256"] = ""
        return payload

    truncated = max_bytes > 0 and len(data) > max_bytes
    if truncated:
        data = data[:max_bytes]
    payload["content_truncated"] = truncated

    if want_hash == "sha256":
        # Hash what we actually read (best-effort)
        h = hashlib.sha256()
        h.update(data)
        payload["sha256"] = h.hexdigest()

    mode = mode.lower()
    if mode == "base64":
        payload["contents"] = base64.b64encode(data).decode("ascii")
        payload["encoding"] = "base64"
        return payload

    if mode == "text":
        try:
            payload["contents"] = data.decode("utf-8")
            payload["encoding"] = "utf-8"
            payload["encoding_error"] = False
        except UnicodeDecodeError:
            payload["contents"] = ""
            payload["encoding"] = "utf-8"
            payload["encoding_error"] = True
        return payload

    # auto
    if not detect_binary(data):
        try:
            payload["contents"] = data.decode("utf-8")
            payload["encoding"] = "utf-8"
            payload["encoding_error"] = False
            return payload
        except UnicodeDecodeError:
            pass

    payload["contents"] = base64.b64encode(data).decode("ascii")
    payload["encoding"] = "base64"
    return payload

# ---------------------------
# Traversal context
# ---------------------------

@dataclass
class Context:
    follow_symlinks: bool
    confine_to_root: bool
    root_real: Path                      # canonical real path of root (if following, resolved; else absolute)
    max_depth: int
    max_nodes: int                       # 0 => unlimited
    case_insensitive: bool
    exclude_dirs: List[str]              # normalized patterns (POSIX, maybe lowercase)
    exclude_files: List[str]             # normalized patterns (POSIX, maybe lowercase)
    exclude_exts: List[str]              # lowercase with leading dot
    with_contents: bool = False
    contents_mode: str = "auto"          # "auto" | "text" | "base64"
    max_content_bytes: int = 262144      # 256 KiB per file
    hash_alg: str = "none"               # "none" | "sha256"
    node_count: int = 0                  # emitted nodes so far
    truncated: bool = False              # set if we stopped early due to max_nodes
    visited_inodes: Optional[Set[Tuple[int, int]]] = None  # (st_dev, st_ino) when following

    def mark_emit(self) -> bool:
        """Increment node count; return False if limit exceeded (and mark truncated)."""
        if self.max_nodes <= 0:
            return True  # unlimited: don't count
        if self.node_count < self.max_nodes:
            self.node_count += 1
            return True
        self.truncated = True
        return False

# ---------------------------
# Core traversal
# ---------------------------

def build_tree(
    root: Path,
    rel_dir: str,
    ctx: Context,
    depth: int = 0,
) -> Tree:
    # Directory node
    node: Tree = {
        "type": "directory",
        "name": (root.name if not rel_dir else Path(rel_dir).name),
        "path": rel_dir if rel_dir else ".",
        "children": [],
    }

    # Count this directory node
    if not ctx.mark_emit():
        return node

    # Depth check: if we've reached max depth, don't descend into children
    if ctx.max_depth >= 0 and depth >= ctx.max_depth:
        return node

    try:
        with os.scandir(root) as it:
            # Sort: directories first, then files; both by casefolded name for stability
            entries = sorted(
                it,
                key=lambda e: (not safe_is_dir(e, ctx.follow_symlinks), e.name.casefold()),
            )
            for entry in entries:
                # Stop early if node limit reached
                if ctx.max_nodes > 0 and ctx.node_count >= ctx.max_nodes:
                    ctx.truncated = True
                    break

                try:
                    # Symlink handling
                    is_link = entry.is_symlink()

                    # When following symlinks, prevent loops + escaping root
                    if ctx.follow_symlinks and (entry.is_dir(True) or entry.is_file(True)):
                        try:
                            st = entry.stat(follow_symlinks=True)
                            key = (st.st_dev, st.st_ino)
                        except FileNotFoundError:
                            # Broken symlink or race; skip
                            continue

                        if ctx.visited_inodes is None:
                            ctx.visited_inodes = set()

                        if key in ctx.visited_inodes:
                            # Already visited (cycle)
                            continue

                        # Confine traversal to root if requested
                        if ctx.confine_to_root:
                            target_real = Path(entry.path).resolve()
                            if target_real != ctx.root_real and ctx.root_real not in target_real.parents:
                                # Outside root; skip
                                continue

                    # Directory branch
                    if safe_is_dir(entry, ctx.follow_symlinks):
                        if should_skip_dir(rel_dir, entry.name, ctx.exclude_dirs, ctx.case_insensitive):
                            continue

                        child_rel = str(Path(rel_dir) / entry.name) if rel_dir else entry.name

                        # If following, mark visited inode now (after we decided to descend)
                        if ctx.follow_symlinks:
                            try:
                                st = entry.stat(follow_symlinks=True)
                                key = (st.st_dev, st.st_ino)
                                ctx.visited_inodes.add(key)  # type: ignore[union-attr]
                            except FileNotFoundError:
                                continue

                        child = build_tree(
                            Path(entry.path),
                            child_rel,
                            ctx,
                            depth + 1,
                        )
                        append_child(node, child)
                        continue

                    # File branch (regular or via symlink)
                    if safe_is_file(entry, ctx.follow_symlinks):
                        rel_file = str(Path(rel_dir) / entry.name) if rel_dir else entry.name
                        if should_skip_file(rel_file, entry.name, ctx.exclude_exts, ctx.exclude_files, ctx.case_insensitive):
                            continue

                        file_node: Tree = {
                            "type": "file",
                            "name": entry.name,
                            "path": rel_file,
                        }
                        # If it's a symlink (and we're not following), represent as symlink node with target
                        if not ctx.follow_symlinks and is_link:
                            try:
                                target = os.readlink(entry.path)
                            except OSError:
                                target = None
                            file_node["type"] = "symlink"
                            file_node["target"] = target if target is not None else ""

                        # Optionally include file contents (only for real file nodes)
                        if ctx.with_contents and file_node["type"] == "file":
                            payload = read_file_payload(
                                Path(entry.path),
                                mode=ctx.contents_mode,
                                max_bytes=ctx.max_content_bytes,
                                want_hash=ctx.hash_alg,
                            )
                            # Merge selected fields into node
                            file_node.update(payload)  # type: ignore[arg-type]

                        if ctx.mark_emit():
                            append_child(node, file_node)  # type: ignore[arg-type]
                        else:
                            return node
                        continue

                    # If not a dir/file, but a symlink and we are NOT following -> emit symlink node (unknown target type)
                    if is_link and not ctx.follow_symlinks:
                        rel_file = str(Path(rel_dir) / entry.name) if rel_dir else entry.name
                        if should_skip_file(rel_file, entry.name, ctx.exclude_exts, ctx.exclude_files, ctx.case_insensitive):
                            continue
                        try:
                            target = os.readlink(entry.path)
                        except OSError:
                            target = None
                        link_node: Tree = {
                            "type": "symlink",
                            "name": entry.name,
                            "path": rel_file,
                            "target": target if target is not None else "",
                        }
                        if ctx.mark_emit():
                            append_child(node, link_node)  # type: ignore[arg-type]
                        else:
                            return node
                        continue

                    # Other types (socket, fifo, device): skip silently
                except PermissionError:
                    # Skip items we can't stat/read
                    continue
    except PermissionError:
        pass

    # If we truncated overall, mark on root node only
    if not rel_dir and ctx.truncated:
        node["truncated"] = True
    return node

def safe_is_dir(entry: os.DirEntry, follow: bool) -> bool:
    try:
        return entry.is_dir(follow_symlinks=follow)
    except PermissionError:
        return False

def safe_is_file(entry: os.DirEntry, follow: bool) -> bool:
    try:
        return entry.is_file(follow_symlinks=follow)
    except PermissionError:
        return False

def append_child(dir_node: Tree, child: Tree) -> None:
    # Ensure children list exists
    children = dir_node.get("children")
    if not isinstance(children, list):
        dir_node["children"] = [child]
    else:
        children.append(child)

# ---------------------------
# CLI / main
# ---------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dump a JSON project tree for a codebase, with exclusions for dirs, file extensions, and files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=".",
        help="Root directory to scan",
    )
    parser.add_argument(
        "--exclude-dirs",
        "-D",
        action="append",
        default=[],
        help="Comma-separated list or repeatable option of directory names or glob patterns to exclude "
             "(e.g. '.git,build,dist' or 'docs/_build,**/__pycache__').",
    )
    parser.add_argument(
        "--exclude-exts",
        "-E",
        action="append",
        default=[],
        help="Comma-separated list or repeatable option of file extensions to exclude (with or without dot), "
             "case-insensitive (e.g. '.pyc,.log' or 'pyc,log'). "
             "Note: only the final suffix is checked; use --exclude-files for multi-suffix names (e.g. '*.tar.gz').",
    )
    parser.add_argument(
        "--exclude-files",
        "-F",
        action="append",
        default=[],
        help="Comma-separated list or repeatable option of file basenames or glob patterns to exclude "
             "(e.g. 'uv.lock,pyproject.lock,*.min.js').",
    )
    parser.add_argument(
        "--follow-symlinks",
        action="store_true",
        help="Follow symlinks (default: do not follow).",
    )
    parser.add_argument(
        "--confine-to-root",
        action="store_true",
        help="When following symlinks, skip any entry whose resolved path is outside the root.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=-1,
        help="Maximum directory depth to traverse (0 = root only, 1 = root + its children, -1 = unlimited).",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=0,
        help="Abort traversal after emitting this many nodes (0 = unlimited). Helpful for huge repos.",
    )
    parser.add_argument(
        "--ignore-case",
        "-i",
        action="store_true",
        help="Case-insensitive matching for exclude patterns (dirs/files).",
    )
    parser.add_argument(
        "--uv-default-excludes",
        action="store_true",
        help="Exclude typical uv/Python build dirs and caches (.venv, **/__pycache__, .pytest_cache, .mypy_cache, "
             ".ruff_cache, .nox, .tox, .hypothesis, build, dist, .git, node_modules) and files (uv.lock, OS junk).",
    )
    # ---- New content options ----
    parser.add_argument(
        "--with-contents",
        action="store_true",
        help="Include file contents in 'contents' with 'encoding' metadata for each file node.",
    )
    parser.add_argument(
        "--contents-mode",
        choices=["auto", "text", "base64"],
        default="auto",
        help="How to embed file contents: auto (UTF-8 when possible, else Base64), always text (UTF-8), or always Base64.",
    )
    parser.add_argument(
        "--max-content-bytes",
        type=int,
        default=262144,
        help="Max bytes to read per file for contents. If exceeded, 'content_truncated' is set true.",
    )
    parser.add_argument(
        "--hash",
        choices=["none", "sha256"],
        default="none",
        help="Optionally include a hash digest for files.",
    )
    # ---- Output ----
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON with indentation.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="-",
        help="Output file path ('-' for stdout).",
    )

    args = parser.parse_args()

    root_path = Path(args.root).resolve()
    # Parse & normalize patterns
    raw_exclude_dirs = parse_csv(args.exclude_dirs)
    raw_exclude_files = parse_csv(args.exclude_files)
    exclude_dirs = normalize_patterns(raw_exclude_dirs, case_insensitive=args.ignore_case)
    exclude_files = normalize_patterns(raw_exclude_files, case_insensitive=args.ignore_case)
    exclude_exts = normalize_exts(parse_csv(args.exclude_exts))

    if args.uv_default_excludes:
        uv_dirs = [
            ".venv",
            "**/__pycache__",
            ".pytest_cache",
            ".mypy_cache",
            ".ruff_cache",
            ".nox",
            ".tox",
            ".hypothesis",
            "build",
            "dist",
            ".git",
            ".hg",
            ".svn",
            ".direnv",
            ".idea",
            ".vscode",
            "node_modules",
            "out",
            "site",
            "KaBUC_SpurPlan.egg-info",
            "tests_hide"
        ]
        uv_files = [
            "uv.lock",
            ".DS_Store",
            "Thumbs.db",
            ".coverage",
            "coverage.xml",
            "project_tree.py",
            "project_tree_out.json",
            "restore_from_project_tree.py",
            "codebase.json",
        ]
        exclude_dirs.extend(normalize_patterns(uv_dirs, case_insensitive=args.ignore_case))
        exclude_files.extend(normalize_patterns(uv_files, case_insensitive=args.ignore_case))

    # Determine canonical root for confinement and loop prevention
    root_real = root_path if not args.follow_symlinks else root_path.resolve()

    ctx = Context(
        follow_symlinks=args.follow_symlinks,
        confine_to_root=args.confine_to_root,
        root_real=root_real,
        max_depth=args.max_depth,
        max_nodes=args.max_nodes,
        case_insensitive=args.ignore_case,
        exclude_dirs=exclude_dirs,
        exclude_files=exclude_files,
        exclude_exts=exclude_exts,
        with_contents=args.with_contents,
        contents_mode=args.contents_mode,
        max_content_bytes=args.max_content_bytes,
        hash_alg=args.hash,
        visited_inodes=set() if args.follow_symlinks else None,
    )

    # When following symlinks, mark root as visited to avoid revisiting via a back-symlink.
    if ctx.follow_symlinks and ctx.visited_inodes is not None:
        try:
            st = root_real.stat()
            ctx.visited_inodes.add((st.st_dev, st.st_ino))
        except OSError:
            pass

    tree = build_tree(
        root=root_path,
        rel_dir="",
        ctx=ctx,
        depth=0,
    )

    # If we truncated due to node limit, ensure the marker is visible on the root
    if ctx.truncated and "truncated" not in tree:
        tree["truncated"] = True  # type: ignore[index]

    data = json.dumps(tree, indent=2 if args.pretty else None, ensure_ascii=False)
    if args.output == "-" or not args.output:
        print(data)
    else:
        Path(args.output).write_text(data, encoding="utf-8")

if __name__ == "__main__":
    main()