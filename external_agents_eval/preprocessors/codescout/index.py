"""Deterministic, offline repository graph and source retrieval.

Entity locations refer to the frozen bytes parsed here, not a later filesystem read.
No project imports, subprocesses, history inspection, or network access are used.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import os
import re
from bisect import bisect_right
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath

LANGUAGES = {
    ".py": "python", ".pyi": "python", ".js": "javascript", ".jsx": "javascript",
    ".mjs": "javascript", ".cjs": "javascript", ".ts": "typescript",
    ".mts": "typescript", ".cts": "typescript", ".tsx": "tsx",
    ".go": "go", ".rs": "rust",
}
EXCLUDED = (
    ".git", "node_modules", ".venv", "venv", "__pycache__", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", ".tox", ".next", ".cache", "target",
)
CLASS_NODES = {
    "class_definition", "class_declaration", "abstract_class_declaration", "class",
    "interface_declaration", "struct_item", "enum_item", "trait_item", "type_spec",
}
FUNCTION_NODES = {
    "function_definition", "function_declaration", "generator_function_declaration",
    "method_definition", "method_declaration", "method_signature", "function_signature",
    "function_item", "function_signature_item", "method_spec", "method_elem",
}
IMPORT_NODES = {
    "import_statement", "import_from_statement", "import_spec", "use_declaration",
    "extern_crate_declaration",
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass
class Entity:
    id: str
    kind: str
    name: str
    qualified_name: str
    path: str
    line_start: int
    line_end: int
    byte_start: int
    byte_end: int
    signature: str
    parent: str
    syntax: str


@dataclass
class SourceFile:
    path: str
    language: str | None
    size: int
    sha256: str | None
    data: bytes | None = field(repr=False)
    parse_errors: bool = False
    unavailable: str | None = None

    def metadata(self):
        return {key: value for key, value in vars(self).items() if key != "data"}


@dataclass
class RepositoryIndex:
    files: dict[str, SourceFile]
    entities: list[Entity]
    edges: list[dict]
    skipped: list[dict]
    excluded_dirs: tuple[str, ...]

    @property
    def parser_versions(self):
        languages = {source.language for source in self.files.values() if source.language}
        packages = {"tree-sitter"} | {
            "tree-sitter-" + ("typescript" if language == "tsx" else language)
            for language in languages}
        return {name: importlib.metadata.version(name) for name in sorted(packages)}

    @property
    def fingerprint(self):
        value = {path: source.metadata() for path, source in sorted(self.files.items())}
        return digest(json.dumps(value, sort_keys=True).encode())

    def graph(self):
        return {
            "schema_version": 1, "fingerprint": self.fingerprint,
            "parser_versions": self.parser_versions,
            "files": [source.metadata() for source in self.files.values()],
            "entities": [asdict(entity) for entity in self.entities], "edges": self.edges,
            "skipped": self.skipped, "excluded_directories": self.excluded_dirs,
        }

    def outline(self, max_chars=240_000):
        # Retain every eligible path. If entity annotations do not fit, count them
        # explicitly; never silently chop the file tree in alphabetical order.
        tree = "REPOSITORY FILE TREE (paths relative to repository root)\n" + "\n".join(self.files)
        if len(tree) + 250 > max_chars:
            raise ValueError("Complete file tree exceeds max_outline_chars; increase this setting")
        lines = [tree, "\nCODE ENTITIES (path:line type qualified_name signature)"]
        size = sum(len(line) + 1 for line in lines)
        omitted = 0
        for entity in self.entities:
            if entity.kind not in {"class", "function"}:
                continue
            line = (f"{entity.path}:{entity.line_start} {entity.kind} "
                    f"{entity.qualified_name} {entity.signature[:180]}")
            if size + len(line) + 250 > max_chars:
                omitted += 1
            else:
                lines.append(line)
                size += len(line) + 1
        lines.append(f"ENTITY ANNOTATIONS OMITTED BY BUDGET: {omitted}; file tree is complete.")
        return "\n".join(lines), {"omitted_entities": omitted, "complete_file_tree": True}

    def retrieve(self, target_type, target_name, *, max_chars=240_000):
        name = target_name.strip().strip("`\"'").replace("\\", "/")
        name = name.removeprefix("/app/")
        name = name.removeprefix("./")
        if name.startswith("/") or ".." in PurePosixPath(name).parts:
            return {"status": "invalid_path", "matches": [], "text": ""}
        if target_type == "file":
            source = self.files.get(name)
            if source is None:
                return {"status": "not_found", "matches": [], "text": ""}
            matches = [(source, None)]
        else:
            candidates = [entity for entity in self.entities if entity.kind == target_type]
            # Resolve both public dotted names and file-qualified names. Keep all
            # same-tier matches: never choose an arbitrary first same-name method.
            normalized = name.replace("::", ".").replace("/", ".").replace(":", ".")
            matched = []
            parts = normalized.split(".")
            for offset in range(len(parts)):
                suffix = ".".join(parts[offset:])
                matched = [entity for entity in candidates if (
                    suffix == entity.qualified_name or
                    _full_name(entity).endswith("." + suffix) or
                    _full_name(entity) == suffix
                )]
                if matched:
                    break
            if not matched:
                return {"status": "not_found", "matches": [], "text": ""}
            if len(matched) > 8:
                return {"status": "ambiguous_too_many", "text": "",
                        "matches": [asdict(entity) for entity in matched]}
            matches = [(self.files[entity.path], entity) for entity in matched]
        records, blocks = [], []
        budget = max_chars // len(matches) - 300
        for source, entity in matches:
            record = {"path": source.path, "file_sha256": source.sha256,
                      "entity": asdict(entity) if entity else None}
            if source.data is None:
                record.update(status=source.unavailable, text="")
            else:
                data = source.data[entity.byte_start:entity.byte_end] if entity else source.data
                start = entity.line_start if entity else 1
                text = data.decode("utf-8")
                # No implicit source truncation: an oversized target is unavailable,
                # visible in artifacts, and skipped as in paper Algorithm 1.
                if len(text) > budget:
                    record.update(status="source_budget_exceeded", chars=len(text), text="")
                else:
                    record.update(status="retrieved", line_start=start,
                                  line_end=start + text.count("\n"),
                                  source_sha256=digest(data), text=text)
                    blocks.append(f"FILE: {source.path}\nSTART LINE: {start}\n{text}")
            records.append(record)
        complete = all(record["status"] == "retrieved" for record in records)
        # Passing only one of multiple ambiguous matches would silently bias the
        # analysis. Skip the target if any match cannot be supplied in full.
        return {"status": "retrieved" if complete else "unavailable",
                "ambiguous": len(matches) > 1, "matches": records,
                "text": "\n\n".join(blocks) if complete else ""}


def _full_name(entity):
    module = str(PurePosixPath(entity.path).with_suffix("")).replace("/", ".")
    module = module.removesuffix(".__init__")
    return module + "." + entity.qualified_name


def _parser(language):
    from tree_sitter import Language, Parser

    module = importlib.import_module("tree_sitter_" + (
        "typescript" if language == "tsx" else language))
    factory = getattr(module, "language_" + language) if language in {
        "typescript", "tsx"} else module.language
    return Parser(Language(factory()))


def _text(node, data):
    return data[node.start_byte:node.end_byte].decode("utf-8") if node else ""


def _extract(source, parser):
    data, path = source.data, source.path
    tree = parser.parse(data)
    root = tree.root_node
    line_starts = [0, *(match.end() for match in re.finditer(b"\n", data))]
    source.parse_errors = root.has_error
    entities, edges = [], []

    def add(node, kind, name, scope, parent, span=None):
        location = span if span is not None else node
        body = node.child_by_field_name("body")
        header = data[node.start_byte:body.start_byte] if body else data[
            node.start_byte:min(node.end_byte, node.start_byte + 240)]
        qualified = ".".join((*scope, name))
        entity = Entity(
            f"{path}:{location.start_byte}:{kind}:{qualified}", kind, name, qualified, path,
            bisect_right(line_starts, location.start_byte),
            bisect_right(line_starts, location.end_byte),
            location.start_byte, location.end_byte,
            # Full files were UTF-8 validated before parsing. A bounded signature
            # prefix alone may end inside its final character; omit that partial
            # character, never alter the full source bytes used for retrieval.
            re.sub(r"\s+", " ", header.decode("utf-8", errors="ignore")).strip(), parent, node.type,
        )
        entities.append(entity)
        edges.append({"source": parent, "relation": "contains", "target": entity.id})
        return entity

    def walk(node, scope=(), parent=None):
        parent = parent or "file:" + path
        name = _text(node.child_by_field_name("name"), data)
        if node.type == "impl_item":
            owner = _text(node.child_by_field_name("type"), data)
            trait = _text(node.child_by_field_name("trait"), data)
            if trait:
                edges.append({"source": "symbol:" + owner,
                              "relation": "implements", "target": "symbol:" + trait})
            scope = (*scope, re.sub(r"<.*", "", owner)) if owner else scope
        elif node.type in CLASS_NODES and name:
            span = node.parent if node.parent and node.parent.type == "decorated_definition" else node
            entity = add(node, "class", name, scope, parent, span)
            bases = node.child_by_field_name("superclasses")
            if bases:
                for base in bases.named_children:
                    edges.append({"source": entity.id, "relation": "inherits",
                                  "target": "symbol:" + _text(base, data)})
            for child in node.named_children:
                if child.type in {"class_heritage", "extends_type_clause"}:
                    edges.append({"source": entity.id, "relation": "inherits",
                                  "target": "symbol:" + _text(child, data)})
            scope, parent = (*scope, name), entity.id
        elif node.type in FUNCTION_NODES and name:
            receiver = node.child_by_field_name("receiver")
            if receiver:
                types = []
                stack = [receiver]
                while stack:
                    current = stack.pop()
                    if current.type in {"type_identifier", "generic_type"}:
                        types.append(_text(current, data))
                    stack.extend(reversed(current.named_children))
                if types:
                    scope = (*scope, re.sub(r"<.*|\[.*", "", types[0]))
            span = node.parent if node.parent and node.parent.type == "decorated_definition" else node
            entity = add(node, "function", name, scope, parent, span)
            scope, parent = (*scope, name), entity.id
        elif node.type in {"variable_declarator", "assignment", "let_declaration",
                           "short_var_declaration", "var_spec", "const_spec",
                           "field_declaration", "public_field_definition", "property_signature"}:
            lhs = (node.child_by_field_name("name") or node.child_by_field_name("left") or
                   node.child_by_field_name("pattern"))
            value = node.child_by_field_name("value") or node.child_by_field_name("right")
            label = _text(lhs, data)
            # Destructuring remains in metadata; do not invent individual bindings.
            if label:
                callable_value = value and value.type in {
                    "arrow_function", "function_expression", "generator_function"}
                entity = add(node, "function" if callable_value else "variable",
                             label, scope, parent)
                if callable_value or (value and value.type == "object"):
                    scope, parent = (*scope, label), entity.id
        elif node.type in IMPORT_NODES or (
                node.type == "export_statement" and node.child_by_field_name("source")):
            raw = _text(node, data)
            module = (node.child_by_field_name("source") or node.child_by_field_name("path") or
                      node.child_by_field_name("module_name") or node.child_by_field_name("argument"))
            module_name = _text(module, data).strip("\"'`") if module else raw
            entity = add(node, "import", module_name, scope, parent)
            modules = [module_name]
            if source.language == "python" and node.type == "import_statement":
                modules = [_text(child.child_by_field_name("name") or child, data)
                           for child in node.named_children]
            for dependency in modules:
                edges.append({"source": "file:" + path, "relation": "imports",
                              "target": "module:" + dependency, "declaration": entity.id})
        for child in node.named_children:
            walk(child, scope, parent)

    walk(root)
    return entities, edges


def build_index(repository, *, max_file_bytes=4_000_000, excluded_dirs=EXCLUDED):
    root = Path(repository).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Repository must be a directory")
    files, entities, edges, skipped, parsers = {}, [], [], [], {}
    for folder, directories, names in os.walk(root, followlinks=False):
        retained = []
        for name in sorted(directories):
            path = Path(folder) / name
            if name in excluded_dirs or path.is_symlink():
                skipped.append({"path": path.relative_to(root).as_posix(),
                                "reason": "symlink" if path.is_symlink() else "excluded_directory"})
            else:
                retained.append(name)
        directories[:] = retained
        for name in sorted(names):
            path = Path(folder) / name
            relative = path.relative_to(root).as_posix()
            if path.is_symlink() or name == ".git":
                skipped.append({"path": relative, "reason": "symlink_or_git_metadata"})
                continue
            if not path.is_file():
                skipped.append({"path": relative, "reason": "non_regular_file"})
                continue
            size = path.stat().st_size
            language = LANGUAGES.get(path.suffix)
            data, sha, reason = None, None, None
            if size > max_file_bytes:
                reason = "file_size_limit"
            else:
                raw = path.read_bytes()
                sha = digest(raw)
                try:
                    raw.decode("utf-8")
                    if b"\x00" in raw:
                        raise UnicodeError("Binary file")
                    data = raw
                except UnicodeError:
                    reason = "non_utf8_or_binary"
            source = SourceFile(relative, language, size, sha, data, unavailable=reason)
            files[relative] = source
            if language and data is not None:
                if language not in parsers:
                    parsers[language] = _parser(language)
                extracted, relations = _extract(source, parsers[language])
                entities.extend(extracted)
                edges.extend(relations)
    files = dict(sorted(files.items()))
    entities.sort(key=lambda entity: (entity.path, entity.byte_start, entity.kind))
    return RepositoryIndex(files, entities, edges, skipped, tuple(excluded_dirs))
