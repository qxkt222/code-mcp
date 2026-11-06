"""Rust front end: items, impl/trait methods, uses and call sites."""

from __future__ import annotations

import tree_sitter_rust
from tree_sitter import Language, Node, Parser

from ...types import REF_CALL, REF_IMPORT, REF_INHERIT, REF_TYPE
from . import (
    Collector,
    ParsedFile,
    clean_block_doc,
    collapse_ws,
    node_end_line,
    node_start_line,
)

_PARSER = Parser(Language(tree_sitter_rust.language()))

_ITEM_KINDS = {
    "struct_item": "struct",
    "enum_item": "enum",
    "type_item": "type",
    "const_item": "constant",
    "static_item": "constant",
}


def parse_rust(source: bytes) -> ParsedFile:
    """Parse Rust source bytes into the index representation."""
    collector = Collector(source)
    tree = _PARSER.parse(source)
    walker = _Walker(collector)
    walker.walk(tree.root_node)
    walker.finish()
    return collector.result


def _text(node: Node | None) -> str:
    if node is None or node.text is None:
        return ""
    return node.text.decode("utf-8", errors="replace")


def _callee_name(node: Node | None) -> str:
    if node is None:
        return ""
    kind = node.type
    if kind in ("identifier", "field_identifier", "type_identifier"):
        return _text(node)
    if kind == "scoped_identifier":
        return _text(node).rpartition("::")[2]
    if kind == "field_expression":
        return _text(node.child_by_field_name("field"))
    if kind == "generic_function":
        return _callee_name(node.child_by_field_name("function"))
    return ""


def _type_identifier_names(node: Node) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()

    def visit(current: Node) -> None:
        if current.type == "type_identifier":
            text = _text(current)
            if text and text not in seen:
                seen.add(text)
                found.append(text)
            return
        for child in current.named_children:
            visit(child)

    visit(node)
    return found


class _Walker:
    def __init__(self, collector: Collector) -> None:
        self.c = collector
        self.names: list[str] = []
        self.scopes: list[str] = []  # "mod" | "impl" | "trait" | "function"
        self._impl_inherits: list[tuple[str, str]] = []

    @property
    def qualname(self) -> str:
        return "::".join(self.names)

    def finish(self) -> None:
        defined = {symbol.qualname for symbol in self.c.result.symbols}
        for type_name, trait_name in self._impl_inherits:
            origin = type_name if type_name in defined else ""
            self.c.add_ref(origin, trait_name, REF_INHERIT)

    # ------------------------------------------------------------------ walk

    def walk(self, node: Node) -> None:
        kind = node.type
        if kind in ("line_comment", "block_comment"):
            self.c.add_comment(node)
            return
        if kind == "use_declaration":
            self._use(node)
            return
        if kind in ("function_item", "function_signature_item"):
            self._function(node)
            return
        if kind in _ITEM_KINDS:
            self._item(node, _ITEM_KINDS[kind])
            return
        if kind == "trait_item":
            self._trait(node)
            return
        if kind == "mod_item":
            self._module(node)
            return
        if kind == "impl_item":
            self._impl(node)
            return
        if kind == "call_expression":
            self._call(node)
        elif kind == "struct_expression":
            name = _text(node.child_by_field_name("name"))
            if name:
                self.c.add_ref(self.qualname, name, REF_TYPE)

        for child in node.children:
            self.walk(child)

    # -------------------------------------------------------------- definitions

    def _function(self, node: Node) -> None:
        name = _text(node.child_by_field_name("name"))
        if not name:
            return
        kind = "method" if self.scopes and self.scopes[-1] in ("impl", "trait") else "function"
        qualname = "::".join([*self.names, name])
        start = self._attribute_start(node)
        self.c.add_symbol(
            name=name,
            qualname=qualname,
            kind=kind,
            start_byte=start.start_byte,
            end_byte=node.end_byte,
            start_line=node_start_line(start),
            end_line=node_end_line(node),
            signature=self._signature(node),
            docstring=self._docstring(start),
        )
        self._type_refs(node, qualname)
        self.names.append(name)
        self.scopes.append("function")
        for child in node.children:
            self.walk(child)
        self.scopes.pop()
        self.names.pop()

    def _item(self, node: Node, kind: str) -> None:
        name = _text(node.child_by_field_name("name"))
        if not name:
            self._walk_children(node)
            return
        qualname = "::".join([*self.names, name])
        start = self._attribute_start(node)
        self.c.add_symbol(
            name=name,
            qualname=qualname,
            kind=kind,
            start_byte=start.start_byte,
            end_byte=node.end_byte,
            start_line=node_start_line(start),
            end_line=node_end_line(node),
            signature=self._signature(node),
            docstring=self._docstring(start),
        )
        self._type_refs(node, qualname)
        self._walk_children(node)

    def _trait(self, node: Node) -> None:
        name = _text(node.child_by_field_name("name"))
        if not name:
            self._walk_children(node)
            return
        start = self._attribute_start(node)
        self.c.add_symbol(
            name=name,
            qualname="::".join([*self.names, name]),
            kind="trait",
            start_byte=start.start_byte,
            end_byte=node.end_byte,
            start_line=node_start_line(start),
            end_line=node_end_line(node),
            signature=self._signature(node),
            docstring=self._docstring(start),
        )
        self.names.append(name)
        self.scopes.append("trait")
        self._walk_children(node)
        self.scopes.pop()
        self.names.pop()

    def _module(self, node: Node) -> None:
        name = _text(node.child_by_field_name("name"))
        if not name:
            self._walk_children(node)
            return
        start = self._attribute_start(node)
        self.c.add_symbol(
            name=name,
            qualname="::".join([*self.names, name]),
            kind="module",
            start_byte=start.start_byte,
            end_byte=node.end_byte,
            start_line=node_start_line(start),
            end_line=node_end_line(node),
            signature=self._signature(node),
            docstring=self._docstring(start),
        )
        self.names.append(name)
        self.scopes.append("mod")
        self._walk_children(node)
        self.scopes.pop()
        self.names.pop()

    def _impl(self, node: Node) -> None:
        type_node = node.child_by_field_name("type")
        type_name = self._impl_type_name(type_node)
        trait_node = node.child_by_field_name("trait")
        if trait_node is not None and type_name:
            trait_name = self._impl_type_name(trait_node)
            if trait_name:
                self._impl_inherits.append((type_name, trait_name))

        if not type_name:
            self._walk_children(node)
            return
        self.names.append(type_name)
        self.scopes.append("impl")
        self._walk_children(node)
        self.scopes.pop()
        self.names.pop()

    def _impl_type_name(self, node: Node | None) -> str:
        if node is None:
            return ""
        if node.type in ("type_identifier", "identifier"):
            return _text(node)
        if node.type == "generic_type":
            base = node.child_by_field_name("type")
            if base is not None:
                return self._impl_type_name(base)
            for child in node.named_children:
                if child.type == "type_identifier":
                    return _text(child)
        for child in node.named_children:
            name = self._impl_type_name(child)
            if name:
                return name
        return ""

    # ---------------------------------------------------------------- helpers

    def _walk_children(self, node: Node) -> None:
        for child in node.children:
            self.walk(child)

    def _attribute_start(self, node: Node) -> Node:
        start = node
        sibling = node.prev_named_sibling
        while sibling is not None and sibling.type == "attribute_item":
            start = sibling
            sibling = sibling.prev_named_sibling
        return start

    def _signature(self, node: Node) -> str:
        end = node.end_byte
        body = node.child_by_field_name("body")
        if body is not None:
            end = body.start_byte
        raw = self.c.snippet(node.start_byte, end).rstrip()
        if raw.endswith(";"):
            raw = raw[:-1].rstrip()
        return collapse_ws(raw)

    def _docstring(self, start: Node) -> str:
        parts: list[str] = []
        sibling = start.prev_named_sibling
        while sibling is not None:
            if sibling.type == "attribute_item":
                sibling = sibling.prev_named_sibling
                continue
            if sibling.type == "line_comment":
                raw = self.c.node_text(sibling).strip()
                if raw.startswith("///") and not raw.startswith("////"):
                    parts.append(raw[3:].lstrip())
                    sibling = sibling.prev_named_sibling
                    continue
                break
            if sibling.type == "block_comment":
                raw = self.c.node_text(sibling).strip()
                if raw.startswith("/**") and not raw.startswith("/*!"):
                    parts.append(clean_block_doc(raw))
                    sibling = sibling.prev_named_sibling
                    continue
                break
            break
        return "\n".join(reversed(parts))

    def _type_refs(self, node: Node, qualname: str) -> None:
        nodes: list[Node] = []
        params = node.child_by_field_name("parameters")
        if params is not None:
            for param in params.named_children:
                annotation = param.child_by_field_name("type")
                if annotation is not None:
                    nodes.append(annotation)
        return_type = node.child_by_field_name("return_type")
        if return_type is not None:
            nodes.append(return_type)
        type_field = node.child_by_field_name("type")
        if type_field is not None and node.type != "function_item":
            nodes.append(type_field)
        if node.type == "type_item":
            value = node.child_by_field_name("value")
            if value is not None:
                nodes.append(value)
        body = node.child_by_field_name("body")
        if node.type == "struct_item" and body is not None:
            for field in body.named_children:
                if field.type == "field_declaration":
                    annotation = field.child_by_field_name("type")
                    if annotation is not None:
                        nodes.append(annotation)

        seen: set[str] = set()
        for annotation in nodes:
            for type_name in _type_identifier_names(annotation):
                if type_name in seen:
                    continue
                seen.add(type_name)
                self.c.add_ref(qualname, type_name, REF_TYPE)

    # ---------------------------------------------------------------- imports

    def _use(self, node: Node) -> None:
        argument = node.child_by_field_name("argument")
        if argument is None:
            return
        for module, local, original in self._expand_use(argument):
            if local:
                self.c.add_import(module, local)
            if original and original not in ("self", "*"):
                self.c.add_ref(self.qualname, original, REF_IMPORT)

    def _expand_use(self, node: Node) -> list[tuple[str, str, str]]:
        kind = node.type
        if kind == "scoped_use_list":
            path = node.child_by_field_name("path")
            items = node.child_by_field_name("list")
            if items is None:
                items = next((c for c in node.named_children if c.type == "use_list"), None)
            prefix = _text(path)
            return self._prefix_use(prefix, self._expand_use(items) if items else [])
        if kind == "use_list":
            entries: list[tuple[str, str, str]] = []
            for child in node.named_children:
                entries.extend(self._expand_use(child))
            return entries
        if kind == "use_as_clause":
            path = node.child_by_field_name("path")
            alias = node.child_by_field_name("alias")
            entries = self._expand_use(path) if path is not None else []
            if not entries:
                return []
            module, _local, original = entries[0]
            return [(module, _text(alias) or original, original)]
        if kind == "scoped_identifier":
            text = _text(node)
            module, _, name = text.rpartition("::")
            return [(module, name, name)]
        if kind == "identifier":
            text = _text(node)
            return [("", text, text)]
        if kind == "self":
            return [("", "", "self")]
        if kind in ("crate", "super", "metavariable", "use_wildcard"):
            text = _text(node)
            if text.endswith("::*"):
                return [(text[:-3], "*", "*")]
            return [("", "", text)]
        return []

    def _prefix_use(
        self, prefix: str, entries: list[tuple[str, str, str]]
    ) -> list[tuple[str, str, str]]:
        out: list[tuple[str, str, str]] = []
        for module, local, original in entries:
            full = f"{prefix}::{module}" if module else prefix
            if not local:
                local = ""
            out.append((full, local, original))
        return out

    # ------------------------------------------------------------------ calls

    def _call(self, node: Node) -> None:
        name = _callee_name(node.child_by_field_name("function"))
        if name:
            self.c.add_ref(self.qualname, name, REF_CALL)
