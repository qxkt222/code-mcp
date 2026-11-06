"""Python front end: symbols, imports, comments and call sites."""

from __future__ import annotations

import ast

import tree_sitter_python
from tree_sitter import Language, Node, Parser

from ...types import REF_CALL, REF_IMPORT, REF_INHERIT, REF_TYPE
from . import Collector, ParsedFile, collapse_ws, node_end_line, node_start_line

_PARSER = Parser(Language(tree_sitter_python.language()))


def parse_python(source: bytes) -> ParsedFile:
    """Parse Python source bytes into the index representation."""
    collector = Collector(source)
    tree = _PARSER.parse(source)
    _Walker(collector).walk(tree.root_node)
    return collector.result


def _expr_name(node: Node | None) -> str:
    """Last component of an expression usable as a reference name."""
    if node is None:
        return ""
    if node.type == "identifier":
        return node.text.decode("utf-8", errors="replace") if node.text else ""
    if node.type == "attribute":
        return _node_name(node.child_by_field_name("attribute"))
    if node.type == "keyword_argument":
        return ""
    return ""


def _node_name(node: Node | None) -> str:
    if node is None or node.text is None:
        return ""
    return node.text.decode("utf-8", errors="replace")


class _Walker:
    def __init__(self, collector: Collector) -> None:
        self.c = collector
        self.names: list[str] = []
        self.scopes: list[str] = []  # "class" | "function"

    @property
    def qualname(self) -> str:
        return ".".join(self.names)

    # ------------------------------------------------------------------ walk

    def walk(self, node: Node) -> None:
        kind = node.type
        if kind == "decorated_definition":
            self._decorated(node)
            return
        if kind == "function_definition":
            self._definition(node, node)
            return
        if kind == "class_definition":
            self._definition(node, node)
            return
        if kind == "import_statement":
            self._import(node)
            return
        if kind == "import_from_statement":
            self._import_from(node)
            return
        if kind == "comment":
            self.c.add_comment(node)
            return
        if kind == "call":
            self._call(node)
        elif kind == "assignment":
            self._assignment(node)

        for child in node.children:
            self.walk(child)

    # ------------------------------------------------------------- definitions

    def _decorated(self, node: Node) -> None:
        # Decorator expressions are evaluated in the enclosing scope.
        for child in node.children:
            if child.type == "decorator":
                self.walk(child)
        inner = node.child_by_field_name("definition")
        if inner is not None:
            self._definition(inner, node)

    def _definition(self, node: Node, span: Node) -> None:
        name_node = node.child_by_field_name("name")
        name = _node_name(name_node)
        if not name:
            return

        is_class = node.type == "class_definition"
        if is_class:
            kind = "class"
        elif self.scopes and self.scopes[-1] == "class":
            kind = "method"
        else:
            kind = "function"

        qualname = ".".join([*self.names, name])
        self.c.add_symbol(
            name=name,
            qualname=qualname,
            kind=kind,
            start_byte=span.start_byte,
            end_byte=span.end_byte,
            start_line=node_start_line(span),
            end_line=node_end_line(span),
            signature=self._signature(node),
            docstring=self._docstring(node),
        )

        if is_class:
            self._inheritance(node, qualname)

        self.names.append(name)
        self.scopes.append("class" if is_class else "function")

        if not is_class:
            self._type_refs(node, qualname)

        for child in node.children:
            self.walk(child)

        self.scopes.pop()
        self.names.pop()

    def _inheritance(self, node: Node, qualname: str) -> None:
        supers = node.child_by_field_name("superclasses")
        if supers is None:
            return
        for arg in supers.named_children:
            if arg.type == "keyword_argument":
                continue
            name = _expr_name(arg)
            if name:
                self.c.add_ref(qualname, name, REF_INHERIT)

    def _type_refs(self, node: Node, qualname: str) -> None:
        params = node.child_by_field_name("parameters")
        if params is not None:
            for param in params.named_children:
                annotation = param.child_by_field_name("type")
                if annotation is not None:
                    for type_name in self._type_names(annotation):
                        self.c.add_ref(qualname, type_name, REF_TYPE)
        annotation = node.child_by_field_name("return_type")
        if annotation is not None:
            for type_name in self._type_names(annotation):
                self.c.add_ref(qualname, type_name, REF_TYPE)

    def _type_names(self, node: Node) -> list[str]:
        found: list[str] = []
        seen: set[str] = set()

        def visit(current: Node) -> None:
            if current.type == "identifier":
                text = _node_name(current)
                if text and text not in seen:
                    seen.add(text)
                    found.append(text)
                return
            if current.type == "attribute":
                text = _node_name(current.child_by_field_name("attribute"))
                if text and text not in seen:
                    seen.add(text)
                    found.append(text)
                return
            for child in current.named_children:
                visit(child)

        visit(node)
        return found

    def _docstring(self, node: Node) -> str:
        body = node.child_by_field_name("body")
        if body is None:
            return ""
        first = next((child for child in body.named_children if child.type != "comment"), None)
        if first is None or first.type != "expression_statement" or first.named_child_count != 1:
            return ""
        expr = first.named_children[0]
        if expr.type not in ("string", "concatenated_string"):
            return ""
        try:
            value = ast.literal_eval(self.c.node_text(expr))
        except (ValueError, SyntaxError):
            return ""
        return value if isinstance(value, str) else ""

    def _signature(self, node: Node) -> str:
        end = node.end_byte
        body = node.child_by_field_name("body")
        if body is not None:
            end = body.start_byte
        raw = self.c.snippet(node.start_byte, end).rstrip()
        if raw.endswith(":"):
            raw = raw[:-1].rstrip()
        return collapse_ws(raw)

    # ----------------------------------------------------------------- imports

    def _import(self, node: Node) -> None:
        for name_node in node.children_by_field_name("name"):
            if name_node.type == "aliased_import":
                target = name_node.child_by_field_name("name")
                alias = name_node.child_by_field_name("alias")
                module = _node_name(target)
                local = _node_name(alias) or module
                if module:
                    self.c.add_import(module, local)
                    self.c.add_ref(self.qualname, module.rsplit(".", 1)[-1], REF_IMPORT)
            else:
                module = _node_name(name_node)
                if module:
                    self.c.add_import(module, "")
                    self.c.add_ref(self.qualname, module.rsplit(".", 1)[-1], REF_IMPORT)

    def _import_from(self, node: Node) -> None:
        module_node = node.child_by_field_name("module_name")
        module = _node_name(module_node)
        for name_node in node.children_by_field_name("name"):
            if name_node.type == "aliased_import":
                target = name_node.child_by_field_name("name")
                alias = name_node.child_by_field_name("alias")
                original = _node_name(target).rsplit(".", 1)[-1]
                local = _node_name(alias) or original
                self.c.add_import(module, local)
                self.c.add_ref(self.qualname, original, REF_IMPORT)
            elif name_node.type == "wildcard_import":
                self.c.add_import(module, "*")
            else:
                imported = _node_name(name_node)
                original = imported.rsplit(".", 1)[-1]
                self.c.add_import(module, imported)
                self.c.add_ref(self.qualname, original, REF_IMPORT)

    # ------------------------------------------------------------- statements

    def _assignment(self, node: Node) -> None:
        scope = self.scopes[-1] if self.scopes else "module"
        if scope == "function":
            return
        left = node.child_by_field_name("left")
        if left is None or left.type != "identifier":
            return
        name = _node_name(left)
        if not name or not name.strip("_").isupper():
            return
        qualname = ".".join([*self.names, name])
        self.c.add_symbol(
            name=name,
            qualname=qualname,
            kind="constant",
            start_byte=node.start_byte,
            end_byte=node.end_byte,
            start_line=node_start_line(node),
            end_line=node_end_line(node),
            signature=collapse_ws(self.c.node_text(node)),
        )
        annotation = node.child_by_field_name("type")
        if annotation is not None:
            for type_name in self._type_names(annotation):
                self.c.add_ref(self.qualname, type_name, REF_TYPE)

    def _call(self, node: Node) -> None:
        name = _expr_name(node.child_by_field_name("function"))
        if name:
            self.c.add_ref(self.qualname, name, REF_CALL)
