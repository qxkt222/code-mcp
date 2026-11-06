"""TypeScript / JavaScript / TSX front end.

The same grammar parses plain JavaScript; ``tsx`` is used for JSX-bearing
files. ``export`` wrappers are transparent for symbol spans, while
``export default`` and arrow functions assigned to ``const`` are indexed as
functions.
"""

from __future__ import annotations

import tree_sitter_typescript
from tree_sitter import Language, Node, Parser

from ...types import REF_CALL, REF_IMPORT, REF_INHERIT, REF_TYPE
from . import (
    Collector,
    ParsedFile,
    clean_block_doc,
    collapse_ws,
    node_end_line,
    node_start_line,
    strip_quotes,
)

_TS_PARSER = Parser(Language(tree_sitter_typescript.language_typescript()))
_TSX_PARSER = Parser(Language(tree_sitter_typescript.language_tsx()))

_FUNCTION_VALUES = {"arrow_function", "function_expression", "function", "generator_function"}
_CLASS_VALUES = {"class", "class_expression"}
# Anonymous function forms create a variable scope but no symbol of their own.
_SCOPE_ONLY_VALUES = {"arrow_function", "function_expression", "generator_function"}

# Signatures are summaries: a long initializer is elided and the whole string
# is capped. Short initializers stay so ``const MAX = 10`` still reads whole.
_MAX_SIGNATURE = 200
_SHORT_INITIALIZER = 80


def _clip_signature(text: str) -> str:
    """Collapse whitespace and cap the signature at 200 characters."""
    text = collapse_ws(text)
    if len(text) <= _MAX_SIGNATURE:
        return text
    return text[: _MAX_SIGNATURE - 3].rstrip() + "..."


def parse_typescript(source: bytes, lang: str = "typescript") -> ParsedFile:
    """Parse TypeScript/JavaScript/TSX source bytes into the index form."""
    collector = Collector(source)
    parser = _TSX_PARSER if lang == "tsx" else _TS_PARSER
    tree = parser.parse(source)
    _Walker(collector).walk(tree.root_node)
    for symbol in collector.result.symbols:
        symbol.signature = _clip_signature(symbol.signature)
    return collector.result


def _text(node: Node | None) -> str:
    if node is None or node.text is None:
        return ""
    return node.text.decode("utf-8", errors="replace")


def _callee_name(node: Node | None) -> str:
    if node is None:
        return ""
    if node.type == "identifier":
        return _text(node)
    if node.type == "member_expression":
        return _text(node.child_by_field_name("property"))
    if node.type == "non_null_expression":
        return _callee_name(node.named_children[0] if node.named_children else None)
    if node.type in ("parenthesized_expression", "call_expression", "super"):
        return ""
    return ""


class _Walker:
    def __init__(self, collector: Collector) -> None:
        self.c = collector
        self.names: list[str] = []
        self.scopes: list[str] = []  # "class" | "interface" | "function"
        # Qualname path of the innermost container (symbol or object literal
        # key); used to name methods found inside object literals.
        self.containers: list[str] = []
        # Qualname of the innermost real symbol; calls and `new` expressions
        # are credited to it so refs stay attached to store actions.
        self.owners: list[str] = []

    @property
    def qualname(self) -> str:
        return ".".join(self.names)

    @property
    def ref_qualname(self) -> str:
        return self.owners[-1] if self.owners else ""

    # ------------------------------------------------------------------ walk

    def walk(self, node: Node) -> None:
        kind = node.type
        if kind == "comment":
            self.c.add_comment(node)
            return
        if kind == "import_statement":
            self._import(node)
            return
        if kind == "export_statement":
            self._export(node)
            return
        if kind in ("function_declaration", "generator_function_declaration"):
            self._function(node)
            return
        if kind in ("class_declaration", "abstract_class_declaration"):
            self._class(node)
            return
        if kind == "interface_declaration":
            self._interface(node)
            return
        if kind == "type_alias_declaration":
            self._type_alias(node)
            return
        if kind == "enum_declaration":
            self._enum(node)
            return
        if kind in ("lexical_declaration", "variable_declaration"):
            self._variables(node)
            return
        if kind == "method_definition":
            self._method(node)
            return
        if kind == "method_signature":
            self._method_signature(node)
            return
        if kind in _SCOPE_ONLY_VALUES:
            self._scope_only(node)
            return
        if kind == "pair":
            self._pair(node)
            return
        if kind == "call_expression":
            self._call(node)
        elif kind == "new_expression":
            self._new(node)

        for child in node.children:
            self.walk(child)

    # ---------------------------------------------------------------- exports

    def _export(self, node: Node) -> None:
        declaration = node.child_by_field_name("declaration")
        value = node.child_by_field_name("value")
        source = node.child_by_field_name("source")
        is_default = any(child.type == "default" for child in node.children)
        prefix = "export default " if is_default else "export "

        # Decorators on an exported declaration live on the export statement.
        for child in node.children:
            if child.type == "decorator":
                self.walk(child)

        doc = self._leading_jsdoc(node)

        if declaration is not None:
            self._declaration(declaration, prefix, doc)
        elif value is not None and value.type in _FUNCTION_VALUES:
            self._anon_default(value, node, doc)
        elif source is not None:
            self._export_from(node, source)

    def _declaration(self, node: Node, prefix: str, doc: str) -> None:
        kind = node.type
        if kind in ("function_declaration", "generator_function_declaration"):
            self._function(node, prefix, doc)
        elif kind in ("class_declaration", "abstract_class_declaration"):
            self._class(node, prefix, doc)
        elif kind == "interface_declaration":
            self._interface(node, prefix, doc)
        elif kind == "type_alias_declaration":
            self._type_alias(node, prefix, doc)
        elif kind == "enum_declaration":
            self._enum(node, prefix, doc)
        elif kind in ("lexical_declaration", "variable_declaration"):
            self._variables(node, prefix, doc)
        else:
            self.walk(node)

    def _export_from(self, node: Node, source: Node) -> None:
        module = strip_quotes(_text(source))
        clause = next((c for c in node.named_children if c.type == "export_clause"), None)
        if clause is None:
            self.c.add_import(module, "*")
            return
        for spec in clause.named_children:
            if spec.type != "export_specifier":
                continue
            name = _text(spec.child_by_field_name("name"))
            alias = _text(spec.child_by_field_name("alias"))
            self.c.add_import(module, alias or name)
            if name:
                self.c.add_ref(self.qualname, name, REF_IMPORT)

    # ------------------------------------------------------------ definitions

    def _function(self, node: Node, prefix: str = "", doc: str | None = None) -> None:
        name = _text(node.child_by_field_name("name"))
        if not name:
            return
        qualname = ".".join([*self.names, name])
        body = node.child_by_field_name("body")
        end = body.start_byte if body is not None else node.end_byte
        signature = prefix + collapse_ws(self.c.snippet(self._keyword_start(node), end))
        self.c.add_symbol(
            name=name,
            qualname=qualname,
            kind="function",
            start_byte=node.start_byte,
            end_byte=node.end_byte,
            start_line=node_start_line(node),
            end_line=node_end_line(node),
            signature=signature,
            docstring=self._leading_jsdoc(node) if doc is None else doc,
        )
        self._push(name, "function")
        self._type_refs(node, qualname)
        for child in node.children:
            self.walk(child)
        self._pop()

    def _anon_default(self, value: Node, export_node: Node, doc: str) -> None:
        name = "default"
        qualname = ".".join([*self.names, name])
        body = value.child_by_field_name("body")
        end = body.start_byte if body is not None else value.end_byte
        signature = "export default " + collapse_ws(self.c.snippet(self._keyword_start(value), end))
        self.c.add_symbol(
            name=name,
            qualname=qualname,
            kind="function",
            start_byte=value.start_byte,
            end_byte=value.end_byte,
            start_line=node_start_line(value),
            end_line=node_end_line(value),
            signature=signature,
            docstring=doc or self._leading_jsdoc(export_node),
        )
        self._push(name, "function")
        self._type_refs(value, qualname)
        for child in value.children:
            self.walk(child)
        self._pop()

    def _class(self, node: Node, prefix: str = "", doc: str | None = None) -> None:
        name = _text(node.child_by_field_name("name")) or "default"
        qualname = ".".join([*self.names, name])
        body = node.child_by_field_name("body")
        end = body.start_byte if body is not None else node.end_byte
        signature = prefix + collapse_ws(self.c.snippet(self._keyword_start(node), end))
        start = self._decorator_start(node)
        self.c.add_symbol(
            name=name,
            qualname=qualname,
            kind="class",
            start_byte=start.start_byte,
            end_byte=node.end_byte,
            start_line=node_start_line(start),
            end_line=node_end_line(node),
            signature=signature,
            docstring=self._leading_jsdoc(start if start is not node else node) if doc is None else doc,
        )
        self._heritage(node, qualname)

        for child in node.children:
            if child.type == "decorator":
                self.walk(child)
        self._push(name, "class")
        for child in node.children:
            if child.type == "decorator":
                continue
            self.walk(child)
        self._pop()

    def _interface(self, node: Node, prefix: str = "", doc: str | None = None) -> None:
        name = _text(node.child_by_field_name("name"))
        if not name:
            return
        qualname = ".".join([*self.names, name])
        body = node.child_by_field_name("body")
        end = body.start_byte if body is not None else node.end_byte
        signature = prefix + collapse_ws(self.c.snippet(self._keyword_start(node), end))
        self.c.add_symbol(
            name=name,
            qualname=qualname,
            kind="interface",
            start_byte=node.start_byte,
            end_byte=node.end_byte,
            start_line=node_start_line(node),
            end_line=node_end_line(node),
            signature=signature,
            docstring=self._leading_jsdoc(node) if doc is None else doc,
        )
        for clause in node.named_children:
            if clause.type != "extends_type_clause":
                continue
            for type_node in clause.children_by_field_name("type"):
                self.c.add_ref(qualname, _text(type_node), REF_INHERIT)

        self._push(name, "interface")
        for child in node.children:
            self.walk(child)
        self._pop()

    def _type_alias(self, node: Node, prefix: str = "", doc: str | None = None) -> None:
        name = _text(node.child_by_field_name("name"))
        if not name:
            return
        qualname = ".".join([*self.names, name])
        signature = prefix + collapse_ws(self.c.node_text(node).rstrip().rstrip(";"))
        self.c.add_symbol(
            name=name,
            qualname=qualname,
            kind="type",
            start_byte=node.start_byte,
            end_byte=node.end_byte,
            start_line=node_start_line(node),
            end_line=node_end_line(node),
            signature=signature,
            docstring=self._leading_jsdoc(node) if doc is None else doc,
        )
        value = node.child_by_field_name("value")
        if value is not None:
            for type_name in self._type_names(value):
                self.c.add_ref(qualname, type_name, REF_TYPE)

    def _enum(self, node: Node, prefix: str = "", doc: str | None = None) -> None:
        name = _text(node.child_by_field_name("name"))
        if not name:
            return
        body = node.child_by_field_name("body")
        end = body.start_byte if body is not None else node.end_byte
        signature = prefix + collapse_ws(self.c.snippet(self._keyword_start(node), end))
        self.c.add_symbol(
            name=name,
            qualname=".".join([*self.names, name]),
            kind="enum",
            start_byte=node.start_byte,
            end_byte=node.end_byte,
            start_line=node_start_line(node),
            end_line=node_end_line(node),
            signature=signature,
            docstring=self._leading_jsdoc(node) if doc is None else doc,
        )
        self._walk_children(node)

    def _variables(self, node: Node, prefix: str = "", doc: str | None = None) -> None:
        keyword = next(
            (child.type for child in node.children if child.type in ("const", "let", "var")),
            "const",
        )
        declarators = [c for c in node.named_children if c.type == "variable_declarator"]
        for index, declarator in enumerate(declarators):
            name_node = declarator.child_by_field_name("name")
            if name_node is None or name_node.type != "identifier":
                self._walk_children(declarator)
                continue
            name = _text(name_node)
            value = declarator.child_by_field_name("value")
            value_kind = value.type if value is not None else ""

            if value_kind in _FUNCTION_VALUES:
                self._value_function(node, declarator, value, name, keyword, prefix, doc if index == 0 else "")
            elif value_kind in _CLASS_VALUES:
                self._value_class(declarator, value, name, keyword, prefix)
            elif keyword == "const" and not self.scopes:
                signature = f"{prefix}{keyword} " + collapse_ws(
                    self._initializer_signature(declarator, value)
                )
                self.c.add_symbol(
                    name=name,
                    qualname=name,
                    kind="constant",
                    start_byte=declarator.start_byte,
                    end_byte=declarator.end_byte,
                    start_line=node_start_line(declarator),
                    end_line=node_end_line(declarator),
                    signature=signature,
                    docstring=(doc or "") if index == 0 else "",
                )
                annotation = declarator.child_by_field_name("type")
                if annotation is not None:
                    for type_name in self._type_names(annotation):
                        self.c.add_ref("", type_name, REF_TYPE)
                if value is not None:
                    # The const is a symbol: object-literal methods inside its
                    # initializer are named ``<const>.<key>``.
                    self.containers.append(name)
                    self.owners.append(name)
                    try:
                        self.walk(value)
                    finally:
                        self.owners.pop()
                        self.containers.pop()
            else:
                self._walk_children(declarator)

    def _initializer_signature(self, declarator: Node, value: Node | None) -> str:
        """Declarator text up to the initializer.

        A short initializer (< 80 chars collapsed) is kept whole; a long one is
        elided so an object literal cannot leak into the indexed signature.
        """
        end = declarator.end_byte
        if value is not None:
            keep = len(collapse_ws(self.c.node_text(value))) < _SHORT_INITIALIZER
            end = value.end_byte if keep else value.start_byte
        return self.c.snippet(declarator.start_byte, end).rstrip().rstrip(";").rstrip()

    def _value_function(
        self,
        declaration: Node,
        declarator: Node,
        value: Node,
        name: str,
        keyword: str,
        prefix: str,
        doc: str,
    ) -> None:
        qualname = ".".join([*self.names, name])
        body = value.child_by_field_name("body")
        end = body.start_byte if body is not None else value.end_byte
        signature = f"{prefix}{keyword} " + collapse_ws(
            self.c.snippet(declarator.start_byte, end)
        )
        self.c.add_symbol(
            name=name,
            qualname=qualname,
            kind="function",
            start_byte=declarator.start_byte,
            end_byte=value.end_byte,
            start_line=node_start_line(declarator),
            end_line=node_end_line(value),
            signature=signature,
            docstring=doc or self._leading_jsdoc(declaration),
        )
        self._push(name, "function")
        self._type_refs(value, qualname)
        for child in value.children:
            self.walk(child)
        self._pop()

    def _value_class(
        self,
        declarator: Node,
        value: Node,
        name: str,
        keyword: str,
        prefix: str,
    ) -> None:
        body = value.child_by_field_name("body")
        end = body.start_byte if body is not None else value.end_byte
        signature = f"{prefix}{keyword} " + collapse_ws(
            self.c.snippet(declarator.start_byte, end)
        )
        qualname = ".".join([*self.names, name])
        self.c.add_symbol(
            name=name,
            qualname=qualname,
            kind="class",
            start_byte=declarator.start_byte,
            end_byte=value.end_byte,
            start_line=node_start_line(declarator),
            end_line=node_end_line(value),
            signature=signature,
            docstring=self._leading_jsdoc(declarator),
        )
        self._push(name, "class")
        for child in value.children:
            self.walk(child)
        self._pop()

    def _method(self, node: Node, prefix: str = "") -> None:
        name = _text(node.child_by_field_name("name"))
        if not name:
            return
        qualname = ".".join([*self.names, name])
        body = node.child_by_field_name("body")
        end = body.start_byte if body is not None else node.end_byte
        signature = prefix + collapse_ws(self.c.snippet(node.start_byte, end))
        start = self._decorator_start(node)
        self.c.add_symbol(
            name=name,
            qualname=qualname,
            kind="method",
            start_byte=start.start_byte,
            end_byte=node.end_byte,
            start_line=node_start_line(start),
            end_line=node_end_line(node),
            signature=signature.rstrip(";").rstrip(),
            docstring=self._leading_jsdoc(start),
        )
        for sibling in self._decorators(node):
            self.walk(sibling)
        self._push(name, "function")
        self._type_refs(node, qualname)
        for child in node.children:
            self.walk(child)
        self._pop()

    def _method_signature(self, node: Node) -> None:
        name = _text(node.child_by_field_name("name"))
        if not name:
            return
        qualname = ".".join([*self.names, name])
        signature = collapse_ws(self.c.node_text(node).rstrip().rstrip(";"))
        self.c.add_symbol(
            name=name,
            qualname=qualname,
            kind="method",
            start_byte=node.start_byte,
            end_byte=node.end_byte,
            start_line=node_start_line(node),
            end_line=node_end_line(node),
            signature=signature,
            docstring=self._leading_jsdoc(node),
        )
        self._type_refs(node, qualname)
        self._push(name, "function")
        for child in node.children:
            self.walk(child)
        self._pop()

    def _scope_only(self, node: Node) -> None:
        """Walk an anonymous function body without giving it a name.

        ``const`` declarations inside callbacks must not surface as
        module-level constants, but the callback itself has no name to prefix
        them with, so only ``scopes`` is pushed.
        """
        self.scopes.append("function")
        try:
            for child in node.children:
                self.walk(child)
        finally:
            self.scopes.pop()

    def _pair(self, node: Node) -> None:
        """Index a function-valued object-literal member as a method.

        Store actions (``create((set) => ({ sendMessage: ... }))``) become
        addressable symbols. The span covers the whole ``pair`` so an edit can
        replace the member in place; nested object literals contribute their
        key to the qualname of deeper methods.
        """
        key = node.child_by_field_name("key")
        value = node.child_by_field_name("value")
        if key is None or value is None:
            self._walk_children(node)
            return
        if key.type not in ("property_identifier", "identifier"):
            self._walk_children(node)
            return

        name = _text(key)
        parent = self.containers[-1] if self.containers else ""
        qualname = f"{parent}.{name}" if parent else name

        if value.type in _FUNCTION_VALUES:
            body = value.child_by_field_name("body")
            end = body.start_byte if body is not None else value.end_byte
            signature = collapse_ws(self.c.snippet(node.start_byte, end))
            signature = signature.rstrip("=>").rstrip()
            self.c.add_symbol(
                name=name,
                qualname=qualname,
                kind="method",
                start_byte=node.start_byte,
                end_byte=node.end_byte,
                start_line=node_start_line(node),
                end_line=node_end_line(node),
                signature=signature,
                docstring=self._leading_jsdoc(node),
            )
            self._type_refs(value, qualname)
            self.containers.append(qualname)
            self.owners.append(qualname)
            try:
                self.walk(value)
            finally:
                self.owners.pop()
                self.containers.pop()
            return

        # A plain object value still contributes its key to the qualname of
        # any methods nested one level deeper.
        self.containers.append(qualname)
        try:
            self._walk_children(node)
        finally:
            self.containers.pop()

    # ---------------------------------------------------------------- helpers

    def _push(self, name: str, scope: str) -> None:
        self.names.append(name)
        self.scopes.append(scope)
        qualname = ".".join(self.names)
        self.containers.append(qualname)
        self.owners.append(qualname)

    def _pop(self) -> None:
        self.names.pop()
        self.scopes.pop()
        self.containers.pop()
        self.owners.pop()

    def _walk_children(self, node: Node) -> None:
        for child in node.children:
            self.walk(child)

    def _keyword_start(self, node: Node) -> int:
        """Byte offset of the declaration keyword, skipping decorators."""
        for child in node.children:
            if child.type != "decorator":
                return child.start_byte
        return node.start_byte

    def _decorators(self, node: Node) -> list[Node]:
        found: list[Node] = []
        sibling = node.prev_named_sibling
        while sibling is not None and sibling.type == "decorator":
            found.append(sibling)
            sibling = sibling.prev_named_sibling
        found.reverse()
        return found

    def _decorator_start(self, node: Node) -> Node:
        decorators = self._decorators(node)
        return decorators[0] if decorators else node

    def _leading_jsdoc(self, node: Node) -> str:
        parts: list[str] = []
        sibling = node.prev_named_sibling
        while sibling is not None:
            if sibling.type == "decorator":
                sibling = sibling.prev_named_sibling
                continue
            if sibling.type == "comment":
                raw = self.c.node_text(sibling)
                if raw.startswith("/**") and not raw.startswith("/***"):
                    parts.append(clean_block_doc(raw))
                    sibling = sibling.prev_named_sibling
                    continue
            break
        return "\n".join(reversed(parts))

    def _heritage(self, node: Node, qualname: str) -> None:
        for child in node.named_children:
            if child.type != "class_heritage":
                continue
            for clause in child.named_children:
                if clause.type == "extends_clause":
                    value = clause.child_by_field_name("value")
                    name = _callee_name(value)
                    if name:
                        self.c.add_ref(qualname, name, REF_INHERIT)
                elif clause.type == "implements_clause":
                    for type_node in clause.named_children:
                        if type_node.type == "type_identifier":
                            self.c.add_ref(qualname, _text(type_node), REF_INHERIT)

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
        seen: set[str] = set()
        for annotation in nodes:
            for type_name in self._type_names(annotation):
                if type_name in seen:
                    continue
                seen.add(type_name)
                self.c.add_ref(qualname, type_name, REF_TYPE)

    def _type_names(self, node: Node) -> list[str]:
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

    # ---------------------------------------------------------------- imports

    def _import(self, node: Node) -> None:
        source = node.child_by_field_name("source")
        module = strip_quotes(_text(source)) if source is not None else ""
        clause = next((c for c in node.named_children if c.type == "import_clause"), None)
        if clause is None:
            self.c.add_import(module, "")
            return
        for part in clause.named_children:
            if part.type == "identifier":
                name = _text(part)
                self.c.add_import(module, name)
                self.c.add_ref(self.qualname, name, REF_IMPORT)
            elif part.type == "namespace_import":
                name = _text(part.named_children[0]) if part.named_children else ""
                self.c.add_import(module, name)
            elif part.type == "named_imports":
                for spec in part.named_children:
                    if spec.type != "import_specifier":
                        continue
                    name = _text(spec.child_by_field_name("name"))
                    alias = _text(spec.child_by_field_name("alias"))
                    self.c.add_import(module, alias or name)
                    if name:
                        self.c.add_ref(self.qualname, name, REF_IMPORT)

    # ------------------------------------------------------------------ calls

    def _call(self, node: Node) -> None:
        name = _callee_name(node.child_by_field_name("function"))
        if name:
            self.c.add_ref(self.ref_qualname, name, REF_CALL)

    def _new(self, node: Node) -> None:
        name = _callee_name(node.child_by_field_name("constructor"))
        if name:
            self.c.add_ref(self.ref_qualname, name, REF_TYPE)
