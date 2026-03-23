"""
Multi-language AST parser using tree-sitter >= 0.23.
Provides structured extraction of functions, classes, and imports.
"""
from dataclasses import dataclass, field
from typing import Optional

from tree_sitter import Language, Parser, Node

from logger import logger


@dataclass
class ExtractedNode:
    """A parsed code element (function, class, method, import)."""
    name: str
    node_type: str              # function, class, method, import
    text: bytes
    line_start: int
    line_end: int
    signature: str = ""         # e.g. "def foo(a, b) -> int"
    class_name: str | None = None
    children: list["ExtractedNode"] = field(default_factory=list)


# Maps file extension → (grammar module import path, language loader function name)
_LANGUAGE_REGISTRY: dict[str, tuple[str, str]] = {
    ".py":   ("tree_sitter_python",     "language"),
    ".js":   ("tree_sitter_javascript",  "language"),
    ".jsx":  ("tree_sitter_javascript",  "language"),
    ".mjs":  ("tree_sitter_javascript",  "language"),
    ".cjs":  ("tree_sitter_javascript",  "language"),
    ".ts":   ("tree_sitter_typescript",  "language_typescript"),
    ".mts":  ("tree_sitter_typescript",  "language_typescript"),
    ".cts":  ("tree_sitter_typescript",  "language_typescript"),
    ".tsx":  ("tree_sitter_typescript",  "language_tsx"),
    ".go":   ("tree_sitter_go",          "language"),
    ".java": ("tree_sitter_java",        "language"),
    ".rs":   ("tree_sitter_rust",        "language"),
    ".c":    ("tree_sitter_c",           "language"),
    ".h":    ("tree_sitter_c",           "language"),
    ".cpp":  ("tree_sitter_cpp",         "language"),
    ".cc":   ("tree_sitter_cpp",         "language"),
    ".cxx":  ("tree_sitter_cpp",         "language"),
    ".hpp":  ("tree_sitter_cpp",         "language"),
    ".hh":   ("tree_sitter_cpp",         "language"),
}

# Extension → language name mapping
EXTENSION_TO_LANG_NAME: dict[str, str] = {
    ".py": "python", ".js": "javascript", ".jsx": "javascript",
    ".mjs": "javascript", ".cjs": "javascript",
    ".ts": "typescript", ".mts": "typescript", ".cts": "typescript",
    ".tsx": "tsx", ".go": "go", ".java": "java", ".rs": "rust",
    ".c": "c", ".h": "c", ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp",
    ".hpp": "cpp", ".hh": "cpp",
}

# Node types to extract per language
_FUNCTION_TYPES: dict[str, set[str]] = {
    "python":     {"function_definition"},
    "javascript": {"function_declaration", "arrow_function", "method_definition"},
    "typescript": {"function_declaration", "arrow_function", "method_definition"},
    "tsx":        {"function_declaration", "arrow_function", "method_definition"},
    "go":         {"function_declaration", "method_declaration"},
    "java":       {"method_declaration", "constructor_declaration"},
    "rust":       {"function_item"},
    "c":          {"function_definition"},
    "cpp":        {"function_definition"},
}

_CLASS_TYPES: dict[str, set[str]] = {
    "python":     {"class_definition"},
    "javascript": {"class_declaration"},
    "typescript": {"class_declaration", "interface_declaration"},
    "tsx":        {"class_declaration", "interface_declaration"},
    "go":         {"type_declaration"},
    "java":       {"class_declaration", "interface_declaration"},
    "rust":       {"impl_item", "struct_item", "trait_item"},
    "c":          {"struct_specifier"},
    "cpp":        {"class_specifier", "struct_specifier"},
}

_IMPORT_TYPES: dict[str, set[str]] = {
    "python":     {"import_statement", "import_from_statement"},
    "javascript": {"import_statement"},
    "typescript": {"import_statement"},
    "tsx":        {"import_statement"},
    "go":         {"import_declaration"},
    "java":       {"import_declaration"},
    "rust":       {"use_declaration"},
    "c":          {"preproc_include"},
    "cpp":        {"preproc_include", "using_declaration"},
}


class TreeSitterService:
    """Multi-language AST parser using tree-sitter."""

    def __init__(self):
        self._parsers: dict[str, Parser] = {}
        self._languages: dict[str, Language] = {}

    def get_language(self, extension: str) -> Optional[Language]:
        """
        Get Language object for a file extension.

        Returns:
            Language object or None if unsupported
        """
        ext = extension.lower()
        if ext in self._languages:
            return self._languages[ext]

        if ext not in _LANGUAGE_REGISTRY:
            return None

        module_name, func_name = _LANGUAGE_REGISTRY[ext]
        try:
            import importlib
            mod = importlib.import_module(module_name)
            lang_func = getattr(mod, func_name)
            lang = Language(lang_func())
            self._languages[ext] = lang
            return lang
        except Exception as e:
            logger.warning(f"Failed to load grammar for {ext}: {e}")
            return None

    def get_parser(self, extension: str) -> Optional[Parser]:
        """Get or create a parser for a file extension."""
        ext = extension.lower()
        if ext in self._parsers:
            return self._parsers[ext]

        lang = self.get_language(ext)
        if lang is None:
            return None

        parser = Parser(lang)
        self._parsers[ext] = parser
        return parser

    def get_lang_name(self, extension: str) -> Optional[str]:
        """Map extension to language name."""
        return EXTENSION_TO_LANG_NAME.get(extension.lower())

    def parse_file(self, file_path: str, source_bytes: bytes, extension: str):
        """
        Parse a file and return the tree-sitter tree.

        Returns:
            tree-sitter Tree or None
        """
        parser = self.get_parser(extension)
        if parser is None:
            return None
        try:
            return parser.parse(source_bytes)
        except Exception as e:
            logger.warning(f"Failed to parse {file_path}: {e}")
            return None

    def extract_functions(
        self, tree, source_bytes: bytes, language: str
    ) -> list[ExtractedNode]:
        """Extract function/method nodes from a parsed tree."""
        func_types = _FUNCTION_TYPES.get(language, set())
        if not func_types:
            return []

        results = []
        self._walk_for_types(
            tree.root_node, source_bytes, func_types, "function", language, results
        )
        return results

    def extract_classes(
        self, tree, source_bytes: bytes, language: str
    ) -> list[ExtractedNode]:
        """Extract class/struct nodes from a parsed tree."""
        class_types = _CLASS_TYPES.get(language, set())
        if not class_types:
            return []

        results = []
        self._walk_for_types(
            tree.root_node, source_bytes, class_types, "class", language, results
        )
        return results

    def extract_imports(
        self, tree, source_bytes: bytes, language: str
    ) -> list[str]:
        """Extract import statements as strings."""
        import_types = _IMPORT_TYPES.get(language, set())
        if not import_types:
            return []

        imports = []
        for child in tree.root_node.children:
            if child.type in import_types:
                text = child.text.decode("utf-8", errors="replace").strip()
                if text:
                    imports.append(text)
        return imports

    def extract_all(
        self, tree, source_bytes: bytes, language: str
    ) -> tuple[list[ExtractedNode], list[ExtractedNode], list[str]]:
        """Extract functions, classes, and imports in one pass."""
        func_types = _FUNCTION_TYPES.get(language, set())
        class_types = _CLASS_TYPES.get(language, set())
        import_types = _IMPORT_TYPES.get(language, set())

        functions = []
        classes = []
        imports = []

        self._extract_recursive(
            tree.root_node, source_bytes, language,
            func_types, class_types, import_types,
            functions, classes, imports,
            parent_class=None,
        )
        return functions, classes, imports

    def _extract_recursive(
        self,
        node: Node,
        source_bytes: bytes,
        language: str,
        func_types: set[str],
        class_types: set[str],
        import_types: set[str],
        functions: list[ExtractedNode],
        classes: list[ExtractedNode],
        imports: list[str],
        parent_class: str | None,
    ):
        """Recursively extract code elements from AST."""
        for child in node.children:
            if child.type in import_types:
                text = child.text.decode("utf-8", errors="replace").strip()
                if text:
                    imports.append(text)

            elif child.type in class_types:
                name = self._get_name(child, language)
                sig = self._get_signature(child, source_bytes, language)
                extracted = ExtractedNode(
                    name=name,
                    node_type="class",
                    text=child.text,
                    line_start=child.start_point.row + 1,
                    line_end=child.end_point.row + 1,
                    signature=sig,
                )
                classes.append(extracted)

                # Recurse into class body for methods
                body = self._get_body_node(child, language)
                if body:
                    self._extract_recursive(
                        body, source_bytes, language,
                        func_types, class_types, import_types,
                        functions, classes, imports,
                        parent_class=name,
                    )

            elif child.type in func_types:
                name = self._get_name(child, language)
                sig = self._get_signature(child, source_bytes, language)
                node_type = "method" if parent_class else "function"
                extracted = ExtractedNode(
                    name=name,
                    node_type=node_type,
                    text=child.text,
                    line_start=child.start_point.row + 1,
                    line_end=child.end_point.row + 1,
                    signature=sig,
                    class_name=parent_class,
                )
                functions.append(extracted)
            else:
                # Continue recursing for nested definitions
                self._extract_recursive(
                    child, source_bytes, language,
                    func_types, class_types, import_types,
                    functions, classes, imports,
                    parent_class=parent_class,
                )

    def _walk_for_types(
        self,
        node: Node,
        source_bytes: bytes,
        target_types: set[str],
        node_type: str,
        language: str,
        results: list[ExtractedNode],
        parent_class: str | None = None,
    ):
        """Walk tree and collect nodes of specific types."""
        for child in node.children:
            if child.type in target_types:
                name = self._get_name(child, language)
                sig = self._get_signature(child, source_bytes, language)
                extracted = ExtractedNode(
                    name=name,
                    node_type=node_type,
                    text=child.text,
                    line_start=child.start_point.row + 1,
                    line_end=child.end_point.row + 1,
                    signature=sig,
                    class_name=parent_class,
                )
                results.append(extracted)
            # Always recurse to find nested definitions
            self._walk_for_types(
                child, source_bytes, target_types,
                node_type, language, results, parent_class
            )

    def _get_name(self, node: Node, language: str) -> str:
        """Extract the name of a function/class node."""
        # Try common field names
        for field_name in ("name", "declarator"):
            name_node = node.child_by_field_name(field_name)
            if name_node:
                # For C/C++, declarator may contain nested declarator with the name
                if name_node.type in ("function_declarator", "pointer_declarator"):
                    inner = name_node.child_by_field_name("declarator")
                    if inner:
                        return inner.text.decode("utf-8", errors="replace")
                return name_node.text.decode("utf-8", errors="replace")

        # For arrow functions assigned to variables: const foo = () => {}
        if node.type == "arrow_function" and node.parent:
            parent = node.parent
            if parent.type in ("variable_declarator", "pair"):
                name_node = parent.child_by_field_name("name")
                if name_node:
                    return name_node.text.decode("utf-8", errors="replace")

        # For Go type declarations
        if node.type == "type_declaration":
            for child in node.children:
                if child.type == "type_spec":
                    name_node = child.child_by_field_name("name")
                    if name_node:
                        return name_node.text.decode("utf-8", errors="replace")

        return "<anonymous>"

    def _get_signature(self, node: Node, source_bytes: bytes, language: str) -> str:
        """Extract the signature line of a function/class."""
        text = node.text.decode("utf-8", errors="replace")
        # Take just the first line as the signature
        first_line = text.split("\n")[0].strip()
        # Limit length
        if len(first_line) > 200:
            first_line = first_line[:200] + "..."
        return first_line

    def _get_body_node(self, node: Node, language: str) -> Optional[Node]:
        """Get the body node of a class/struct."""
        # Python: class_definition → body (block)
        body = node.child_by_field_name("body")
        if body:
            return body

        # C++/Java: class body in { } block
        for child in node.children:
            if child.type in ("class_body", "field_declaration_list",
                              "declaration_list", "block"):
                return child
        return None


# Global instance
treesitter_service = TreeSitterService()
