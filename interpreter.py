import re
from typing import List, Tuple, NamedTuple

class Token(NamedTuple):
    type: str
    value: str
    line: int
    column: int

class ConciseTokenizer:
    def __init__(self):
        # 1. Define keywords and native types
        self.keywords = {'st', 'fn', 'return', 'if', 'else', 'try', 'catch', 'as'}
        self.types = {'String', 'Int', 'Float', 'List'}
        
        # 2. Token definitions ordered by extraction priority
        self.token_specs = [
            ('COMMENT',   r'//.*'),                         # Inline comments
            ('NUMBER',    r'\d+(\.\d+)?'),                  # Int and Float literals
            ('STRING',    r'"[^"\\]*(?:\\.[^"\\]*)*"'),     # String literals (handles escapes)
            ('COALESCE',  r'\?\?'),                         # Nullish coalescing operator
            ('ARROW',     r'->'),                           # Return type arrow
            ('OP_ASSIGN', r'\+=|-=|\*=|/='),                # Compound assignment
            ('COMPARE',   r'==|!=|>=|<=|>|<'),              # Comparison operators
            ('IDENTIFIER',r'[a-zA-Z_][a-zA-Z0-9_]*'),       # Variable and function names
            ('ASSIGN',    r'='),                            # Assignment
            ('COLON',     r':'),                            # Type or named argument separator
            ('COMMA',     r','),                            # Parameter separator
            ('DOT',       r'\.'),                           # Method / Property access
            ('BUBBLE',    r'\?'),                           # Error propagation bubble
            ('BANG',      r'!'),                            # Fallible marker
            ('LPAREN',    r'\('),                           # Parentheses
            ('RPAREN',    r'\)'),
            ('LBRACKET',  r'\['),                           # Brackets for lists / indexing
            ('RBRACKET',  r'\]'),
            ('PLUS',      r'\+'),                           # Math operators
            ('MINUS',     r'-'),
            ('MULT',      r'\*'),
            ('DIV',       r'/'),
            ('NEWLINE',   r'\n'),                           # Line breaks
            ('WHITESPACE',r'[ \t]+'),                       # Non-newline spacing
            ('MISMATCH',  r'.'),                            # Catch-all for illegal characters
        ]
        
        # Compile master regex pattern
        self.master_regex = re.compile(
            '|'.join(f'(?P<{name}>{pattern})' for name, pattern in self.token_specs)
        )

    def tokenize(self, code: str) -> List[Token]:
        tokens: List[Token] = []
        indent_stack = [0]  # Stack tracking current indentation space depths
        
        # Split code into lines to accurately process tracking and line numbers
        lines = code.splitlines()
        
        for line_num, line_text in enumerate(lines, start=1):
            # Skip empty lines or lines with just comments to avoid artificial indentation adjustments
            stripped = line_text.strip()
            if not stripped or stripped.startswith('//'):
                continue
                
            # Measure starting white space of the current line
            whitespace_match = re.match(r'^[ \t]*', line_text)
            indent_str = whitespace_match.group(0) if whitespace_match else ''
            
            # Expand tabs to 4 spaces for uniform calculations
            indent_level = indent_str.count(' ') + (indent_str.count('\t') * 4)
            
            # Handle indentation updates
            if indent_level > indent_stack[-1]:
                indent_stack.append(indent_level)
                tokens.append(Token('INDENT', indent_str, line_num, 1))
            else:
                while indent_level < indent_stack[-1]:
                    indent_stack.pop()
                    tokens.append(Token('DEDENT', '', line_num, 1))
                if indent_level != indent_stack[-1]:
                    raise SyntaxError(f"Indentation error on line {line_num}: Unaligned indentation level.")

            # Process the characters inside the line
            col_offset = len(indent_str)
            content_to_scan = line_text[col_offset:]
            
            for match in self.master_regex.finditer(content_to_scan):
                kind = match.lastgroup
                value = match.group(kind)
                column = match.start() + col_offset + 1
                
                if kind == 'WHITESPACE' or kind == 'COMMENT':
                    continue
                elif kind == 'IDENTIFIER':
                    if value in self.keywords:
                        kind = value.upper()
                    elif value in self.types:
                        kind = 'TYPE'
                elif kind == 'MISMATCH':
                    raise SyntaxError(f"Unexpected character '{value}' on line {line_num}, column {column}")
                
                tokens.append(Token(kind, value, line_num, column))
            
            # Mark the clean ending of a logical instruction line
            tokens.append(Token('NEWLINE', '\n', line_num, len(line_text) + 1))
            
        # Clean up remaining open indent blocks at EOF
        while len(indent_stack) > 1:
            indent_stack.pop()
            tokens.append(Token('DEDENT', '', len(lines), 1))
            
        return tokens

from dataclasses import dataclass, field
from typing import List, Optional, Union

# Base AST Node
class ASTNode: pass

@dataclass
class ProgramNode(ASTNode):
    body: List[ASTNode] = field(default_factory=list)

@dataclass
class TypeNode(ASTNode):
    name: str
    is_fallible: bool = False  # True if it ends with '!'

@dataclass
class VariableDeclNode(ASTNode):
    type_node: TypeNode
    name: str

@dataclass
class StructNode(ASTNode):
    name: str
    fields: List[VariableDeclNode] = field(default_factory=list)

@dataclass
class ParameterNode(ASTNode):
    name: str
    type_node: TypeNode

@dataclass
class FunctionNode(ASTNode):
    name: str
    params: List[ParameterNode]
    return_type: Optional[TypeNode]
    body: List[ASTNode] = field(default_factory=list)

@dataclass
class IdentifierNode(ASTNode):
    name: str

@dataclass
class LiteralNode(ASTNode):
    value: str
    type: str  # 'NUMBER' or 'STRING'

@dataclass
class BinaryOpNode(ASTNode):
    left: ASTNode
    op: str
    right: ASTNode

@dataclass
class AssignmentNode(ASTNode):
    target: ASTNode  # Identifier, Indexing, or Property
    value: ASTNode
    op: str = "="    # Can support "=", "+=", etc.

@dataclass
class VarInitNode(ASTNode):
    type_node: TypeNode
    name: str
    value: ASTNode

@dataclass
class IndexNode(ASTNode):
    var: ASTNode
    index: ASTNode

@dataclass
class BubbleNode(ASTNode):
    expr: ASTNode

@dataclass
class CallNode(ASTNode):
    func: ASTNode
    args: List[ASTNode] = field(default_factory=list)

@dataclass
class ReturnNode(ASTNode):
    def __init__(self, value=None):
        self.value = value  # This will hold an Expression Node, or None for void returns

    def __repr__(self):
        return f"ReturnNode(value={self.value})"


class ConciseParser:
    def __init__(self, tokens: List[Token]):
        # Filter out loose sequential duplicate newlines to simplify parsing
        self.tokens = self._clean_tokens(tokens)
        self.pos = 0

    def _clean_tokens(self, tokens: List[Token]) -> List[Token]:
        cleaned = []
        for t in tokens:
            if t.type == 'NEWLINE' and cleaned and cleaned[-1].type == 'NEWLINE':
                continue
            cleaned.append(t)
        return cleaned

    def current_token(self) -> Token:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return Token('EOF', '', -1, -1)

    def consume(self, expected_type: str) -> Token:
        token = self.current_token()
        if token.type == expected_type:
            self.pos += 1
            return token
        raise SyntaxError(f"Line {token.line}: Expected token '{expected_type}', got '{token.type}' ('{token.value}')")

    def parse(self) -> ProgramNode:
        program = ProgramNode()
        while self.current_token().type != 'EOF':
            # Skip floating newlines at root level
            if self.current_token().type == 'NEWLINE':
                self.consume('NEWLINE')
                continue
            stmt = self.parse_statement()
            if stmt:
                program.body.append(stmt)
        return program

    def parse_statement(self) -> ASTNode:
        token = self.current_token()
        
        if token.type == 'ST':
            return self.parse_struct()
        elif token.type == 'FN':
            return self.parse_function()

        elif token.type == 'RETURN':
            return self.parse_return_statement()
            
        # Look-ahead check: If it's a TYPE, or an IDENTIFIER followed by another IDENTIFIER
        elif token.type == 'TYPE' or (
            token.type == 'IDENTIFIER' and 
            self.pos + 1 < len(self.tokens) and 
            self.tokens[self.pos + 1].type == 'IDENTIFIER'
        ):
            return self.parse_variable_declaration_or_assignment()
            
        else:
            expr = self.parse_expression()
            if self.current_token().type in ('ASSIGN', 'OP_ASSIGN'):
                op_tok = self.current_token()
                self.consume(op_tok.type)
                val = self.parse_expression()
                expr = AssignmentNode(target=expr, value=val, op=op_tok.value)
            
            if self.current_token().type == 'NEWLINE':
                self.consume('NEWLINE')
            return expr

    def parse_struct(self) -> StructNode:
        self.consume('ST')
        name_tok = self.consume('IDENTIFIER')
        self.consume('NEWLINE')
        self.consume('INDENT')
        
        struct_node = StructNode(name=name_tok.value)
        while self.current_token().type != 'DEDENT' and self.current_token().type != 'EOF':
            if self.current_token().type == 'NEWLINE':
                self.consume('NEWLINE')
                continue
            
            # FIX: Uses the unified type helper for properties
            t_node = self.parse_type_name()
            prop_tok = self.consume('IDENTIFIER')
            self.consume('NEWLINE')
            
            struct_node.fields.append(VariableDeclNode(type_node=t_node, name=prop_tok.value))
            
        self.consume('DEDENT')
        return struct_node

    def parse_function(self) -> FunctionNode:
        self.consume('FN')
        func_name = self.consume('IDENTIFIER').value
        self.consume('LPAREN')
        
        params = []
        if self.current_token().type != 'RPAREN':
            while True:
                p_name = self.consume('IDENTIFIER').value
                self.consume('COLON')
                # FIX: Uses the unified type helper for function parameters
                p_type_node = self.parse_type_name()
                params.append(ParameterNode(name=p_name, type_node=p_type_node))
                if self.current_token().type == 'COMMA':
                    self.consume('COMMA')
                else:
                    break
                    
        self.consume('RPAREN')
        
        return_type = None
        if self.current_token().type == 'ARROW':
            self.consume('ARROW')
            # FIX: Uses the unified type helper for function return values
            return_type = self.parse_type_name()
            
        self.consume('NEWLINE')
        self.consume('INDENT')
        
        func_node = FunctionNode(name=func_name, params=params, return_type=return_type)
        while self.current_token().type != 'DEDENT' and self.current_token().type != 'EOF':
            if self.current_token().type == 'NEWLINE':
                self.consume('NEWLINE')
                continue
            stmt = self.parse_statement()
            if stmt:
                func_node.body.append(stmt)
                
        self.consume('DEDENT')
        return func_node

    def parse_variable_declaration_or_assignment(self) -> ASTNode:
        # FIX: Uses the unified type helper for explicit variable instantiations
        t_node = self.parse_type_name()
        var_tok = self.consume('IDENTIFIER')
        
        if self.current_token().type == 'ASSIGN':
            self.consume('ASSIGN')
            value_expr = self.parse_expression()
            return VarInitNode(type_node=t_node, name=var_tok.value, value=value_expr)
        
        return VariableDeclNode(type_node=t_node, name=var_tok.value)


    def parse_expression(self) -> ASTNode:
        return self.parse_coalesce()

    def parse_coalesce(self) -> ASTNode:
        # Binary structural operation tree parsing for '??'
        node = self.parse_comparison()
        while self.current_token().type == 'COALESCE':
            op = self.consume('COALESCE').value
            right = self.parse_comparison()
            node = BinaryOpNode(left=node, op=op, right=right)
        return node

    def parse_comparison(self) -> ASTNode:
        node = self.parse_additive()
        while self.current_token().type == 'COMPARE':
            op = self.consume('COMPARE').value
            right = self.parse_additive()
            node = BinaryOpNode(left=node, op=op, right=right)
        return node

    def parse_additive(self) -> ASTNode:
        node = self.parse_primary_postfix()
        while self.current_token().type in ('PLUS', 'MINUS'):
            op = self.consume(self.current_token().type).value
            right = self.parse_primary_postfix()
            node = BinaryOpNode(left=node, op=op, right=right)
        return node

    def parse_primary_postfix(self) -> ASTNode:
        node = self.parse_primary()
        
        # Handle trailing loops for structures like expressions: users[id]?, method calls, or indices
        while True:
            if self.current_token().type == 'LBRACKET':
                self.consume('LBRACKET')
                idx = self.parse_expression()
                self.consume('RBRACKET')
                node = IndexNode(var=node, index=idx)
            elif self.current_token().type == 'BUBBLE':
                self.consume('BUBBLE')
                node = BubbleNode(expr=node)
            elif self.current_token().type == 'LPAREN':
                self.consume('LPAREN')
                args = []
                if self.current_token().type != 'RPAREN':
                    while True:
                        args.append(self.parse_expression())
                        if self.current_token().type == 'COMMA':
                            self.consume('COMMA')
                        else:
                            break
                self.consume('RPAREN')
                node = CallNode(func=node, args=args)
            else:
                break
        return node

    def parse_primary(self) -> ASTNode:
        token = self.current_token()
        if token.type == 'NUMBER':
            self.consume('NUMBER')
            return LiteralNode(value=token.value, type='NUMBER')
        elif token.type == 'STRING':
            self.consume('STRING')
            return LiteralNode(value=token.value, type='STRING')
        elif token.type == 'IDENTIFIER':
            self.consume('IDENTIFIER')
            return IdentifierNode(name=token.value)
        raise SyntaxError(f"Line {token.line}: Unexpected token in expression evaluation: '{token.type}' ('{token.value}')")

    def parse_type_name(self) -> TypeNode:
        token = self.current_token()
        if token.type in ('TYPE', 'IDENTIFIER'):
            self.pos += 1  # Consume the type token
            
            # Check for trailing '!' fallibility marker
            is_fallible = False
            if self.current_token().type == 'BANG':
                self.consume('BANG')
                is_fallible = True
                
            return TypeNode(name=token.value, is_fallible=is_fallible)
        raise SyntaxError(f"Line {token.line}: Expected a type name, got '{token.type}' ('{token.value}')")

    def parse_return_statement(self) -> ReturnNode:
        # 1. Consume the 'return' token itself
        self.consume('RETURN')
        
        # 2. Check if it's a void return (immediately followed by a line break)
        if self.current_token().type == 'NEWLINE':
            self.consume('NEWLINE')
            return ReturnNode(value=None)
            
        # 3. Otherwise, parse the returned expression (e.g., a variable, function call, or operation)
        return_value = self.parse_expression()
        
        # 4. Clean up the trailing newline of the statement line
        if self.current_token().type == 'NEWLINE':
            self.consume('NEWLINE')
            
        return ReturnNode(value=return_value)


import json

# Utility to safely convert dataclasses to clean dictionaries for printing
def ast_to_dict(node):
    if isinstance(node, list):
        return [ast_to_dict(i) for i in node]
    if hasattr(node, '__dataclass_fields__'):
        res = {"_node": node.__class__.__name__}
        for field in node.__dataclass_fields__:
            res[field] = ast_to_dict(getattr(node, field))
        return res
    return node

# Code block showcasing structs, functions, indexing, and error bubbling
concise_source = """
st User
    String name
    Int age

fn get_user(id: Int) -> User!
    User user = users[id]?
    return user
"""

try:
    # 1. Tokenize code
    tokenizer = ConciseTokenizer()
    tokens = tokenizer.tokenize(concise_source)
    # print(tokens)

    # 2. Parse into AST
    parser = ConciseParser(tokens)
    ast_tree = parser.parse()
    
    # Print formatted JSON tree
    print(json.dumps(ast_to_dict(ast_tree), indent=2))

except SyntaxError as e:
    print(f"Compilation Parser Error: {e}")