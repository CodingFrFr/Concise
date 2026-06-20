import re
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Union

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
    parameters: List['TypeNode'] = field(default_factory=list)
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
class PropertyAccessNode(ASTNode):
    obj: ASTNode
    prop: str

@dataclass
class BubbleNode(ASTNode):
    expr: ASTNode

@dataclass
class ArgumentNode(ASTNode):
    name: Optional[str]
    value: ASTNode

@dataclass
class CallNode(ASTNode):
    func: ASTNode
    args: List[ArgumentNode] = field(default_factory=list)

@dataclass
class ListLiteralNode(ASTNode):
    elements: List[ASTNode] = field(default_factory=list)

@dataclass
class CastNode(ASTNode):
    expr: ASTNode
    type_node: TypeNode

@dataclass
class IfNode(ASTNode):
    condition: ASTNode
    then_body: List[ASTNode] = field(default_factory=list)
    else_body: Optional[List[ASTNode]] = None

@dataclass
class TryCatchNode(ASTNode):
    try_body: List[ASTNode] = field(default_factory=list)
    catch_var: Optional[str] = None
    catch_body: List[ASTNode] = field(default_factory=list)

@dataclass
class ReturnNode(ASTNode):
    value: Optional[ASTNode] = None


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
        elif token.type == 'IF':
            return self.parse_if_statement()
        elif token.type == 'TRY':
            return self.parse_try_catch()
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
        t_node = self.parse_type_name()
        var_tok = self.consume('IDENTIFIER')
        
        if self.current_token().type == 'ASSIGN':
            self.consume('ASSIGN')
            value_expr = self.parse_expression()
            return VarInitNode(type_node=t_node, name=var_tok.value, value=value_expr)
        
        return VariableDeclNode(type_node=t_node, name=var_tok.value)

    def parse_if_statement(self) -> IfNode:
        self.consume('IF')
        condition = self.parse_expression()
        self.consume('NEWLINE')
        self.consume('INDENT')

        then_body = []
        while self.current_token().type not in ('DEDENT', 'EOF'):
            if self.current_token().type == 'NEWLINE':
                self.consume('NEWLINE')
                continue
            then_body.append(self.parse_statement())

        self.consume('DEDENT')
        else_body = None
        if self.current_token().type == 'ELSE':
            self.consume('ELSE')
            self.consume('NEWLINE')
            self.consume('INDENT')
            else_body = []
            while self.current_token().type not in ('DEDENT', 'EOF'):
                if self.current_token().type == 'NEWLINE':
                    self.consume('NEWLINE')
                    continue
                else_body.append(self.parse_statement())
            self.consume('DEDENT')

        return IfNode(condition=condition, then_body=then_body, else_body=else_body)

    def parse_try_catch(self) -> TryCatchNode:
        self.consume('TRY')
        self.consume('NEWLINE')
        self.consume('INDENT')

        try_body = []
        while self.current_token().type not in ('DEDENT', 'EOF'):
            if self.current_token().type == 'NEWLINE':
                self.consume('NEWLINE')
                continue
            try_body.append(self.parse_statement())

        self.consume('DEDENT')

        catch_var = None
        catch_body = []
        if self.current_token().type == 'CATCH':
            self.consume('CATCH')
            if self.current_token().type == 'IDENTIFIER':
                catch_var = self.consume('IDENTIFIER').value
            self.consume('NEWLINE')
            self.consume('INDENT')
            while self.current_token().type not in ('DEDENT', 'EOF'):
                if self.current_token().type == 'NEWLINE':
                    self.consume('NEWLINE')
                    continue
                catch_body.append(self.parse_statement())
            self.consume('DEDENT')

        return TryCatchNode(try_body=try_body, catch_var=catch_var, catch_body=catch_body)

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
        node = self.parse_multiplicative()
        while self.current_token().type in ('PLUS', 'MINUS'):
            op = self.consume(self.current_token().type).value
            right = self.parse_multiplicative()
            node = BinaryOpNode(left=node, op=op, right=right)
        return node

    def parse_multiplicative(self) -> ASTNode:
        node = self.parse_primary_postfix()
        while self.current_token().type in ('MULT', 'DIV'):
            op = self.consume(self.current_token().type).value
            right = self.parse_primary_postfix()
            node = BinaryOpNode(left=node, op=op, right=right)
        return node

    def parse_primary_postfix(self) -> ASTNode:
        node = self.parse_primary()
        
        # Handle trailing loops for structures like expressions: property access, indexing, error bubbling, method calls, and casts
        while True:
            if self.current_token().type == 'DOT':
                self.consume('DOT')
                prop_name = self.consume('IDENTIFIER').value
                node = PropertyAccessNode(obj=node, prop=prop_name)
            elif self.current_token().type == 'LBRACKET':
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
                        args.append(self.parse_argument())
                        if self.current_token().type == 'COMMA':
                            self.consume('COMMA')
                        else:
                            break
                self.consume('RPAREN')
                node = CallNode(func=node, args=args)
            elif self.current_token().type == 'AS':
                self.consume('AS')
                type_node = self.parse_type_name()
                node = CastNode(expr=node, type_node=type_node)
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
        elif token.type == 'LBRACKET':
            return self.parse_list_literal()
        elif token.type == 'LPAREN':
            self.consume('LPAREN')
            expr = self.parse_expression()
            self.consume('RPAREN')
            return expr
        elif token.type == 'IDENTIFIER':
            self.consume('IDENTIFIER')
            return IdentifierNode(name=token.value)
        raise SyntaxError(f"Line {token.line}: Unexpected token in expression evaluation: '{token.type}' ('{token.value}')")

    def parse_list_literal(self) -> ListLiteralNode:
        self.consume('LBRACKET')
        elements: List[ASTNode] = []
        if self.current_token().type != 'RBRACKET':
            while True:
                elements.append(self.parse_expression())
                if self.current_token().type == 'COMMA':
                    self.consume('COMMA')
                else:
                    break
        self.consume('RBRACKET')
        return ListLiteralNode(elements=elements)

    def parse_argument(self) -> ArgumentNode:
        if self.current_token().type == 'IDENTIFIER' and self.pos + 1 < len(self.tokens) and self.tokens[self.pos + 1].type == 'COLON':
            arg_name = self.consume('IDENTIFIER').value
            self.consume('COLON')
            value = self.parse_expression()
            return ArgumentNode(name=arg_name, value=value)
        return ArgumentNode(name=None, value=self.parse_expression())

    def parse_type_name(self) -> TypeNode:
        token = self.current_token()
        if token.type in ('TYPE', 'IDENTIFIER'):
            self.pos += 1  # Consume the type token
            name = token.value
            parameters: List[TypeNode] = []

            if self.current_token().type == 'LBRACKET':
                self.consume('LBRACKET')
                while True:
                    parameters.append(self.parse_type_name())
                    if self.current_token().type == 'COMMA':
                        self.consume('COMMA')
                        continue
                    break
                self.consume('RBRACKET')

            is_fallible = False
            if self.current_token().type == 'BANG':
                self.consume('BANG')
                is_fallible = True
                
            return TypeNode(name=name, parameters=parameters, is_fallible=is_fallible)
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


class ConciseRuntimeError(Exception):
    def __init__(self, message: str, error_value: Optional['ErrorValue'] = None):
        super().__init__(message)
        self.message = message
        self.error_value = error_value

class ConciseReturn(Exception):
    def __init__(self, value: Any):
        self.value = value

class ErrorValue:
    def __init__(self, message: str):
        self.message = message

    def __repr__(self) -> str:
        return f"ErrorValue({self.message!r})"

class RuntimeErrorObject:
    def __init__(self, message: str):
        self.message = message

    def __repr__(self) -> str:
        return f"RuntimeErrorObject({self.message!r})"

class StructInstance:
    def __init__(self, type_name: str, fields: List[str], values: Dict[str, Any]):
        object.__setattr__(self, '_type_name', type_name)
        object.__setattr__(self, '_fields', fields)
        object.__setattr__(self, '_values', values)

    def __getattr__(self, name: str) -> Any:
        if name in self._values:
            return self._values[name]
        raise AttributeError(f"{self._type_name} has no field '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        if name in ('_type_name', '_fields', '_values'):
            object.__setattr__(self, name, value)
        elif name in self._fields:
            self._values[name] = value
        else:
            raise AttributeError(f"{self._type_name} has no field '{name}'")

    def __repr__(self) -> str:
        return f"<{self._type_name} {self._values}>"

class ConciseList:
    def __init__(self, items: Optional[List[Any]] = None):
        self.items = list(items or [])

    def push(self, value: Any) -> None:
        self.items.append(value)

    def length(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Any:
        try:
            return self.items[index]
        except Exception:
            return None

    def __setitem__(self, index: int, value: Any) -> None:
        if index < 0 or index >= len(self.items):
            raise ConciseRuntimeError('Index out of range')
        self.items[index] = value

    def __repr__(self) -> str:
        return f"ConciseList({self.items})"

class FunctionValue:
    def __init__(self, name: str, params: List[ParameterNode], body: List[ASTNode], interpreter: 'ConciseInterpreter'):
        self.name = name
        self.params = params
        self.body = body
        self.interpreter = interpreter

    def call(self, args: List[Any], kwargs: Dict[str, Any]) -> Any:
        frame: Dict[str, Any] = {}
        for index, param in enumerate(self.params):
            if param.name in kwargs:
                frame[param.name] = kwargs[param.name]
            elif index < len(args):
                frame[param.name] = args[index]
            else:
                frame[param.name] = None

        self.interpreter.push_scope(frame)
        try:
            for statement in self.body:
                self.interpreter.exec_statement(statement)
        except ConciseReturn as ret:
            return ret.value
        finally:
            self.interpreter.pop_scope()
        return None

    def __repr__(self) -> str:
        return f"FunctionValue({self.name})"

class StructConstructor:
    def __init__(self, type_name: str, fields: List[VariableDeclNode]):
        self.type_name = type_name
        self.fields = [field.name for field in fields]

    def call(self, args: List[Any], kwargs: Dict[str, Any]) -> StructInstance:
        values: Dict[str, Any] = {}
        for index, field_name in enumerate(self.fields):
            if field_name in kwargs:
                values[field_name] = kwargs[field_name]
            elif index < len(args):
                values[field_name] = args[index]
            else:
                values[field_name] = None

        extra_kw = set(kwargs) - set(self.fields)
        if extra_kw:
            raise ConciseRuntimeError(f"Unexpected constructor fields: {extra_kw}")

        return StructInstance(self.type_name, self.fields, values)

    def __repr__(self) -> str:
        return f"StructConstructor({self.type_name})"

class ConciseInterpreter:
    def __init__(self):
        self.globals: Dict[str, Any] = {}
        self.scopes: List[Dict[str, Any]] = [self.globals]
        self.setup_builtins()

    def setup_builtins(self) -> None:
        self.globals['log'] = self.builtin_log
        self.globals['error'] = self.builtin_error

    def push_scope(self, scope: Dict[str, Any]) -> None:
        self.scopes.append(scope)

    def pop_scope(self) -> None:
        if len(self.scopes) > 1:
            self.scopes.pop()

    def current_scope(self) -> Dict[str, Any]:
        return self.scopes[-1]

    def define_var(self, name: str, value: Any) -> None:
        self.current_scope()[name] = value

    def set_var(self, name: str, value: Any) -> None:
        for scope in reversed(self.scopes):
            if name in scope:
                scope[name] = value
                return
        self.current_scope()[name] = value

    def get_var(self, name: str) -> Any:
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        raise ConciseRuntimeError(f"Undefined variable '{name}'")

    def run(self, program: ProgramNode) -> None:
        for statement in program.body:
            self.exec_statement(statement)

    def exec_statement(self, node: ASTNode) -> Any:
        if isinstance(node, StructNode):
            self.define_var(node.name, StructConstructor(node.name, node.fields))
            return None
        if isinstance(node, FunctionNode):
            self.define_var(node.name, FunctionValue(node.name, node.params, node.body, self))
            return None
        if isinstance(node, VarInitNode):
            value = self.eval_expression(node.value)
            self.define_var(node.name, value)
            return None
        if isinstance(node, VariableDeclNode):
            self.define_var(node.name, None)
            return None
        if isinstance(node, AssignmentNode):
            value = self.eval_expression(node.value)
            self.assign_target(node.target, value, node.op)
            return None
        if isinstance(node, IfNode):
            condition = self.eval_expression(node.condition)
            if self.is_truthy(condition):
                for stmt in node.then_body:
                    self.exec_statement(stmt)
            elif node.else_body is not None:
                for stmt in node.else_body:
                    self.exec_statement(stmt)
            return None
        if isinstance(node, TryCatchNode):
            try:
                for stmt in node.try_body:
                    self.exec_statement(stmt)
            except ConciseRuntimeError as exc:
                catch_scope: Dict[str, Any] = {}
                if node.catch_var:
                    catch_scope[node.catch_var] = RuntimeErrorObject(exc.message)
                self.push_scope(catch_scope)
                try:
                    for stmt in node.catch_body:
                        self.exec_statement(stmt)
                finally:
                    self.pop_scope()
            return None
        if isinstance(node, ReturnNode):
            value = self.eval_expression(node.value) if node.value is not None else None
            raise ConciseReturn(value)
        if isinstance(node, CallNode):
            self.eval_expression(node)
            return None
        # Expression statements
        self.eval_expression(node)
        return None

    def assign_target(self, target: ASTNode, value: Any, op: str = '=') -> None:
        if op == '=':
            self.set_target(target, value)
            return
        current = self.eval_expression(self.copy_target(target))
        if op == '+=':
            value = current + value
        elif op == '-=':
            value = current - value
        elif op == '*=':
            value = current * value
        elif op == '/=':
            value = current / value
        else:
            raise ConciseRuntimeError(f"Unsupported assignment operator '{op}'")
        self.set_target(target, value)

    def copy_target(self, target: ASTNode) -> ASTNode:
        return target

    def set_target(self, target: ASTNode, value: Any) -> None:
        if isinstance(target, IdentifierNode):
            self.set_var(target.name, value)
            return
        if isinstance(target, PropertyAccessNode):
            obj = self.eval_expression(target.obj)
            if isinstance(obj, StructInstance):
                setattr(obj, target.prop, value)
                return
            if isinstance(obj, ConciseList) and target.prop in ('push', 'length'):
                raise ConciseRuntimeError(f"Cannot assign to method '{target.prop}'")
            setattr(obj, target.prop, value)
            return
        if isinstance(target, IndexNode):
            container = self.eval_expression(target.var)
            index = self.eval_expression(target.index)
            if isinstance(container, ConciseList):
                container[index] = value
                return
            try:
                container[index] = value
                return
            except Exception as exc:
                raise ConciseRuntimeError(str(exc))
        raise ConciseRuntimeError(f"Invalid assignment target {target}")

    def eval_expression(self, node: ASTNode) -> Any:
        if isinstance(node, LiteralNode):
            if node.type == 'NUMBER':
                return float(node.value) if '.' in node.value else int(node.value)
            return node.value[1:-1]
        if isinstance(node, IdentifierNode):
            return self.get_var(node.name)
        if isinstance(node, TypeNode):
            return node.name
        if isinstance(node, BinaryOpNode):
            left = self.eval_expression(node.left)
            right = self.eval_expression(node.right)
            if node.op == '??':
                return right if left is None else left
            if node.op == '==':
                return left == right
            if node.op == '!=':
                return left != right
            if node.op == '>=':
                return left >= right
            if node.op == '<=':
                return left <= right
            if node.op == '>':
                return left > right
            if node.op == '<':
                return left < right
            if node.op == '+':
                return left + right
            if node.op == '-':
                return left - right
            if node.op == '*':
                return left * right
            if node.op == '/':
                return left / right
            raise ConciseRuntimeError(f"Unsupported binary operator '{node.op}'")
        if isinstance(node, AssignmentNode):
            self.assign_target(node.target, self.eval_expression(node.value), node.op)
            return self.eval_expression(node.target)
        if isinstance(node, VarInitNode):
            value = self.eval_expression(node.value)
            self.define_var(node.name, value)
            return value
        if isinstance(node, IndexNode):
            container = self.eval_expression(node.var)
            index = self.eval_expression(node.index)
            if isinstance(container, ConciseList):
                return container[index]
            try:
                return container[index]
            except Exception:
                return None
        if isinstance(node, PropertyAccessNode):
            obj = self.eval_expression(node.obj)
            if isinstance(obj, StructInstance):
                return getattr(obj, node.prop)
            if isinstance(obj, ConciseList):
                if node.prop == 'push':
                    return lambda *args: obj.push(*args)
                if node.prop == 'length':
                    return lambda: obj.length()
            if hasattr(obj, node.prop):
                return getattr(obj, node.prop)
            raise ConciseRuntimeError(f"Property '{node.prop}' not found on object")
        if isinstance(node, BubbleNode):
            value = self.eval_expression(node.expr)
            if isinstance(value, ErrorValue):
                raise ConciseRuntimeError(value.message, value)
            if value is None:
                raise ConciseRuntimeError('Unexpected null value in bubble expression')
            return value
        if isinstance(node, CallNode):
            func = self.eval_expression(node.func)
            args: List[Any] = []
            kwargs: Dict[str, Any] = {}
            for arg in node.args:
                value = self.eval_expression(arg.value)
                if arg.name:
                    kwargs[arg.name] = value
                else:
                    args.append(value)
            if isinstance(func, FunctionValue):
                return func.call(args, kwargs)
            if isinstance(func, StructConstructor):
                return func.call(args, kwargs)
            if callable(func):
                return func(*args, **kwargs)
            raise ConciseRuntimeError('Attempted to call a non-callable object')
        if isinstance(node, ListLiteralNode):
            return ConciseList([self.eval_expression(item) for item in node.elements])
        if isinstance(node, CastNode):
            value = self.eval_expression(node.expr)
            return self.cast_value(value, node.type_node)
        if isinstance(node, IfNode):
            raise ConciseRuntimeError('If expressions are not supported in expression position')
        if isinstance(node, TryCatchNode):
            raise ConciseRuntimeError('Try/catch is not supported in expression position')
        if isinstance(node, ReturnNode):
            raise ConciseRuntimeError('Return is not supported in expression position')
        raise ConciseRuntimeError(f"Unknown expression type {type(node).__name__}")

    def cast_value(self, value: Any, type_node: TypeNode) -> Any:
        if type_node.name == 'String':
            return str(value)
        if type_node.name == 'Int':
            try:
                return int(value)
            except Exception:
                return ErrorValue(f"Cannot cast {value!r} to Int")
        if type_node.name == 'Float':
            try:
                return float(value)
            except Exception:
                return ErrorValue(f"Cannot cast {value!r} to Float")
        return value

    def is_truthy(self, value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, ErrorValue):
            return False
        return bool(value)

    def builtin_log(self, message: str) -> None:
        if isinstance(message, str):
            formatted = re.sub(r'\{([^}]+)\}', lambda m: str(self.evaluate_template_expression(m.group(1).strip())), message)
            print(formatted)
            return None
        print(message)
        return None

    def builtin_error(self, message: str) -> ErrorValue:
        return ErrorValue(message)

    def evaluate_template_expression(self, expression: str) -> Any:
        tokenizer = ConciseTokenizer()
        tokens = tokenizer.tokenize(expression)
        parser = ConciseParser(tokens)
        ast = parser.parse_expression()
        return self.eval_expression(ast)

    def eval_json(self, ast_json: Dict[str, Any]) -> None:
        ast = ast_from_dict(ast_json)
        self.run(ast)


def ast_from_dict(data: Any) -> Any:
    if isinstance(data, list):
        return [ast_from_dict(item) for item in data]
    if not isinstance(data, dict):
        return data
    node_type = data.get('_node')
    if node_type == 'ProgramNode':
        return ProgramNode(body=ast_from_dict(data['body']))
    if node_type == 'TypeNode':
        return TypeNode(name=data['name'], parameters=ast_from_dict(data['parameters']), is_fallible=data['is_fallible'])
    if node_type == 'VariableDeclNode':
        return VariableDeclNode(type_node=ast_from_dict(data['type_node']), name=data['name'])
    if node_type == 'StructNode':
        return StructNode(name=data['name'], fields=ast_from_dict(data['fields']))
    if node_type == 'ParameterNode':
        return ParameterNode(name=data['name'], type_node=ast_from_dict(data['type_node']))
    if node_type == 'FunctionNode':
        return FunctionNode(name=data['name'], params=ast_from_dict(data['params']), return_type=ast_from_dict(data['return_type']) if data['return_type'] is not None else None, body=ast_from_dict(data['body']))
    if node_type == 'IdentifierNode':
        return IdentifierNode(name=data['name'])
    if node_type == 'LiteralNode':
        return LiteralNode(value=data['value'], type=data['type'])
    if node_type == 'BinaryOpNode':
        return BinaryOpNode(left=ast_from_dict(data['left']), op=data['op'], right=ast_from_dict(data['right']))
    if node_type == 'AssignmentNode':
        return AssignmentNode(target=ast_from_dict(data['target']), value=ast_from_dict(data['value']), op=data['op'])
    if node_type == 'VarInitNode':
        return VarInitNode(type_node=ast_from_dict(data['type_node']), name=data['name'], value=ast_from_dict(data['value']))
    if node_type == 'IndexNode':
        return IndexNode(var=ast_from_dict(data['var']), index=ast_from_dict(data['index']))
    if node_type == 'PropertyAccessNode':
        return PropertyAccessNode(obj=ast_from_dict(data['obj']), prop=data['prop'])
    if node_type == 'BubbleNode':
        return BubbleNode(expr=ast_from_dict(data['expr']))
    if node_type == 'ArgumentNode':
        return ArgumentNode(name=data['name'], value=ast_from_dict(data['value']))
    if node_type == 'CallNode':
        return CallNode(func=ast_from_dict(data['func']), args=ast_from_dict(data['args']))
    if node_type == 'ListLiteralNode':
        return ListLiteralNode(elements=ast_from_dict(data['elements']))
    if node_type == 'CastNode':
        return CastNode(expr=ast_from_dict(data['expr']), type_node=ast_from_dict(data['type_node']))
    if node_type == 'IfNode':
        return IfNode(condition=ast_from_dict(data['condition']), then_body=ast_from_dict(data['then_body']), else_body=ast_from_dict(data['else_body']) if data['else_body'] is not None else None)
    if node_type == 'TryCatchNode':
        return TryCatchNode(try_body=ast_from_dict(data['try_body']), catch_var=data['catch_var'], catch_body=ast_from_dict(data['catch_body']))
    if node_type == 'ReturnNode':
        return ReturnNode(value=ast_from_dict(data['value']) if data['value'] is not None else None)
    raise ValueError(f"Unknown AST node type: {node_type}")

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


def main() -> None:
    concise_source = """
st User
    String name
    Int age

fn get_user(id: Int) -> User!
    User user = users[id]?
    return user
"""

    try:
        tokenizer = ConciseTokenizer()
        tokens = tokenizer.tokenize(concise_source)
        parser = ConciseParser(tokens)
        ast_tree = parser.parse()
        print(json.dumps(ast_to_dict(ast_tree), indent=2))

        interpreter = ConciseInterpreter()
        interpreter.eval_json(ast_to_dict(ast_tree))
    except SyntaxError as e:
        print(f"Compilation Parser Error: {e}")


if __name__ == '__main__':
    main()