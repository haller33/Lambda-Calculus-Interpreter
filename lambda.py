r"""Lambda Calculus Interpreter
To use, see `program.txt`. Each line can contain a single
macro or expression in lambda calculus notation, allowing
either λ or \ to be used as the lambda character.
Parentheses may be omitted; function application is
treated as left-associative. Macros may include any
previously-declared macros in their definitions, but may
not include themselves.

The script will perform call-by-name reduction on all
macros declared in the program before substituting them
into the expressions. Each expression will then be reduced
to its beta-eta normal form and output as a string.

After the program is executed, an interactive REPL will
be opened where you can evaluate further expressions and
macro declarations in real time.

Note that if you attempt to evaluate an expression that has
no normal form in `program.txt`, such as the Ω and Y
combinators, the program will likely go through infinite
recursion and crash. However, in the REPL, this will be
detected and fail gracefully.
"""

import copy
import re


class Token:
    def __init__(self, s):
        self.value = s
        if s == '\n':
            self.type = 'linebreak'
        elif s == '\\' or s == 'λ':
            self.type = 'lambda'
        elif s == '(':
            self.type = 'paren_open'
        elif s == ')':
            self.type = 'paren_close'
        elif s == '=':
            self.type = 'assignment'
        elif s == '.':
            self.type = 'lambda_delim'
        elif s == '^':
            self.type = 'noreduce'
        elif re.fullmatch('[a-z]+', s):
            self.type = 'variable'
        elif re.fullmatch('[A-Z0-9]+', s):
            self.type = 'macro'
        else:
            raise ValueError(
                'Lexical analysis error, invalid token ' + repr(s))

    def __repr__(self):
        return 'Token(' + repr(self.value) + ')'

    def __str__(self):
        return f'<Token {self.type}: {self.value.strip()}>'


def tokenize(s):
    tokens = re.findall(r'[\n\\λ()=.\^]|[a-zA-Z0-9]+|#[^\n]*', s)
    return [Token(t) for t in tokens if t[0] != '#']

# Types of nodes:
# - Expressions
#   | Lambda expressions
#   | Atoms (variables, macros)
#   | Function evaluation f(g)
# - Macro Declarations
#   | <Macro> = <Expression>
# - Program
#   | List of macro declarations, separated by newlines
#   | Also can include expressions, which are evaluated + printed out


def partition_last(tokens):
    assert tokens
    depth = 0
    split_index = None
    for i, t in enumerate(tokens):
        if depth == 0:
            split_index = i
            if t.type == 'lambda':
                return split_index
        if t.type == 'paren_open':
            depth += 1
        if t.type == 'paren_close':
            depth -= 1
    return split_index


def name_generator():
    # -> a, b, c, d, e, ..., z, aa, ab, ac, ..., az, ba, bb, ...
    digits = []
    while True:
        idx = 0
        while idx < len(digits) and digits[idx] == 25:
            digits[idx] = 0
            idx += 1
        if idx < len(digits):
            digits[idx] += 1
        else:
            digits.append(0)
        yield ''.join('abcdefghijklmnopqrstuvwxyz'[i] for i in reversed(digits))


class Expression:
    def __init__(self, tokens):
        if not tokens:
            raise ValueError('Parsing error, cannot have empty expression')
        split_index = partition_last(tokens)
        if split_index > 0:
            self.type = 'invocation'
            self.func = Expression(tokens[:split_index])
            self.arg = Expression(tokens[split_index:])
        else:
            if tokens[0].type == 'paren_open':
                if len(tokens) < 2 or tokens[-1].type != 'paren_close':
                    raise ValueError('Parsing error, unbalanced parentheses')
                self.__init__(tokens[1:-1])
            elif tokens[0].type == 'lambda':
                if len(tokens) < 3 or tokens[1].type != 'variable' or tokens[2].type != 'lambda_delim':
                    raise ValueError(
                        'Parsing error, improper lambda declaration')
                self.type = 'lambda'
                self.var = tokens[1].value
                self.body = Expression(tokens[3:])
            elif tokens[0].type == 'variable':
                self.type = 'variable'
                self.value = tokens[0].value
                if len(tokens) > 1:
                    raise ValueError(
                        'Parsing error, unbalanced parentheses after variable')
            elif tokens[0].type == 'macro':
                self.type = 'macro'
                self.value = tokens[0].value
                if len(tokens) > 1:
                    raise ValueError(
                        'Parsing error, unbalanced parentheses after macro')
            else:
                raise ValueError(
                    'Parsing error, invalid start of expression ' + tokens[0].type)

    def __str__(self):
        if self.type == 'invocation':
            return f'({self.func} {self.arg})'
        elif self.type == 'lambda':
            return f'(λ{self.var}.{self.body})'
        elif self.type == 'variable' or self.type == 'macro':
            return self.value
        else:
            raise ValueError('Cannot determine type of Expression')

    def macrosub(self, macros, bound_vars=set()):
        # Shallow macro substitution given a dict of previously-subbed macros
        if self.type == 'macro':
            if self.value not in macros:
                raise ValueError('Macro ' + self.value +
                                 ' not defined during substitution.')
            return copy.deepcopy(macros[self.value]).rename_vars(bound_vars)
        if self.type == 'invocation':
            self.func = self.func.macrosub(macros, bound_vars)
            self.arg = self.arg.macrosub(macros, bound_vars)
        if self.type == 'lambda':
            self.body = self.body.macrosub(macros, bound_vars | {self.var})
        return self

    def subst(self, variable, value, bound_vars=set()):
        value = value.rename_vars(self.var_names(), bound_vars)
        ret = self._subst(variable, value)
        return ret

    def _subst(self, variable, value):
        if self.type == 'invocation':
            self.func = self.func._subst(variable, value)
            self.arg = self.arg._subst(variable, value)
        if self.type == 'lambda':
            if self.var == variable:
                raise ValueError('Duplicate variable name found')
            self.body = self.body._subst(variable, value)
        if self.type == 'variable':
            if self.value == variable:
                return copy.deepcopy(value)
        return self

    def rename_vars(self, conflicts, bound_vars=set()):
        my_names = self.var_names()
        all_names = conflicts | bound_vars | my_names
        g = (n for n in name_generator() if n not in all_names)
        sub_table = {}
        for name in conflicts & my_names:
            sub_table[name] = next(g)
        return self._rename_vars(sub_table)

    def rename_long_vars(self):
        my_names = self.var_names()
        g = (n for n in name_generator() if len(n) >= 2 or n not in my_names)
        sub_table = {}
        for name in my_names:
            if len(name) >= 2:
                sub_table[name] = next(g)
        return self._rename_vars(sub_table)

    def _rename_vars(self, sub_table):
        if self.type == 'invocation':
            self.func = self.func._rename_vars(sub_table)
            self.arg = self.arg._rename_vars(sub_table)
        if self.type == 'lambda':
            if self.var in sub_table:
                self.var = sub_table[self.var]
            self.body = self.body._rename_vars(sub_table)
        if self.type == 'variable':
            if self.value in sub_table:
                self.value = sub_table[self.value]
        return self

    def var_names(self, include_free=False):
        if self.type == 'invocation':
            return self.func.var_names(include_free) | self.arg.var_names(include_free)
        if self.type == 'lambda':
            ret = self.body.var_names(include_free)
            ret.add(self.var)
            return ret
        if include_free and self.type == 'variable':
            return {self.value}
        return set()

    def reduce_cbn(self, bound_vars=set()):
        if self.type == 'invocation':
            e = self.func.reduce_cbn(bound_vars)
            if e.type == 'lambda':
                return e.body.subst(e.var, self.arg, bound_vars).reduce_cbn(bound_vars)
            else:
                self.func = e
        return self

    def reduce_eta(self):
        if (self.type == 'lambda' and
            self.body.type == 'invocation' and
            self.body.arg.type == 'variable' and
            self.body.arg.value == self.var and
                self.var not in self.body.func.var_names(True)):
            return self.body.func
        return self

    def reduce_normal(self, bound_vars=set()):
        if self.type == 'lambda':
            self.body = self.body.reduce_normal(bound_vars | {self.var})
            return self.reduce_eta()
        if self.type == 'invocation':
            e = self.func.reduce_cbn(bound_vars)
            if e.type == 'lambda':
                return e.body.subst(e.var, self.arg, bound_vars).reduce_normal(bound_vars)
            else:
                self.func = e.reduce_normal(bound_vars)
                self.arg = self.arg.reduce_normal(bound_vars)
        return self


class MacroDeclaration:
    def __init__(self, tokens):
        self.reduce = True
        if tokens[0].type == 'noreduce':
            self.reduce = False
            tokens.pop(0)
        if len(tokens) < 2 or tokens[0].type != 'macro' or tokens[1].type != 'assignment':
            raise ValueError('Parsing error, expected macro assignment')
        self.macro = tokens[0].value
        self.expr = Expression(tokens[2:])

    def __str__(self):
        return f'{"^" if not self.reduce else ""}{self.macro} = {self.expr}'

    def macrosub(self, macros):
        self.expr = self.expr.macrosub(macros)
        if self.reduce:
            self.expr = self.expr.reduce_normal()
        return self


class Program:
    def __init__(self, tokens):
        lines = []
        while tokens:
            lineStart = len(tokens)
            while lineStart > 0 and tokens[lineStart - 1].type != 'linebreak':
                lineStart -= 1
            if lineStart < len(tokens):
                lines.append(tokens[lineStart:])
            tokens = tokens[:(lineStart - 1 if lineStart else 0)]
        self.statements = []
        for line in reversed(lines):
            if any(t.type == 'assignment' for t in line):
                self.statements.append(MacroDeclaration(line))
            else:
                self.statements.append(Expression(line))
        self.macrosub()

    @classmethod
    def from_string(cls, program_string):
        tokens = tokenize(program_string)
        return cls(tokens)

    def __str__(self):
        return '\n'.join(str(statement) for statement in self.statements)

    def macrosub(self):
        macros = {}
        for i, st in enumerate(self.statements):
            st = st.macrosub(macros)
            if isinstance(st, MacroDeclaration):
                macros[st.macro] = st.expr
            else:
                self.statements[i] = st
        return self

    def evaluate(self):
        for st in self.statements:
            if isinstance(st, Expression):
                yield st.reduce_normal()


def run_repl(program):
    from colorama import init, Fore, Style
    init()
    macros = {}
    for st in program.statements:
        if isinstance(st, MacroDeclaration):
            macros[st.macro] = st.expr
    while True:
        try:
            try:
                line = input(Fore.GREEN + '> ' + Fore.YELLOW)
            finally:
                print(Style.RESET_ALL, end='')

            if line.strip() == '!help':
                print(__doc__)
                continue

            line = tokenize(line)
            if line:
                try:
                    if any(t.type == 'assignment' for t in line):
                        st = MacroDeclaration(line).macrosub(macros)
                        macros[st.macro] = st.expr
                    else:
                        st = Expression(line).macrosub(macros)
                        out = str(st.reduce_normal().rename_long_vars())
                        print(Fore.LIGHTBLUE_EX + out + Style.RESET_ALL)
                except RecursionError:
                    print(Fore.RED + 'Infinite reduction' + Style.RESET_ALL)
        except ValueError as exc:
            print(Fore.RED + str(exc) + Style.RESET_ALL)


def main():
    print('=============== PROGRAM ===============')
    p = Program.from_string(open('program.txt', 'r').read())
    print(p)
    print('\n========== EVALUATION RESULT ==========')
    for expr in p.evaluate():
        print(expr)
    print('\n================ REPL =================')
    print(r'Try `ADD (\f.\x.f (f x)) (\f.\x.f x)`, or type `!help` to learn more')
    run_repl(p)


if __name__ == '__main__':
    main()

