"""Deterministic arithmetic and date calculations whose inputs are checked against cited sources."""
import ast
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation, localcontext
import re

NUMBER = re.compile(r'(?<![\w.])-?\d[\d,]*(?:\.\d+)?(?!\w|\.\d)')
WORDS = {w: Decimal(i) for i, w in enumerate(
    'zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen '
    'fifteen sixteen seventeen eighteen nineteen twenty'.split())}
MAX_EXPRESSION = 500
ALWAYS_ALLOWED = {Decimal(0), Decimal(1), Decimal(100)}  # identities and percent conversion


MONTHS = ('january|february|march|april|may|june|july|august|september|october|november|december|'
          'jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec')
DATES = re.compile(r'\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b'
                   rf'|\b(?:{MONTHS})\.?\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,?\s+\d{{4}})?\b', re.I)


def extract_numbers(text):
    """Numbers written as digits (commas allowed) or as the words zero..twenty.

    Dates are removed first, so "August 20, 2026" does not supply 20 or 2026 as a quantity.
    """
    text = DATES.sub(' ', text)
    found = set()
    for match in NUMBER.finditer(text):
        try:
            found.add(Decimal(match.group().replace(',', '')))
        except InvalidOperation:
            pass
    for word in re.findall(r'[a-z]+', text.casefold()):
        if word in WORDS:
            found.add(WORDS[word])
    return found


def _evaluate(node, names):
    if isinstance(node, ast.Expression):
        return _evaluate(node.body, names)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return Decimal(str(node.value))
    if isinstance(node, ast.Name):
        if node.id not in names:
            raise ValueError(f'Unknown input name: {node.id}')
        return names[node.id]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _evaluate(node.operand, names)
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
        left, right = _evaluate(node.left, names), _evaluate(node.right, names)
        if isinstance(node.op, ast.Div):
            if right == 0:
                raise ValueError('Division by zero')
            return left / right
        return {ast.Add: left + right, ast.Sub: left - right, ast.Mult: left * right}[type(node.op)]
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id in {'min', 'max'} and node.args and not node.keywords):
        values = [_evaluate(arg, names) for arg in node.args]
        return min(values) if node.func.id == 'min' else max(values)
    raise ValueError('Expressions support numbers, input names, + - * /, parentheses, min() and max()')


def calculate(params, sources):
    """Evaluate an expression of literal numbers; each literal must appear in a cited source.

    `sources` maps evidence IDs (and "question") to text. 100 is always allowed for percentages.
    """
    expression = params['expression']
    if not isinstance(expression, str) or len(expression) > MAX_EXPRESSION:
        raise ValueError('expression must be a string of at most 500 characters')
    cited = params['sources']
    unknown = [source for source in cited if source not in sources]
    if unknown:
        raise ValueError(f'Unknown sources {unknown}; cite existing evidence IDs or "question"')
    # Replace each literal with a name bound to an exact Decimal, so 88.00 keeps its precision.
    literals = {}
    def bind(match):
        name = f'n{len(literals)}'
        literals[name] = Decimal(match.group().replace(',', ''))
        return name
    try:
        tree = ast.parse(re.sub(r'\d[\d,]*(?:\.\d+)?|\.\d+', bind, expression), mode='eval')
    except SyntaxError:
        raise ValueError('expression must be arithmetic on numbers, e.g. "(176.00 - 50.00) * 20 / 100"') from None
    with localcontext() as context:
        context.prec = 28
        result = _evaluate(tree, literals)
    available = set().union(*(extract_numbers(sources[source]) for source in cited)) | ALWAYS_ALLOWED
    missing = sorted({str(value) for value in literals.values() if value not in available})
    if missing:
        raise ValueError(f'Numbers {missing} do not appear in cited sources {cited}. Use numbers exactly as '
                         'written in the evidence or question, or cite the source that contains them.')
    return {'label': params['label'], 'expression': expression, 'sources': cited,
            'result': str(result), 'method': 'deterministic decimal arithmetic'}


def _date(value, field):
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError):
        raise ValueError(f'{field} must be an ISO date (YYYY-MM-DD)') from None


def date_calculate(params):
    """add_days, days_between (second minus first) or compare two ISO dates."""
    operation = params['operation']
    first = _date(params.get('date'), 'date')
    if operation == 'add_days':
        days = params.get('days')
        if type(days) is not int:
            raise ValueError('add_days requires integer days')
        result = (first + timedelta(days=days)).isoformat()
    else:
        second = _date(params.get('other_date'), 'other_date')
        if operation == 'days_between':
            result = (second - first).days
        elif operation == 'compare':
            result = 'before' if first < second else 'after' if first > second else 'same day'
        else:
            raise ValueError('operation must be add_days, days_between or compare')
    return {'label': params.get('label', ''), 'operation': operation,
            'inputs': {k: params[k] for k in ('date', 'other_date', 'days') if k in params},
            'sources': params.get('sources', []), 'result': result,
            'method': 'deterministic calendar arithmetic; compare reports date relative to other_date'}
