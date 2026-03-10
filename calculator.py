import re


def calculator(expression: str) -> str:
    allowed = re.fullmatch(r"[0-9\.\+\-\*\/\(\) ]+", expression)
    if not allowed:
        return "ERROR: unsupported characters in expression."
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as exc:
        return f"ERROR: {exc}"