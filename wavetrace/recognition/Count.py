"""Display naming for the people-count head's classes (count = presence with N classes)."""


def countName(c: int, max_count: int) -> str:
    """Display label for a count class: the top level is the open-ended 'N+' bin."""
    return f"{c}+" if c >= max_count else str(c)
