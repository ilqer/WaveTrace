
def count_name(count: int, max_count: int) -> str:
    """The top class is the open-ended 'N+' bin."""
    return f"{count}+" if count >= max_count else str(count)
