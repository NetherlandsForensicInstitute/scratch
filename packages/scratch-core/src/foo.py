def foo(x: str) -> str:
    """Return a formatted string prefixed with 'bar'.

    Parameters
    ----------
    x : str
        Input string to be prefixed.

    Returns
    -------
    str
        The formatted string in the form ``"bar <x>"``.

    Examples
    --------
    >>> foo("test")
    'bar test'
    """
    return f"bar {x}"
