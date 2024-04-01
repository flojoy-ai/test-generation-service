def enforce_code_snippet(dirty_input: str):
    """Ensures that the output is a code snippet"""
    if "```" in dirty_input:
        dirty_input = dirty_input.split("```")[1]
        dirty_input = dirty_input[dirty_input.index("\n") + 1 :]
    return dirty_input
