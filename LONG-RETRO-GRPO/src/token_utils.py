def truncate_from_left(token_ids: list[int], max_tokens: int) -> list[int]:
    if max_tokens <= 0:
        return token_ids
    if len(token_ids) <= max_tokens:
        return token_ids
    return token_ids[-max_tokens:]
