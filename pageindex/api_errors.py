"""User-facing error messages for API and ingestion failures."""


def format_user_error(exc: Exception) -> str:
    msg = str(exc)
    low = msg.lower()
    if "limitreached" in low or "limit reached" in low:
        return (
            "PageIndex upload limit reached for your API key. "
            "Add credits or raise your plan at https://dash.pageindex.ai"
        )
    if "openrouter" in low:
        return (
            "OpenRouter is disabled in this app. Restart with .\\start.ps1 "
            "and ensure PAGEINDEX_API_KEY is set in .env (not OPENROUTER_API_KEY)."
        )
    if "pageindex_api_key" in low or ("api key" in low and "pageindex" in low):
        return "Set PAGEINDEX_API_KEY in .env — https://dash.pageindex.ai/api-keys"
    if "insufficientcredits" in low or "402" in low or "credit" in low or "quota" in low or "payment" in low:
        return "PageIndex API credits exhausted — add credits at https://dash.pageindex.ai"
    if "401" in low or ("403" in low and "invalid" in low and "key" in low) or "authentication" in low:
        return "Invalid PAGEINDEX_API_KEY — check https://dash.pageindex.ai/api-keys"
    if "timed out" in low:
        return "PageIndex API timed out waiting for document processing. Try again."
    return msg
