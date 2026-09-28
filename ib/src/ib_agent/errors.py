"""Error hierarchy. The CLI turns any `IBAgentError` into a clean message and exit code 1."""


class IBAgentError(Exception):
    """Base class for expected, user-facing failures."""


class ConfigError(IBAgentError):
    pass


class InvalidTickerError(IBAgentError):
    pass


class TickerNotFoundError(IBAgentError):
    pass


class EdgarError(IBAgentError):
    pass


class UnsupportedCompanyError(IBAgentError):
    pass


class FilingNotFoundError(IBAgentError):
    pass
