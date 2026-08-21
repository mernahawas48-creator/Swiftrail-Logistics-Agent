class StateGraphError(RuntimeError):
    """Base error for state-graph lifecycle problems."""


class RunNotFoundError(StateGraphError):
    pass


class InvalidRunStatusError(StateGraphError):
    pass


class InvalidTransitionError(StateGraphError):
    pass
