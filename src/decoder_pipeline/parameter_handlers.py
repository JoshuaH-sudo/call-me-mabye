"""
Parameter-type handlers for state-machine constrained decoding.

Each handler encapsulates the per-type state and character-level
transition logic for one JSON value kind, so that the decoder stays
type-agnostic.  Adding support for a new parameter type means adding
a new ParameterSegmentHandler subclass and registering it in
create_parameter_handler().
"""
import re
from abc import ABC, abstractmethod
from typing import Any

# Characters that can appear inside a JSON number token.
_NUMBER_CHARS = frozenset("-+0123456789.eE")

# A complete JSON number (integer, decimal, or scientific notation).
_NUMBER_COMPLETE_RE = re.compile(
    r"^-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?$"
)

# A valid *prefix* of a JSON number — may not yet be complete.
_NUMBER_PREFIX_RE = re.compile(
    r"^(?:"
    r""                                     # empty (before any char)
    r"|-"                                   # just minus sign
    r"|-?(?:0|[1-9]\d*)"                   # integer part
    r"|-?(?:0|[1-9]\d*)\."                 # trailing dot
    r"|-?(?:0|[1-9]\d*)\.\d*"              # decimal digits
    r"|-?(?:0|[1-9]\d*)(?:\.\d+)?[eE]"    # exponent marker
    r"|-?(?:0|[1-9]\d*)(?:\.\d+)?[eE][+-]"  # exponent sign
    r"|-?(?:0|[1-9]\d*)(?:\.\d+)?[eE][+-]?\d+"  # exponent digits
    r")$"
)


class ParameterSegmentHandler(ABC):
    """
    Abstract base class for handling a specific JSON parameter type during
    state-machine constrained decoding.

    A handler is responsible for:
    - Providing a fresh initial state dict (initial_state).
    - Consuming characters from a token chunk and updating state
      (consume_chunk).
    - Reporting whether the parameter value is fully decoded (is_complete).
    - Reporting whether the current state is still a valid prefix that
      could lead to a complete value (is_valid_prefix).

    The main decoder iterates over vocabulary tokens and, for each one,
    calls consume_chunk on every live state.  States that return None are
    pruned.  When is_complete returns True the decoder advances to the
    next output segment.
    """

    @property
    @abstractmethod
    def type_name(self) -> str:
        """The parameter type string this handler covers (e.g. "string")."""
        ...

    @abstractmethod
    def initial_state(self) -> dict[str, Any]:
        """Return a fresh initial state dict for this handler."""
        ...

    @abstractmethod
    def consume_chunk(
        self,
        state: dict[str, Any],
        chunk: str,
        start: int,
    ) -> tuple[dict[str, Any] | None, int]:
        """
        Attempt to consume characters from chunk[start:].

        Returns:
            (None, start)           — chunk is not a valid continuation.
            (new_state, new_index)  — new_state after consuming chars.

        When is_complete(new_state) is True the full parameter value has
        been decoded.  new_index points to the first character that was
        NOT consumed by this handler (i.e. the first char of the next
        segment).  The caller is responsible for advancing the segment.
        """
        ...

    @abstractmethod
    def is_complete(self, state: dict[str, Any]) -> bool:
        """Return True when the parameter value is fully decoded."""
        ...

    @abstractmethod
    def is_valid_prefix(self, state: dict[str, Any]) -> bool:
        """
        Return True when state represents a valid but not yet complete
        prefix.  Used after a token chunk is fully consumed to prune
        states that cannot lead to a valid value.
        """
        ...


class StringParameterHandler(ParameterSegmentHandler):
    """
    Handles JSON string values enclosed in double quotes: ``"any text"``.

    State keys:
        started  — True after the opening ``"`` has been consumed.
        escaped  — True when the previous character was ``\\``.
        complete — True after the closing ``"`` has been consumed.
    """

    @property
    def type_name(self) -> str:
        return "string"

    def initial_state(self) -> dict[str, Any]:
        return {"started": False, "escaped": False, "complete": False}

    def consume_chunk(
        self,
        state: dict[str, Any],
        chunk: str,
        start: int,
    ) -> tuple[dict[str, Any] | None, int]:
        s: dict[str, Any] = dict(state)
        index = start

        if s["complete"]:
            return s, index

        while index < len(chunk):
            char = chunk[index]

            if not s["started"]:
                # Expect the opening double-quote.
                if char != '"':
                    return None, start
                s["started"] = True
                index += 1
                continue

            if s["escaped"]:
                # The previous character was a backslash; accept any char.
                s["escaped"] = False
                index += 1
                continue

            if char == "\\":
                s["escaped"] = True
                index += 1
                continue

            if char == '"':
                # Closing double-quote: string is complete.
                s["complete"] = True
                index += 1
                return s, index

            # Control characters are not valid inside a JSON string.
            if ord(char) < 0x20:
                return None, start

            index += 1

        return s, index

    def is_complete(self, state: dict[str, Any]) -> bool:
        return bool(state.get("complete"))

    def is_valid_prefix(self, state: dict[str, Any]) -> bool:
        # Any in-progress string state (not mid-escape-sequence oddity)
        # is a valid prefix; consume_chunk already rejects invalid chars.
        return True


class NumberParameterHandler(ParameterSegmentHandler):
    """
    Handles JSON number values (integers and floating-point).

    Numbers have no explicit closing character.  A number is considered
    complete when a non-number character is encountered (e.g. ``,`` or
    ``}``).  That character is NOT consumed — it belongs to the next
    segment.

    State keys:
        buffer   — digits (and optional sign/dot/exponent) accumulated so far.
        complete — True once a non-number char signals the end of the number.
    """

    @property
    def type_name(self) -> str:
        return "number"

    def initial_state(self) -> dict[str, Any]:
        return {"buffer": "", "complete": False}

    def consume_chunk(
        self,
        state: dict[str, Any],
        chunk: str,
        start: int,
    ) -> tuple[dict[str, Any] | None, int]:
        s: dict[str, Any] = dict(state)
        index = start

        if s["complete"]:
            return s, index

        while index < len(chunk):
            char = chunk[index]

            if char in _NUMBER_CHARS:
                s["buffer"] += char
                index += 1
                continue

            # Non-number character: number must be fully valid here.
            if _NUMBER_COMPLETE_RE.fullmatch(s["buffer"]):
                s["complete"] = True
                # Do NOT consume the non-number char; it belongs to the
                # next segment (e.g. "," or "}").
                return s, index

            return None, start

        return s, index

    def is_complete(self, state: dict[str, Any]) -> bool:
        # Complete only when explicitly marked — NOT just because the
        # buffer happens to be a valid number.  Completion is triggered by
        # seeing the first non-number character, which tells us the number
        # token sequence has ended.
        return bool(state.get("complete"))

    def is_valid_prefix(self, state: dict[str, Any]) -> bool:
        buf = state.get("buffer", "")
        return bool(_NUMBER_PREFIX_RE.fullmatch(buf))


class UnsupportedParameterHandler(ParameterSegmentHandler):
    """
    Stub handler for parameter types that are not yet implemented.

    This class is returned by create_parameter_handler() for any type
    that is not "string" or "number".  Its presence signals to the
    decoder that the enclosing function should be *skipped* rather than
    decoded.

    None of the handler methods are intended to be called at runtime;
    they all raise NotImplementedError to make accidental misuse visible.
    Implement this class (or a dedicated subclass) when adding support
    for the corresponding parameter type.
    """

    def __init__(self, unsupported_type: str) -> None:
        self._unsupported_type = unsupported_type

    @property
    def type_name(self) -> str:
        return self._unsupported_type

    @property
    def _not_implemented_msg(self) -> str:
        return (
            f"parameter type '{self._unsupported_type}' is not yet supported"
        )

    def initial_state(self) -> dict[str, Any]:
        raise NotImplementedError(self._not_implemented_msg)

    def consume_chunk(
        self,
        state: dict[str, Any],
        chunk: str,
        start: int,
    ) -> tuple[dict[str, Any] | None, int]:
        raise NotImplementedError(self._not_implemented_msg)

    def is_complete(self, state: dict[str, Any]) -> bool:
        raise NotImplementedError(self._not_implemented_msg)

    def is_valid_prefix(self, state: dict[str, Any]) -> bool:
        raise NotImplementedError(self._not_implemented_msg)


def create_parameter_handler(type_name: str) -> ParameterSegmentHandler:
    """
    Return the appropriate handler for the given parameter type name.

    Known types:
        "string"  -> StringParameterHandler
        "number"  -> NumberParameterHandler

    Any other type returns an UnsupportedParameterHandler, which is a
    stub signalling that the type is not yet implemented.  The decoder
    inspects the returned handler and skips any function that contains
    an unsupported type rather than attempting to decode it.
    """
    if type_name == "string":
        return StringParameterHandler()
    if type_name == "number":
        return NumberParameterHandler()
    return UnsupportedParameterHandler(type_name)
