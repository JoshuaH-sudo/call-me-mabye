"""Parameter extractor sub-package.

Provides three specialised extractors that pull candidate values out of a
raw prompt string:

* :class:`~src.decoder.extractors.number.NumberParameterExtractor`  — numeric
  literals and English number words.
* :class:`~src.decoder.extractors.regex.RegexParameterExtractor`    — regex
  patterns inferred from prompt keywords or quoted literals.
* :class:`~src.decoder.extractors.string.StringParameterExtractor`  — string
  values extracted from quoted spans and bare words.
"""
from .number import NumberParameterExtractor
from .regex import RegexParameterExtractor
from .string import StringParameterExtractor

__all__ = [
    "NumberParameterExtractor",
    "RegexParameterExtractor",
    "StringParameterExtractor",
]
