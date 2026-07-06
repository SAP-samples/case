from dataclasses import dataclass


@dataclass(frozen=True)
class SerializationConfig:
    """Configuration options for table serialization."""

    max_length: int = 16_000
    max_cell_tokens: int = 50
    cell_separator: str = '    '
    scientific_notation: bool = True
    with_header: bool = True
