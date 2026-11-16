from dataclasses import dataclass


@dataclass(frozen=True)
class SerializationConfig:
    """Configuration options for table serialization."""

    max_length: int = 16_000
    max_cell_tokens: int = 50
    cell_separator: str = '    '
    scientific_notation: bool = True
    with_header: bool = True

    def __post_init__(self):
        for name in ('max_length', 'max_cell_tokens'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f'{name} must be a positive integer.')
        if not isinstance(self.cell_separator, str) or not self.cell_separator:
            raise ValueError('cell_separator must be a non-empty string.')
        for name in ('scientific_notation', 'with_header'):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f'{name} must be a boolean.')
