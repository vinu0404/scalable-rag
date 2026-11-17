"""Describe embedding output before it is saved to a vector store.

An embedding adapter produces numbers; a vector-store adapter saves them.
Keeping this record independent lets users replace either adapter.
"""

import math
from dataclasses import dataclass, field

from rag_ingest.domain.chunks import Chunk


@dataclass(frozen=True, kw_only=True)
class EmbeddedChunk:
    """Pair one chunk with its vector and the pipeline that produced it.

    Example: a model returns three numbers for a chunk, represented as
    vector=(0.1, -0.2, 0.3) with dimensions=3. Real model dimensions depend
    on the selected configuration, not on this class.

    dimensions must come from the expected configuration, not merely the
    returned vector length. The adapter must map each provider result to the
    correct chunk. The pipeline fingerprint links to saved model settings.

    This record does not call a model, save vectors, or make data searchable.
    Batch sizes and vector dimensions must also be bounded by runtime policy.
    """

    chunk: Chunk
    pipeline_fingerprint: str
    dimensions: int
    vector: tuple[float, ...] = field(repr=False)

    def __post_init__(self) -> None:
        """Reject invalid records before an indexing adapter receives them.

        Require a tuple of finite Python floats whose count matches the
        expected dimensions. Adapters must convert provider number types to
        Python floats before constructing this record. Reject NaN and infinity.
        These checks do not prove the vector came from the expected model.
        """
        if not isinstance(self.chunk, Chunk):
            raise ValueError("chunk must be a Chunk")
        value = self.pipeline_fingerprint
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)
        ):
            raise ValueError(
                "pipeline_fingerprint must be a lowercase SHA-256 hex digest"
            )
        if type(self.dimensions) is not int or self.dimensions <= 0:
            raise ValueError("dimensions must be a positive integer")
        if not isinstance(self.vector, tuple) or len(self.vector) != self.dimensions:
            raise ValueError("vector must be a tuple matching the expected dimensions")
        if any(
            type(number) is not float or not math.isfinite(number)
            for number in self.vector
        ):
            raise ValueError("vector values must be finite Python floats")
