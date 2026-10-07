"""Documentation-only diagnostic of the unchanged helper; not a production test."""
import json
import numpy as np
from exp12_helpers import max_relative_deviation

# Run with JAX_PLATFORMS=cpu and PYTHONPATH=main/tests:main from repo root.
cases = {
    "integer_difference": ([np.array([11], np.int32)], [np.array([10], np.int32)]),
    "nan_hides_finite_difference": ([np.array([2., np.nan])], [np.array([1., np.nan])]),
    "unexpected_nan": ([np.array([np.nan])], [np.array([1.])]),
}
print(json.dumps({name: max_relative_deviation(*pairs) for name, pairs in cases.items()}, indent=2))
