"""Single literal product-version source.

The product version is intentionally a plain string literal so build backends
can read it statically (``pyproject.toml`` dynamic version). It is distinct
from the public API version (``zecalibrator.api.v1.API_VERSION``) and from the
science-contract, provenance-schema, library-schema and matching-policy
versions. Do not derive the API version from this value.
"""

__version__ = "0.0.1"
