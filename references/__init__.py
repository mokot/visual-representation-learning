import sys

# Allow importing only `lib_bilagrid.py` from the package
if not sys.modules.get("__main__") or sys.modules["__main__"].__file__.endswith(
    "lib_bilagrid.py"
):
    # Import your required functions or classes from lib_bilagrid.py here
    from .lib_bilagrid import slice, total_variation_loss, BilateralGrid
else:
    raise ImportError(
        "The 'references' package is not meant to be imported. "
        "These scripts are retained for reference purposes only."
    )
