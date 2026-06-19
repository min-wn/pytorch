__all__ = ["compile_fx", "register_replacement"]

from collections.abc import Iterable
from typing import Any, Callable, Match, Optional, Union

try:
    from torch._inductor.pattern_matcher import (
        fwd_only,
        PatternExpr,
        ReplaceFn,
        SearchFn,
        TraceFn,
    )
except ImportError:
    from torch._inductor.pattern_matcher import inference_graph as fwd_only

    PatternExpr = Any
    ReplaceFn = Any
    SearchFn = Any
    TraceFn = Any

from . import inference
from . import scope


def compile_fx(gm, example_inputs=None, options=None):
    try:
        import npugraph_ex
    except ModuleNotFoundError as e:
        if e.name != "npugraph_ex":
            raise
        import torchair

        config = torchair.CompilerConfig()
        config.mode = "reduce-overhead"
        return torchair.get_npu_backend(compiler_config=config)(
            gm, example_inputs)
    else:
        return npugraph_ex.compile_fx(gm, example_inputs, options)


def _return_true(match: Match):
    return True


def register_replacement(
    search_fn: SearchFn,
    replace_fn: ReplaceFn,
    example_inputs: Iterable[Any],
    trace_fn: TraceFn = fwd_only,
    extra_check: Callable[[Match], bool] = _return_true,
    search_fn_pattern: Optional[PatternExpr] = None,
    scalar_workaround: Optional[dict[str, Union[float, int]]] = None,
    skip_duplicates: bool = False,
):
    import npugraph_ex

    return npugraph_ex.patterns.pattern_pass_manager.register_replacement(
        search_fn,
        replace_fn,
        example_inputs,
        trace_fn=trace_fn,
        extra_check=extra_check,
        search_fn_pattern=search_fn_pattern,
        scalar_workaround=scalar_workaround,
        skip_duplicates=skip_duplicates,
    )
