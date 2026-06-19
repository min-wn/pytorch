__all__ = ["limit_core_num"]


def limit_core_num(op_aicore_num: int, op_vectorcore_num: int):
    import npugraph_ex

    return npugraph_ex.scope.limit_core_num(
        op_aicore_num, op_vectorcore_num)
