from returns.io import IOResultE, IOSuccess
from returns.result import Success


def unwrap_result[T](result: IOResultE[T]) -> T:
    match result:
        case IOSuccess(Success(value)) | Success(value):
            return value
        case _:
            assert False, "failed to unwrap"
