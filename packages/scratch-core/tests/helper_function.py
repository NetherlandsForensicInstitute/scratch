from returns.io import IOResultE, IOSuccess
from returns.result import ResultE, Success


def unwrap_result[T](result: IOResultE[T] | ResultE[T]) -> T:
    match result:
        case IOSuccess(Success(value)) | Success(value):
            return value
        case _:
            assert False, "failed to unwrap"
