from fastapi import HTTPException, status


class ScanImageException(HTTPException):
    def __init__(self, message: str, status_code: int = status.HTTP_400_BAD_REQUEST):
        self.message = message
        self.status_code = status_code
        super().__init__(status_code=status_code, detail=message)
