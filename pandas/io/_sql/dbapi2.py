from abc import abstractmethod
from collections.abc import (
    Generator,
    Iterator,
    Sequence,
)
from typing import (
    Literal,
    Protocol,
)

from pandas._typing import (
    DtypeArg,
    DtypeBackend,
)

from pandas.core.api import DataFrame

from pandas.io._sql.base import (
    DatesParsingArgs,
    wrap_result,
)


class DBAPI2Cursor(Protocol):
    description: Sequence[Sequence]
    arraysize: int = 1

    @property
    def rowcount(self) -> int: ...
    def close(self) -> None: ...
    def execute(self, query: str, parameters: Sequence | None) -> None: ...
    def executemany(self, query: str, params: Sequence[Sequence] | None) -> None: ...
    def fetchone(self) -> Sequence | None: ...
    def fetchmany(self, size: int = arraysize) -> Sequence[Sequence]: ...
    def fetchall(self) -> Sequence[Sequence]: ...
    def setinputsizes(self, sizes: Sequence) -> None: ...
    def setoutputsize(self, size: int, column: int | None) -> None: ...
    @abstractmethod
    def nextset(self) -> bool | None: ...
    @abstractmethod
    def callproc(self, procname: str, args: Sequence | None) -> None: ...


class DBAPI2Connection(Protocol):
    def close(self) -> None: ...
    def commit(self) -> None: ...
    def cursor(self) -> DBAPI2Cursor: ...
    @abstractmethod
    def rollback(self) -> None: ...


class PandasDBAPI2Interface:
    def __init__(self, conn: DBAPI2Connection):
        self.conn = conn

    @staticmethod
    def _query_iterator(
        cursor: DBAPI2Cursor,
        chunksize: int,
        columns: list[str],
        index_col: str | list[str] | None = None,
        coerce_float: bool = True,
        parse_dates: DatesParsingArgs = None,
        dtype: DtypeArg | None = None,
        dtype_backend: DtypeBackend | Literal["numpy"] = "numpy",
    ) -> Generator[DataFrame]:
        has_read_data = False
        while True:
            data = cursor.fetchmany(chunksize)
            if type(data) == tuple:
                data = list(data)
            if not data:
                cursor.close()
                if not has_read_data:
                    result = DataFrame.from_records(
                        [], columns=columns, coerce_float=coerce_float
                    )
                    if dtype:
                        result = result.astype(dtype)
                    yield result
                break

            has_read_data = True
            yield wrap_result(
                data,
                columns,
                index_col=index_col,
                coerce_float=coerce_float,
                parse_dates=parse_dates,
                dtype=dtype,
                dtype_backend=dtype_backend,
            )

    def read_query(
        self,
        query: str,
        params: Sequence | None = None,
        index_col: str | list[str] | None = None,
        coerce_float: bool = True,
        parse_dates: DatesParsingArgs = None,
        chunksize: int | None = None,
        dtype: DtypeArg | None = None,
        dtype_backend: DtypeBackend | Literal["numpy"] = "numpy",
    ) -> DataFrame | Iterator[DataFrame]:
        cursor = self.conn.cursor()
        cursor.execute(query, params)
        columns = [x[0] for x in cursor.description]
        if chunksize is not None:
            return self._query_iterator(
                cursor,
                chunksize,
                columns,
                index_col=index_col,
                coerce_float=coerce_float,
                parse_dates=parse_dates,
                dtype=dtype,
                dtype_backend=dtype_backend,
            )
        data = cursor.fetchall()
        data = list(data)
        cursor.close()
        return wrap_result(data, columns)
