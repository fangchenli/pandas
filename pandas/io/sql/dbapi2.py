"""
Generic DB-API 2.0 (PEP 249) backend for ``read_sql`` / ``to_sql``.

Builds SQL statements as strings and runs them through the connection's
cursor, without SQLAlchemy. This is what pandas uses for any connection that
is not a SQLAlchemy connectable, an ADBC connection or a ``sqlite3``
connection. The driver's ``paramstyle`` and ``Error`` class are read from its
module, as PEP 249 specifies. Anything dialect-specific beyond that (column
types other than the defaults, placeholders in the user's own queries) is the
user's responsibility.
"""

from __future__ import annotations

from contextlib import (
    contextmanager,
    suppress,
)
import re
import sys
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    cast,
)
import warnings

from pandas._libs import lib
from pandas.errors import (
    DatabaseError,
)
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.common import (
    is_dict_like,
    is_list_like,
)
from pandas.core.dtypes.dtypes import DatetimeTZDtype

from pandas.core.api import (
    DataFrame,
)

from pandas.io.sql.base import (
    PandasSQL,
    SQLTable,
    get_unicode_name,
    wrap_result,
)

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Generator,
        Hashable,
        Iterator,
        Mapping,
        Sequence,
    )
    from types import ModuleType

    from sqlalchemy.sql.expression import (
        Select,
        TextClause,
    )

    from pandas._typing import (
        DtypeArg,
        DtypeBackend,
        DtypeObj,
    )

_PARAMSTYLES = ("qmark", "numeric", "named", "format", "pyformat")

# Column types used when ``dtype=`` doesn't give one, keyed by
# ``lib.infer_dtype``. Users of databases that spell these differently
# (e.g. Oracle, SQL Server) pass their own types via ``dtype=``.
_SQL_TYPES = {
    "string": "TEXT",
    "floating": "DOUBLE PRECISION",
    "integer": "BIGINT",
    "datetime": "TIMESTAMP",
    "datetimetz": "TIMESTAMP WITH TIME ZONE",
    "date": "DATE",
    "time": "TIME",
    "boolean": "BOOLEAN",
}

# Driver modules for MySQL/MariaDB, which quote identifiers with backticks
# (double quotes are string literals there unless ANSI_QUOTES is enabled).
_MYSQL_DRIVERS = frozenset({"pymysql", "MySQLdb", "mysql.connector"})

# MySQL error numbers for a table, or the database it is looked up in, that
# doesn't exist
_MYSQL_MISSING_TABLE_ERRNOS = frozenset({1049, 1146})

# MySQL's TIMESTAMP only covers 1970-2038, and its DATETIME/TIME default to
# whole seconds. It has no time-zone-aware type.
_MYSQL_SQL_TYPES = {
    **_SQL_TYPES,
    "datetime": "DATETIME(6)",
    "datetimetz": "DATETIME(6)",
    "time": "TIME(6)",
}

# Drivers with a "format" or "pyformat" paramstyle that substitute the
# placeholders themselves instead of %-formatting the statement, so a literal
# % in it must not be doubled.
_LITERAL_PERCENT_DRIVERS = frozenset({"mysql.connector", "pg8000"})


def find_driver_module(con: object) -> ModuleType | None:
    """
    Return the DB-API module that ``con`` comes from, or None.

    PEP 249 puts ``paramstyle`` and ``Error`` on the driver module. Look for
    it starting from the modules defining the connection's classes (its own
    class may be a subclass defined elsewhere), then from the module of
    ``con.Error``, PEP 249's optional extension, which wrappers such as
    SQLAlchemy's pool proxy pass through. Each module path is walked from the
    top (e.g. ``psycopg2.extensions`` -> ``psycopg2``) to the first module
    that has both attributes. Only already-imported modules are considered.
    """
    module_names = [cls.__module__ for cls in type(con).__mro__]
    error = getattr(con, "Error", None)
    if isinstance(error, type):
        module_names.append(error.__module__)
    for module_name in module_names:
        parts = module_name.split(".")
        for i in range(1, len(parts) + 1):
            module = sys.modules.get(".".join(parts[:i]))
            if (
                getattr(module, "paramstyle", None) in _PARAMSTYLES
                and getattr(module, "Error", None) is not None
            ):
                return module
    return None


def _is_tz_aware(dtype: DtypeObj) -> bool:
    if isinstance(dtype, DatetimeTZDtype):
        return True
    pa_type = getattr(dtype, "pyarrow_dtype", None)
    return getattr(pa_type, "tz", None) is not None


def quote_identifier(name: object, quote: str) -> str:
    """
    Wrap a table or column name in ``quote``, doubling any ``quote`` inside it.
    """
    uname = get_unicode_name(name)
    if not len(uname):
        raise ValueError("Empty table or column name specified")

    nul_index = uname.find("\x00")
    if nul_index >= 0:
        raise ValueError("SQL identifier cannot contain NUL characters")

    return quote + uname.replace(quote, quote + quote) + quote


class DBAPI2Table(SQLTable):
    """
    SQLTable for DB-API 2.0 connections, without SQLAlchemy.

    Instead of a SQLAlchemy ``Table``, ``self.table`` holds the
    ``CREATE TABLE`` / ``CREATE INDEX`` statements as strings. Identifier
    quoting, placeholders and type names come from the owning
    ``DBAPI2Database``.
    """

    def sql_schema(self) -> str:
        return str(";\n".join(self.table))

    def _execute_create(self) -> None:
        with self.pd_sql.run_transaction() as cur:
            for stmt in self.table:
                cur.execute(stmt)

    def insert_statement(self, *, num_rows: int) -> str:
        names = list(map(str, self.frame.columns))
        escape = self.pd_sql._escape_identifier

        if self.index is not None:
            for idx in self.index[::-1]:
                names.insert(0, idx)

        bracketed_names = [escape(column) for column in names]
        col_names = ",".join(bracketed_names)
        table_name = self.pd_sql._table_identifier(self.name, self.schema)

        num_cols = len(names)
        placeholders = self.pd_sql._placeholders(num_rows * num_cols)
        wildcards = ",".join(
            "(" + ",".join(placeholders[row * num_cols : (row + 1) * num_cols]) + ")"
            for row in range(num_rows)
        )
        target = f"INSERT INTO {table_name} ({col_names})"
        if self.pd_sql._doubles_percent():
            target = target.replace("%", "%%")
        return f"{target} VALUES {wildcards}"

    def _execute_insert(self, conn, keys, data_iter) -> int:
        driver_error = self.pd_sql._wrapped_errors()

        bind = self.pd_sql._binder()
        data_list = (
            list(data_iter) if bind is None else [bind(row) for row in data_iter]
        )
        try:
            conn.executemany(self.insert_statement(num_rows=1), data_list)
        except driver_error as exc:
            raise DatabaseError("Execution failed") from exc
        return conn.rowcount

    def _execute_insert_multi(self, conn, keys, data_iter) -> int:
        data_list = list(data_iter)
        flattened_data = [x for row in data_list for x in row]
        bind = self.pd_sql._binder()
        conn.execute(
            self.insert_statement(num_rows=len(data_list)),
            flattened_data if bind is None else bind(flattened_data),
        )
        return conn.rowcount

    def _create_table_setup(self):
        """
        Return a list of SQL statements that creates a table reflecting the
        structure of a DataFrame.  The first entry will be a CREATE TABLE
        statement while the rest will be CREATE INDEX statements.
        """
        column_names_and_types = self._get_column_names_and_types(self._sql_type_name)
        escape = self.pd_sql._escape_identifier

        create_tbl_stmts = [
            escape(cname) + " " + ctype for cname, ctype, _ in column_names_and_types
        ]

        if self.keys is not None and len(self.keys):
            if not is_list_like(self.keys):
                keys = [self.keys]
            else:
                keys = self.keys
            cnames_br = ", ".join([escape(c) for c in keys])
            create_tbl_stmts.append(
                f"CONSTRAINT {self.name}_pk PRIMARY KEY ({cnames_br})"
            )
        table_name = self.pd_sql._table_identifier(self.name, self.schema)
        create_stmts = [
            "CREATE TABLE "
            + table_name
            + " (\n"
            + ",\n  ".join(create_tbl_stmts)
            + "\n)"
        ]

        ix_cols = [cname for cname, _, is_index in column_names_and_types if is_index]
        if ix_cols:
            cnames = "_".join(ix_cols)
            cnames_br = ",".join([escape(c) for c in ix_cols])
            create_stmts.append(
                "CREATE INDEX "
                + escape("ix_" + self.name + "_" + cnames)
                + "ON "
                + table_name
                + " ("
                + cnames_br
                + ")"
            )

        return create_stmts

    def _sql_type_name(self, col):
        dtype: DtypeArg = self.dtype or {}
        if is_dict_like(dtype):
            dtype = cast("Mapping[Hashable, Any]", dtype)
            if col.name in dtype:
                return dtype[col.name]

        # Infer type of column, while ignoring missing values.
        # Needed for inserting typed data containing NULLs, GH 8778.
        col_type = lib.infer_dtype(col, skipna=True)

        if col_type == "timedelta64":
            warnings.warn(
                "the 'timedelta' type is not supported, and will be "
                "written as integer values (ns frequency) to the database.",
                UserWarning,
                stacklevel=find_stack_level(),
            )
            col_type = "integer"

        elif col_type == "datetime64":
            col_type = "datetime"

        elif col_type == "empty":
            col_type = "string"

        elif col_type == "complex":
            raise ValueError("Complex datatypes not supported")

        sql_types = self.pd_sql._sql_type_map()
        if (
            col_type == "datetime"
            and "datetimetz" in sql_types
            and _is_tz_aware(col.dtype)
        ):
            col_type = "datetimetz"
        if col_type not in sql_types:
            col_type = "string"

        return sql_types[col_type]


class DBAPI2Database(PandasSQL):
    """
    PandasSQL for DB-API 2.0 connections, without SQLAlchemy.

    The driver- and dialect-specific pieces (the driver's exception class,
    identifier quoting, the parameter style, SQL type names and
    ``has_table``) have generic implementations here that subclasses such as
    ``SQLiteDatabase`` override.

    Parameters
    ----------
    con : DB-API 2.0 connection object
    """

    _table_class: type[DBAPI2Table] = DBAPI2Table

    # Whether driver errors raised while executing are re-raised as pandas'
    # DatabaseError. Off for generic connections, where callers have always
    # received the driver's own exceptions.
    _wrap_driver_errors: bool = False

    # Whether to_sql writes to the given ``schema``
    _to_sql_uses_schema: bool = True

    def __init__(self, con, driver: ModuleType) -> None:
        self.con = con
        # The driver's DB-API module, which has ``paramstyle`` and ``Error``
        self.driver = driver
        self._transaction_depth = 0

    def _driver_error(self) -> type[Exception]:
        """
        Return the driver's base exception class (PEP 249 ``Error``).
        """
        # PEP 249 also allows exposing the exception classes on the
        # connection, which works for wrapped connections too.
        for source in (self.con, self.driver):
            error = getattr(source, "Error", None)
            if isinstance(error, type) and issubclass(error, Exception):
                return error
        return Exception

    def _wrapped_errors(self) -> tuple[type[Exception], ...]:
        """
        Return the exception classes to re-raise as DatabaseError.
        """
        return (self._driver_error(),) if self._wrap_driver_errors else ()

    def _escape_identifier(self, name: object) -> str:
        """
        Quote a table or column name for use in generated SQL.
        """
        return quote_identifier(name, "`" if self._is_mysql() else '"')

    def _table_identifier(self, name: object, schema: str | None = None) -> str:
        """
        Quote a table name for use in generated SQL, qualified by ``schema``.
        """
        table_name = self._escape_identifier(name)
        if schema:
            return f"{self._escape_identifier(schema)}.{table_name}"
        return table_name

    def _is_mysql(self) -> bool:
        return self.driver.__name__ in _MYSQL_DRIVERS

    def _paramstyle(self) -> str:
        """
        Return the driver's PEP 249 ``paramstyle``.
        """
        return self.driver.paramstyle

    def _doubles_percent(self) -> bool:
        """
        Whether a literal % in a statement that has parameters is doubled.

        Most "format" and "pyformat" drivers %-format such statements.
        """
        return (
            self._paramstyle() in ("format", "pyformat")
            and self.driver.__name__ not in _LITERAL_PERCENT_DRIVERS
        )

    def _placeholders(self, count: int) -> list[str]:
        """
        Return the parameter placeholders for ``count`` positional values.
        """
        paramstyle = self._paramstyle()
        if paramstyle == "qmark":
            return ["?"] * count
        elif paramstyle == "numeric":
            return [f":{i + 1}" for i in range(count)]
        elif paramstyle == "named":
            return [f":p{i}" for i in range(count)]
        elif paramstyle == "format":
            return ["%s"] * count
        return [f"%(p{i})s" for i in range(count)]

    def _binder(self) -> Callable[[Sequence[Any]], dict[str, Any]] | None:
        """
        Return a function packaging positional values the way
        ``_placeholders`` refers to them, or None if they can be passed as is.
        """
        if self._paramstyle() not in ("named", "pyformat"):
            return None
        keys: list[str] = []

        def bind(values: Sequence[Any]) -> dict[str, Any]:
            # Called once per row by the default insert, so keep the keys
            if len(keys) < len(values):
                keys.extend(f"p{i}" for i in range(len(keys), len(values)))
            return dict(zip(keys, values, strict=False))

        return bind

    def _sql_type_map(self) -> Mapping[str, str]:
        """
        Map ``lib.infer_dtype`` results to SQL column type names.
        """
        return _MYSQL_SQL_TYPES if self._is_mysql() else _SQL_TYPES

    def has_table(self, name: str, schema: str | None = None) -> bool:
        # Ask the database to resolve the name exactly as the CREATE TABLE /
        # INSERT that follow will (information_schema can't do that portably:
        # it also lists tables in other schemas and databases). A failed query
        # aborts the open transaction on some databases (e.g. PostgreSQL), so
        # run it inside a savepoint to keep the caller's uncommitted work. In
        # autocommit mode, or without savepoint support, run it directly.
        # Only an error that says the table is missing means it is; any other
        # (no permission to read it, a transaction that already failed) is
        # the caller's to see.
        driver_error = self._driver_error()
        probe = f"SELECT 1 FROM {self._table_identifier(name, schema)} WHERE 1 = 0"
        savepoint = "pandas_has_table"
        cur = self.con.cursor()
        try:
            try:
                cur.execute(f"SAVEPOINT {savepoint}")
            except driver_error:
                in_savepoint = False
            else:
                in_savepoint = True

            failure = None
            try:
                cur.execute(probe)
                # Unbuffered cursors (e.g. mysql.connector) need the empty
                # result read before the next statement.
                cur.fetchall()
            except driver_error as exc:
                failure = exc

            if in_savepoint:
                # Some databases drop the savepoint already (MySQL in
                # autocommit mode) or can't release one (Oracle); there is
                # nothing to undo in either case.
                with suppress(driver_error):
                    if failure is not None:
                        cur.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
                    cur.execute(f"RELEASE SAVEPOINT {savepoint}")
        finally:
            cur.close()

        if failure is None:
            return True
        if self._is_missing_table_error(failure):
            return False
        raise failure

    def _is_missing_table_error(self, exc: Exception) -> bool:
        """
        Whether ``exc``, raised by selecting from a table, means it is missing.

        Decided by the error code where the driver attaches one. Without a
        code, the table is taken to be missing.
        """
        if self._is_mysql():
            errno = getattr(exc, "errno", None)
            if errno is None and exc.args:
                errno = exc.args[0]
            if isinstance(errno, int):
                return errno in _MYSQL_MISSING_TABLE_ERRNOS

        sqlstate = getattr(exc, "sqlstate", None) or getattr(exc, "pgcode", None)
        if isinstance(sqlstate, str) and len(sqlstate) == 5:
            # Class 42 has every dialect's "undefined table" code, and 3F is
            # an undefined schema; 42501 is insufficient privilege
            return sqlstate[:2] in ("42", "3F") and sqlstate != "42501"
        return True

    @contextmanager
    def run_transaction(self):
        # Only the outermost block commits or rolls back
        outermost = self._transaction_depth == 0
        self._transaction_depth += 1
        cur = self.con.cursor()
        try:
            yield cur
            if outermost:
                self.con.commit()
        except Exception:
            if outermost:
                self.con.rollback()
            raise
        finally:
            self._transaction_depth -= 1
            cur.close()

    def execute(self, sql: str | Select[Any] | TextClause, params=None):
        driver_error = self._wrapped_errors()

        if not isinstance(sql, str):
            raise TypeError("Query must be a string unless using sqlalchemy.")
        args = [] if params is None else [params]
        cur = self.con.cursor()
        try:
            cur.execute(sql, *args)
            return cur
        except driver_error as exc:
            try:
                self.con.rollback()
            except driver_error as inner_exc:  # pragma: no cover
                ex = DatabaseError(
                    f"Execution failed on sql: {sql}\n{exc}\nunable to rollback"
                )
                raise ex from inner_exc

            if re.fullmatch(r"[A-Za-z_]\w*", sql.strip()):
                # GH#54233 a bare identifier is almost certainly an attempt to
                #  read a table by name.
                ex = DatabaseError(
                    f"Execution failed on sql '{sql}': {exc}\n\n"
                    "Reading a table by name is only supported when using a "
                    "SQLAlchemy connection. With a DBAPI connection, pass a SQL "
                    f"query instead, e.g. 'SELECT * FROM {sql.strip()}'."
                )
                raise ex from exc

            ex = DatabaseError(f"Execution failed on sql '{sql}': {exc}")
            raise ex from exc

    @staticmethod
    def _query_iterator(
        cursor,
        chunksize: int,
        columns,
        index_col=None,
        coerce_float: bool = True,
        parse_dates=None,
        dtype: DtypeArg | None = None,
        dtype_backend: DtypeBackend | lib.NoDefault = lib.no_default,
    ) -> Generator[DataFrame]:
        """Return generator through chunked result set"""
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
        sql,
        index_col=None,
        coerce_float: bool = True,
        parse_dates=None,
        params=None,
        chunksize: int | None = None,
        dtype: DtypeArg | None = None,
        dtype_backend: DtypeBackend | lib.NoDefault = lib.no_default,
    ) -> DataFrame | Iterator[DataFrame]:
        cursor = self.execute(sql, params)
        columns = [col_desc[0] for col_desc in cursor.description]

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
        else:
            data = self._fetchall_as_list(cursor)
            cursor.close()

            frame = wrap_result(
                data,
                columns,
                index_col=index_col,
                coerce_float=coerce_float,
                parse_dates=parse_dates,
                dtype=dtype,
                dtype_backend=dtype_backend,
            )
            return frame

    def _fetchall_as_list(self, cur):
        result = cur.fetchall()
        if not isinstance(result, list):
            result = list(result)
        return result

    def to_sql(
        self,
        frame,
        name: str,
        if_exists: Literal["fail", "replace", "append", "delete_rows"] = "fail",
        index: bool = True,
        index_label=None,
        schema=None,
        chunksize: int | None = None,
        dtype: DtypeArg | None = None,
        method: Literal["multi"] | Callable[..., Any] | None = None,
        engine: str = "auto",
        **engine_kwargs,
    ) -> int | None:
        """
        Write records stored in a DataFrame to a SQL database.

        Parameters
        ----------
        frame: DataFrame
        name: string
            Name of SQL table.
        if_exists: {'fail', 'replace', 'append', 'delete_rows'}, default 'fail'
            fail: If table exists, do nothing.
            replace: If table exists, drop it, recreate it, and insert data.
            append: If table exists, insert data. Create if it does not exist.
            delete_rows: If a table exists, delete all records and insert data.
        index : bool, default True
            Write DataFrame index as a column
        index_label : string or sequence, default None
            Column label for index column(s). If None is given (default) and
            `index` is True, then the index names are used.
            A sequence should be given if the DataFrame uses MultiIndex.
        schema : string, default None
            Name of the schema to write to, used to qualify the table name.
            Ignored for sqlite3 connections.
        chunksize : int, default None
            If not None, then rows will be written in batches of this
            size at a time. If None, all rows will be written at once.
        dtype : single type or dict of column name to SQL type, default None
            Optional specifying the datatype for columns. The SQL type should
            be a string. If all columns are of the same type, one single value
            can be used.
        method : {None, 'multi', callable}, default None
            Controls the SQL insertion clause used:

            * None : Uses standard SQL ``INSERT`` clause (one per row).
            * 'multi': Pass multiple values in a single ``INSERT`` clause.
            * callable with signature ``(pd_table, conn, keys, data_iter)``.

            Details and a sample callable implementation can be found in the
            section :ref:`insert method <io.sql.method>`.
        """
        if dtype:
            if not is_dict_like(dtype):
                # error: Value expression in dictionary comprehension has incompatible
                # type "Union[ExtensionDtype, str, dtype[Any], Type[object],
                # Dict[Hashable, Union[ExtensionDtype, Union[str, dtype[Any]],
                # Type[str], Type[float], Type[int], Type[complex], Type[bool],
                # Type[object]]]]"; expected type "Union[ExtensionDtype, str,
                # dtype[Any], Type[object]]"
                dtype = dict.fromkeys(frame, dtype)  # type: ignore[arg-type]
            else:
                dtype = cast("Mapping[Hashable, Any]", dtype)

            for col, my_type in dtype.items():
                if not isinstance(my_type, str):
                    raise ValueError(
                        f"Invalid type '{my_type}' for dtype of column '{col}': "
                        "expected a string. When using a DB-API connection "
                        "(e.g. sqlite3), dtype values must be SQL type "
                        "strings (e.g. 'TEXT', 'FLOAT')."
                    )

        table = self._table_class(
            name,
            self,
            frame=frame,
            index=index,
            if_exists=if_exists,
            index_label=index_label,
            schema=schema if self._to_sql_uses_schema else None,
            dtype=dtype,
        )
        # One transaction, so that a failed insert also undoes a table that was
        # dropped, created or emptied for it, where the database can
        with self.run_transaction():
            table.create()
            return table.insert(chunksize, method)

    def get_table(self, table_name: str, schema: str | None = None) -> None:
        return None  # not supported in fallback mode

    def drop_table(self, name: str, schema: str | None = None) -> None:
        drop_sql = f"DROP TABLE {self._table_identifier(name, schema)}"
        self.execute(drop_sql).close()

    def delete_rows(self, name: str, schema: str | None = None) -> None:
        delete_sql = f"DELETE FROM {self._table_identifier(name, schema)}"
        if self.has_table(name, schema):
            self.execute(delete_sql).close()

    def _create_sql_schema(
        self,
        frame,
        table_name: str,
        keys=None,
        dtype: DtypeArg | None = None,
        schema: str | None = None,
    ) -> str:
        table = self._table_class(
            table_name,
            self,
            frame=frame,
            index=False,
            keys=keys,
            dtype=dtype,
            schema=schema,
        )
        return str(table.sql_schema())
