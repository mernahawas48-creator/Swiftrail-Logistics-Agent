from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pymysql
import pymysql.cursors
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(ENV_PATH)


class SwiftrailDatabaseError(RuntimeError):
    """Safe internal database exception with no credential-bearing details."""


class DatabaseConfigurationError(SwiftrailDatabaseError):
    pass


class DatabaseUnavailableError(SwiftrailDatabaseError):
    pass


class DatabaseOperationError(SwiftrailDatabaseError):
    pass


def _read_port() -> int:
    raw = os.environ.get("DB_PORT", "3306")
    try:
        port = int(raw)
    except ValueError as error:
        raise DatabaseConfigurationError("DB_PORT must be a valid integer.") from error
    if not 1 <= port <= 65535:
        raise DatabaseConfigurationError("DB_PORT must be between 1 and 65535.")
    return port


def _database_config() -> dict:
    password = os.environ.get("DB_PASSWORD", "")
    if not password or password == "CHANGE_ME_LOCAL_ONLY":
        raise DatabaseConfigurationError(
            "Database credentials are not configured. Copy .env.example to .env "
            "and set a local password."
        )

    return {
        "host": os.environ.get("DB_HOST", "127.0.0.1"),
        "port": _read_port(),
        "user": os.environ.get("DB_USER", "swiftrail_app"),
        "password": password,
        "database": os.environ.get("DB_NAME", "swiftrail_db"),
        "cursorclass": pymysql.cursors.DictCursor,
        "charset": "utf8mb4",
        "autocommit": False,
        "connect_timeout": 8,
        "read_timeout": 15,
        "write_timeout": 15,
    }


def get_connection() -> pymysql.connections.Connection:
    try:
        return pymysql.connect(**_database_config())
    except DatabaseConfigurationError:
        raise
    except pymysql.MySQLError as error:
        logger.exception("Unable to connect to the Swiftrail database.")
        raise DatabaseUnavailableError(
            "The Swiftrail database is currently unavailable."
        ) from error


@contextmanager
def db_cursor(
    *, transactional: bool = False
) -> Iterator[tuple[pymysql.connections.Connection, pymysql.cursors.DictCursor]]:
    """Yield a DictCursor and guarantee close/commit/rollback behavior."""

    connection = get_connection()
    cursor = connection.cursor()
    try:
        yield connection, cursor
        if transactional:
            connection.commit()
    except pymysql.MySQLError as error:
        if transactional:
            try:
                connection.rollback()
            except pymysql.MySQLError:
                logger.exception("Rollback failed after a database operation error.")
        logger.exception("A Swiftrail database operation failed.")
        raise DatabaseOperationError(
            "The database operation failed and no unconfirmed write was retained."
        ) from error
    except Exception:
        if transactional:
            try:
                connection.rollback()
            except pymysql.MySQLError:
                logger.exception("Rollback failed after an application error.")
        raise
    finally:
        cursor.close()
        connection.close()
