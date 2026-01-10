"""
Database connection management.

This module handles Prisma client lifecycle and connection pooling.
"""

from __future__ import annotations

from prisma import Prisma


class PrismaConnection:
    """
    Manages Prisma database connection lifecycle.

    Usage:
        async with PrismaConnection() as prisma:
            # use prisma client
    """

    def __init__(self) -> None:
        self._client: Prisma | None = None

    @property
    def client(self) -> Prisma:
        """Get the Prisma client instance."""
        if self._client is None:
            raise RuntimeError("Database not connected. Call connect() first.")
        return self._client

    async def connect(self) -> Prisma:
        """
        Initialize and connect the Prisma client.

        Returns:
            Connected Prisma client instance.
        """
        self._client = Prisma()
        await self._client.connect()

        await self._client.query_raw("PRAGMA journal_mode = WAL")
        await self._client.query_raw("PRAGMA synchronous = NORMAL")
        await self._client.query_raw("PRAGMA cache_size = -64000")
        await self._client.query_raw("PRAGMA temp_store = MEMORY")

        return self._client

    async def disconnect(self) -> None:
        """Close the database connection."""
        if self._client is not None:
            await self._client.disconnect()
            self._client = None

    async def __aenter__(self) -> Prisma:
        """Async context manager entry."""
        return await self.connect()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Async context manager exit."""
        await self.disconnect()
