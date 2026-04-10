import os
import uuid

import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver


class ChatManager:
    def __init__(self, path="C:/chat_history/agent_history.db") -> None:
        """
        Initializes the ChatManager with a database connection.

        Parameters
        ----------
        path : str, optional
            Path to the SQLite database file
            (default is 'C:/chat_history/agent_history.db')
        """
        self.db_path = path
        self.db_dir = os.path.dirname(self.db_path)
        if self.db_dir and not os.path.exists(self.db_dir):
            os.makedirs(self.db_dir, exist_ok=True)

    async def initialize(self) -> None:
        """
        Initializes the database connection and creates required tables.

        Creates an asynchronous SQLite connection and sets up the
        chat_metadata table if it doesn't exist.
        """
        self.__connection: aiosqlite.Connection = await aiosqlite.connect(self.db_path)
        self.checkpointer = AsyncSqliteSaver(self.__connection)

        await self.__connection.execute("""
            CREATE TABLE IF NOT EXISTS chat_metadata (
                thread_id TEXT PRIMARY KEY,
                title TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await self.__connection.commit()

    async def create_chat(self, query: str = "Новый чат") -> str:
        """
        Creates a new chat thread in the database.

        Parameters
        ----------
        query : str, optional
            The initial query or title for the chat (default is "Новый чат")

        Returns
        -------
        str
            The unique thread ID for the created chat
        """
        thread_id = str(uuid.uuid4())

        await self.__connection.execute(
            """
                INSERT INTO chat_metadata(thread_id, title) VALUES(?, ?)
            """,
            (thread_id, query[:50]),
        )
        await self.__connection.commit()
        return thread_id

    async def delete_chat(self, thread_id: str) -> None:
        """
        Deletes a chat thread from the database and removes its history.

        Parameters
        ----------
        thread_id : str
            The unique identifier of the chat thread to be deleted
        """

        await self.__connection.execute(
            """
                DELETE FROM chat_metadata
                WHERE thread_id = ?
            """,
            (thread_id,),
        )
        await self.__connection.commit()

        if self.checkpointer:
            await self.checkpointer.adelete_thread(thread_id=thread_id)

    async def rename_chat(self, thread_id: str, new_title: str) -> None:
        """
        Renames an existing chat thread.

        Parameters
        ----------
        thread_id : str
            The unique identifier of the chat thread to be renamed
        new_title : str
            The new title for the chat thread
        """
        await self.__connection.execute(
            """
                UPDATE chat_metadata SET title = ?, updated_at = CURRENT_TIMESTAMP
                WHERE thread_id = ?
            """,
            (new_title, thread_id),
        )
        await self.__connection.commit()

    async def list_chats(self) -> list:
        """
        Retrieves a list of all chat threads from the database.

        Returns
        -------
        list
            A list of dictionaries containing chat metadata
            (thread_id, title, updated_at),
            ordered by the most recently updated
        """
        self.__connection.row_factory = aiosqlite.Row
        cursor = await self.__connection.execute("""
            SELECT * FROM chat_metadata
            ORDER BY updated_at DESC
        """)
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def close(self):
        """
        Closes the database connection.

        This method should be called when the application shuts down
        to properly close the database connection.
        """
        if self.__connection:
            await self.__connection.close()
