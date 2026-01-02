import sqlite3
import os
import asyncio
from prisma import Prisma

def init_sqlite_vec():
    """Initialize SQLite database with sqlite-vec extension"""
    db_path = os.path.join(os.path.dirname(__file__), "database.db")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        cursor.execute("SELECT load_extension('vec0')")
    except sqlite3.OperationalError:
        pass
    
    conn.close()
    return db_path

async def init_prisma():
    """Initialize Prisma client and connect to database"""
    prisma = Prisma()
    await prisma.connect()
    return prisma

async def close_prisma(prisma: Prisma):
    """Close Prisma connection"""
    await prisma.disconnect()
