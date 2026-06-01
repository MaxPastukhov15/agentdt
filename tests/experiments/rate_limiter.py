import asyncio
import time
from collections import deque

from logger import logger

from collections import deque

class RateLimitedQueue:
    """Очередь с автоматическим контролем rate limit"""
    
    def __init__(self, requests_per_minute=6):  # Снизил до 6
        self.rpm = requests_per_minute
        self.queue = deque()
        self.timestamps = deque()
    
    async def execute(self, coro):
        """Выполнить корутину с соблюдением лимита"""
        now = time.time()
        while self.timestamps and now - self.timestamps[0] > 60:
            self.timestamps.popleft()
        
        if len(self.timestamps) >= self.rpm:
            wait_time = 60 - (now - self.timestamps[0]) + 1
            logger.info(f"⏳ Rate limit reached, waiting {wait_time:.1f}s...")
            await asyncio.sleep(wait_time)
            self.timestamps.clear()
        
        self.timestamps.append(time.time())
        return await coro