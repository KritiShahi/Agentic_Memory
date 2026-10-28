import asyncio

class BoundedAsyncChannel:
    def __init__(self, capacity=1):
        self.capacity = capacity
        self.items = []
        self.cond = asyncio.Condition()

    async def put(self, item):
        async with self.cond:
            while len(self.items) >= self.capacity:
                # Bug: Waits indefinitely; if channel closes or error occurs, never notified
                await self.cond.wait()
            self.items.append(item)
            self.cond.notify()

    async def get(self):
        async with self.cond:
            while not self.items:
                await self.cond.wait()
            val = self.items.pop(0)
            self.cond.notify()
            return val
