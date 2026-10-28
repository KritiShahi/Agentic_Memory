import asyncio

class BoundedAsyncChannel:
    def __init__(self, capacity=1):
        self.capacity = capacity
        self.items = []
        self.is_closed = False
        self.cond = asyncio.Condition()

    async def put(self, item):
        async with self.cond:
            while len(self.items) >= self.capacity:
                if self.is_closed:
                    raise RuntimeError("Channel closed")
                await self.cond.wait()
            if self.is_closed:
                raise RuntimeError("Channel closed")
            self.items.append(item)
            self.cond.notify_all()

    async def get(self):
        async with self.cond:
            while not self.items:
                if self.is_closed:
                    raise RuntimeError("Channel closed")
                await self.cond.wait()
            val = self.items.pop(0)
            self.cond.notify_all()
            return val

    async def close(self):
        async with self.cond:
            self.is_closed = True
            self.cond.notify_all()
