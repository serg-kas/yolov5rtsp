import asyncio
import aiohttp
import threading
import log

class QueueFull(asyncio.QueueFull):
    def __str__(self):
        return "The Telegram queue for sending is full"

class TlgSender(threading.Thread):
    def __init__(self, log_level, bot_token, chat_id, worker_count=2, queue_size=10):
        super(TlgSender, self).__init__()

        self.logger = log.get_logger(name="TlgSender ", level=log_level)

        self.queue = asyncio.Queue(maxsize=queue_size)
        self.workers = []
        self.worker_count = worker_count

        self.bot_token = bot_token
        bot_url_base = "https://api.telegram.org/bot"
        self.bot_url_text = f"{bot_url_base}{self.bot_token}/sendMessage"
        self.bot_url_photo = f"{bot_url_base}{self.bot_token}/sendPhoto"
        self.chat_id = chat_id

    def run(self):
        asyncio.run(self.async_run())

    def stop(self):
        for worker in self.workers:
            self.debug(f"Stopping worker")
            worker.cancel()
        self.logger.info(f"All workers have been stopped")

    async def async_run(self):
        for i in range(self.worker_count):
            self.logger.debug(f"Starting worker {i}")
            self.workers.append(asyncio.create_task(self.worker(f'worker-{i}', self.queue)))
        self.logger.info(f"All workers have been started")
        await asyncio.gather(*self.workers, return_exceptions=True)

    def send(self, text, image):
        self.logger.debug(f"Putting message '{text}' to the queue")
        try:
            self.queue.put_nowait((text, image))
        except asyncio.QueueFull:
            raise QueueFull()

    async def worker(self, name, queue):
        self.logger.info(f"Worker {name} have been started")
        async with aiohttp.ClientSession() as session:
            while True:
                self.logger.info(f"Worker {name} is waiting for a task")
                text, image = await queue.get()

                if image:
                    bot_url = self.bot_url_photo
                    data = {"chat_id": self.chat_id, "caption": text, "photo": image}
                else:
                    bot_url = self.bot_url_text
                    data = {"chat_id": self.chat_id, "text": text}

                async with session.post(bot_url, data=data) as response:
                    self.logger.debug(f"Response to a sent message: {response}")
                    if response.code == 200:
                        self.logger.info(f"Worker {name} sent a message '{text}'.")

                queue.task_done()
        self.logger.info(f"Worker {name} have been stopped")
