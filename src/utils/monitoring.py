import asyncio
import json
import logging
from datetime import datetime

monitor_logger = logging.getLogger("ProgressMonitor")

async def default_progress_monitor(q: asyncio.Queue, logger: logging.Logger = None):
    """
    Default progress monitor that logs updates to the console.
    Uses the provided logger or the module's default logger.
    """
    log_instance = logger or monitor_logger
    while True:
        update = await q.get()
        if update is None:  # Sentinel for stopping the monitor
            q.task_done()
            break

        log_message_parts = [
            f"{datetime.fromtimestamp(update.timestamp).strftime('%Y-%m-%d %H:%M:%S')}",
            f"LVL {update.level.value}",
            f"[{update.agent_name or 'System'}]",
            update.message,
        ]
        if update.data:
            try:
                if isinstance(update.data, (dict, list)):
                    data_str = json.dumps(update.data, indent=2)
                else:
                    data_str = str(update.data)
                log_message_parts.append(f"Data: {data_str}")
            except TypeError:
                log_message_parts.append(
                    f"Data: (Unserializable data: {type(update.data)})"
                )
        
        log_instance.info(" - ".join(log_message_parts))
        q.task_done()
