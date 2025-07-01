import asyncio
import dataclasses
import enum
import json
import logging
import time
import uuid
from typing import Any, Dict, Optional, Tuple

try:
    import jsonschema
except ImportError:
    jsonschema = None


class LogLevel(enum.IntEnum):
    """Enumeration for different logging verbosity levels."""

    NONE = 0
    MINIMAL = 1
    SUMMARY = 2
    DETAILED = 3
    DEBUG = 4


logger = logging.getLogger(__name__)
# --- Custom Logging Filter ---
# This filter ensures that all log records have an 'agent_name' and a normalized 'name' (logger name)
# attribute before being processed by the formatter. This is crucial for consistent log output,
# especially when logs originate from different parts of the application or third-party libraries.
class AgentLogFilter(logging.Filter):
    """
    A logging filter that ensures 'agent_name' and a normalized 'name'
    are present on log records for consistent formatting.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        # Ensure 'agent_name' is always set on the record and is a string.
        # If 'agent_name' is already an attribute (e.g., passed via `extra={'agent_name': ...}`),
        # use it; otherwise, default to "System". This is important for logs from external
        # libraries like httpx that won't have `agent_name` by default.
        current_agent_name = getattr(record, "agent_name", None)
        if current_agent_name is None:
            record.agent_name = "System"
        else:
            record.agent_name = str(current_agent_name)

        # Ensure 'name' (logger name) is always set, is a string,
        # and the 'root' logger's name is replaced by 'DefaultLogger' for cleaner logs.
        current_logger_name = getattr(record, "name", None)
        if not current_logger_name or current_logger_name == "root":
            record.name = "DefaultLogger"
        else:
            record.name = str(current_logger_name)

        return True


# --- Logging Setup Utility ---
def init_agent_logging(
    level: int = logging.INFO, clear_existing_handlers: bool = True
) -> None:
    """
    Sets up a standardized console logging configuration for agent-based systems.

    Includes a custom filter to ensure 'agent_name' is available in log records.

    Args:
        level: The desired logging level for the root logger (e.g., logging.INFO, logging.DEBUG).
        clear_existing_handlers: If True, removes any handlers already attached to the
                                 root logger. This is useful in interactive environments
                                 (like Jupyter notebooks) to prevent duplicate log outputs
                                 when re-running setup code.
    """
    root_logger = logging.getLogger()

    if clear_existing_handlers:
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
            handler.close()  # Important to close handlers before removing

    # Create a StreamHandler to output log messages to the console.
    stream_handler = logging.StreamHandler()

    # Define the log message format.
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - [%(name)s] [%(agent_name)s] %(message)s"
    )
    stream_handler.setFormatter(formatter)

    # Add the custom AgentLogFilter to the stream_handler.
    stream_handler.addFilter(AgentLogFilter())

    # Add the configured stream_handler to the root logger.
    root_logger.addHandler(stream_handler)

    # Set the logging level for the root logger.
    root_logger.setLevel(level)

    logging.getLogger(__name__).info(
        f"Agent logging setup complete. Root logger level set to {logging.getLevelName(level)}."
    )


@dataclasses.dataclass
class ProgressUpdate:
    """Dataclass representing a single progress update during task execution."""

    timestamp: float = dataclasses.field(default_factory=time.time)
    level: LogLevel = LogLevel.SUMMARY  # Changed default from LogLevel.INFO
    message: str = ""
    task_id: Optional[str] = None
    interaction_id: Optional[str] = None
    agent_name: Optional[str] = None
    data: Optional[Dict[str, Any]] = None


@dataclasses.dataclass
class RequestContext:
    """
    Dataclass holding context information for a specific agent invocation within a task.

    Attributes:
        task_id: Unique identifier for the overall task.
        initial_prompt: The initial prompt that started the task.
        progress_queue: Async queue for sending progress updates.
        log_level: The minimum log level to report.
        max_depth: Maximum allowed depth for agent invocations.
        max_interactions: Maximum allowed number of interactions (invocations) for the task.
        interaction_id: Unique identifier for the current agent interaction/invocation.
        depth: Current depth of invocation in the agent call chain.
        interaction_count: Current count of interactions in the task.
        caller_agent_name: Name of the agent that invoked the current agent.
        callee_agent_name: Name of the agent currently being invoked.
        current_tokens_used: Current number of tokens used in the task.
        max_tokens_soft_limit: Soft limit for tokens, suggesting the agent should wrap up.
        max_tokens_hard_limit: Hard limit for tokens, forcing the agent to stop.
    """

    progress_queue: asyncio.Queue[Optional[ProgressUpdate]]
    log_level: LogLevel = LogLevel.SUMMARY
    max_depth: int = 5
    max_interactions: int = 10
    task_id: str = dataclasses.field(default_factory=lambda: str(uuid.uuid4()))
    initial_prompt: Optional[Any] = None
    interaction_id: Optional[str] = None
    depth: int = 0
    interaction_count: int = 0
    caller_agent_name: Optional[str] = None
    callee_agent_name: Optional[str] = None
    current_tokens_used: int = 0
    max_tokens_soft_limit: Optional[int] = None
    max_tokens_hard_limit: Optional[int] = None


# --- Logging Utility ---


class ProgressLogger:
    """Utility class for logging progress updates."""

    @staticmethod
    async def log(
        request_context: Optional[RequestContext],
        level: LogLevel,
        message: str,
        agent_name: Optional[str] = None,
        interaction_id: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Logs a progress message either to the async queue or standard logging.

        Args:
            request_context: The context of the current request, containing the queue and log level.
                             Can be None for logging outside a specific task context.
            level: The severity level of the log message.
            message: The log message content.
            agent_name: The name of the agent generating the log.
            interaction_id: The ID of the specific interaction this log relates to.
            data: Optional dictionary containing additional structured data.
        """
        if (
            request_context
            and request_context.progress_queue
            and level <= request_context.log_level
        ):
            update = ProgressUpdate(
                timestamp=time.time(),
                level=level,
                message=message,
                task_id=request_context.task_id,
                interaction_id=interaction_id or request_context.interaction_id,
                agent_name=agent_name,
                data=data,
            )
            try:
                await request_context.progress_queue.put(update)
                return  # <<<--- Exit after successfully putting on queue to prevent double logging
            except Exception as e:
                # Fallback to standard logging if queue put fails
                logging.error(
                    f"Failed to put log message on queue: {e}. Falling back to standard log."
                )

        # Standard logging fallback (if no queue, level too low, or queue put failed)
        if level > LogLevel.NONE:  # Check level *before* formatting/logging
            log_level_map = {
                LogLevel.MINIMAL: logging.INFO,
                LogLevel.SUMMARY: logging.INFO,
                LogLevel.DETAILED: logging.DEBUG,
                LogLevel.DEBUG: logging.DEBUG,
            }
            std_log_level = log_level_map.get(level, logging.INFO)
            log_msg = f"[Task:{request_context.task_id if request_context else 'N/A'}]"
            current_interaction_id = interaction_id or (
                request_context.interaction_id if request_context else None
            )
            if current_interaction_id:
                log_msg += f" [Interaction:{current_interaction_id}]"
            if agent_name:
                log_msg += f" [{agent_name}]"
            log_msg += f" {message}"
            if data:
                try:
                    log_msg += f" Data: {json.dumps(data)}"
                except TypeError:
                    log_msg += f" Data: (Unserializable data)"
            logging.log(std_log_level, log_msg)


# --- Schema Utility Functions ---

PYTHON_TYPE_TO_JSON_TYPE = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
    type(None): "null",
}


def compile_schema(schema: Any) -> Optional[Dict[str, Any]]:
    """
    Compiles a user-friendly schema into a JSON schema dictionary.
    """
    if schema is None:
        return None

    if isinstance(schema, list) and all(isinstance(i, str) for i in schema):
        return {
            "type": "object",
            "properties": {key: {"type": "string"} for key in schema},
            "required": schema,
        }

    if isinstance(schema, dict):
        is_dict_of_types = all(isinstance(v, type) for v in schema.values())
        is_json_schema = "properties" in schema and "type" in schema

        if is_dict_of_types and not is_json_schema:
            properties = {}
            for key, value_type in schema.items():
                json_type = PYTHON_TYPE_TO_JSON_TYPE.get(value_type)
                if json_type is None:
                    logger.warning(
                        f"Unsupported type '{value_type}' in schema for key '{key}'. "
                        "Defaulting to 'object'."
                    )
                    properties[key] = {"type": "object"}
                else:
                    properties[key] = {"type": json_type}
            return {
                "type": "object",
                "properties": properties,
                "required": list(schema.keys()),
            }
        return schema

    logger.warning(
        f"Invalid schema format provided: {type(schema)}. No schema will be enforced."
    )
    return None


def prepare_for_validation(data: Any, schema: Dict[str, Any]) -> Any:
    """
    Prepares data before validation, e.g., by wrapping a string.
    """
    if (
        isinstance(data, str)
        and schema.get("type") == "object"
        and len(schema.get("required", [])) == 1
    ):
        required_key = schema["required"][0]
        prop_schema = schema.get("properties", {}).get(required_key, {})
        if prop_schema.get("type") == "string":
            logger.debug(
                f"Wrapping string data into object with key '{required_key}' for validation."
            )
            return {required_key: data}
    return data


def validate_data(
    data: Any, compiled_schema: Optional[Dict[str, Any]]
) -> Tuple[bool, Optional[str]]:
    """
    Validates data against a compiled JSON schema.
    """
    if compiled_schema is None:
        return True, None

    if jsonschema is None:
        logger.warning(
            "jsonschema is not installed. Skipping validation. "
            "Please `pip install jsonschema` to enable validation."
        )
        return True, None

    prepared_data = prepare_for_validation(data, compiled_schema)

    try:
        jsonschema.validate(instance=prepared_data, schema=compiled_schema)
        return True, None
    except jsonschema.exceptions.ValidationError as e:
        error_path = " -> ".join(map(str, e.path))
        if error_path:
            error_msg = f"Validation Error at '{error_path}': {e.message}"
        else:
            error_msg = f"Validation Error: {e.message}"
        logger.debug(f"Schema validation failed: {error_msg}\nData: {data}\nSchema: {compiled_schema}")
        return False, error_msg
    except Exception as e:
        error_msg = f"An unexpected error occurred during schema validation: {e}"
        logger.error(error_msg, exc_info=True)
        return False, error_msg

# --- End Schema Utility Functions ---
