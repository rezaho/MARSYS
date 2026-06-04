# AG-UI `Custom` Events — `marsys.*`

**AUTO-GENERATED — DO NOT EDIT BY HAND.** Regenerate via `python packages/framework/scripts/generate_aggui_custom_events_doc.py`. Source: `packages/framework/src/marsys/coordination/aggui/custom_events.py`.

AG-UI's `Custom` event is the documented escape hatch for protocol-internal events that don't fit the standard lifecycle. MARSYS uses it for framework-specific lifecycle signals (branch / parallel-group / convergence / error / resource-limit / user-interaction / memory-compaction) and for the stream-level handshake.

Every event below is validated at emission time against the Pydantic model in `coordination/aggui/custom_events.py`. Validation failure raises — catches schema drift at the source. Consumers that want lenient parsing wrap the iterator in `try/except` at their boundary.

---

## `marsys.aggui.handshake`

First event on every stream — protocol-version handshake. Emitted as a leading ``Custom("marsys.aggui.handshake")`` before ``RunStartedEvent``. AG-UI's ``RunStartedEvent.input`` is a strongly-typed ``RunAgentInput`` (designed to echo the client's request); it can't carry arbitrary protocol metadata. A leading Custom event is the documented escape hatch.

### JSON Schema

```json
{
  "description": "First event on every stream \u2014 protocol-version handshake.\n\nEmitted as a leading ``Custom(\"marsys.aggui.handshake\")`` before\n``RunStartedEvent``. AG-UI's ``RunStartedEvent.input`` is a strongly-typed\n``RunAgentInput`` (designed to echo the client's request); it can't carry\narbitrary protocol metadata. A leading Custom event is the documented\nescape hatch.",
  "properties": {
    "ag_ui_version": {
      "title": "Ag Ui Version",
      "type": "string"
    },
    "marsys_version": {
      "title": "Marsys Version",
      "type": "string"
    },
    "schema_version": {
      "default": 1,
      "title": "Schema Version",
      "type": "integer"
    }
  },
  "required": [
    "marsys_version",
    "ag_ui_version"
  ],
  "title": "AGGUIHandshakeValue",
  "type": "object"
}
```

## `marsys.branch.completed`

### JSON Schema

```json
{
  "properties": {
    "branch_id": {
      "title": "Branch Id",
      "type": "string"
    },
    "last_agent": {
      "title": "Last Agent",
      "type": "string"
    },
    "success": {
      "title": "Success",
      "type": "boolean"
    },
    "total_steps": {
      "title": "Total Steps",
      "type": "integer"
    }
  },
  "required": [
    "branch_id",
    "last_agent",
    "success",
    "total_steps"
  ],
  "title": "BranchCompletedValue",
  "type": "object"
}
```

## `marsys.branch.created`

### JSON Schema

```json
{
  "properties": {
    "branch_id": {
      "title": "Branch Id",
      "type": "string"
    },
    "branch_name": {
      "title": "Branch Name",
      "type": "string"
    },
    "parent_branch_id": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "title": "Parent Branch Id"
    },
    "source_agent": {
      "title": "Source Agent",
      "type": "string"
    },
    "target_agents": {
      "items": {
        "type": "string"
      },
      "title": "Target Agents",
      "type": "array"
    },
    "trigger_type": {
      "title": "Trigger Type",
      "type": "string"
    }
  },
  "required": [
    "branch_id",
    "branch_name",
    "source_agent",
    "target_agents",
    "trigger_type"
  ],
  "title": "BranchCreatedValue",
  "type": "object"
}
```

## `marsys.convergence`

### JSON Schema

```json
{
  "properties": {
    "child_branch_ids": {
      "items": {
        "type": "string"
      },
      "title": "Child Branch Ids",
      "type": "array"
    },
    "convergence_point": {
      "title": "Convergence Point",
      "type": "string"
    },
    "group_id": {
      "title": "Group Id",
      "type": "string"
    },
    "parent_branch_id": {
      "title": "Parent Branch Id",
      "type": "string"
    },
    "successful_count": {
      "title": "Successful Count",
      "type": "integer"
    },
    "total_count": {
      "title": "Total Count",
      "type": "integer"
    }
  },
  "required": [
    "parent_branch_id",
    "child_branch_ids",
    "convergence_point",
    "group_id",
    "successful_count",
    "total_count"
  ],
  "title": "ConvergenceValue",
  "type": "object"
}
```

## `marsys.error`

Non-terminal error (run continues).

### JSON Schema

```json
{
  "description": "Non-terminal error (run continues).",
  "properties": {
    "agent": {
      "title": "Agent",
      "type": "string"
    },
    "error_class": {
      "title": "Error Class",
      "type": "string"
    },
    "message": {
      "title": "Message",
      "type": "string"
    },
    "recoverable": {
      "default": true,
      "title": "Recoverable",
      "type": "boolean"
    },
    "retry_count": {
      "default": 0,
      "title": "Retry Count",
      "type": "integer"
    }
  },
  "required": [
    "agent",
    "error_class",
    "message"
  ],
  "title": "ErrorValue",
  "type": "object"
}
```

## `marsys.generation.metadata`

Cost/latency metadata that doesn't fit AG-UI's lifecycle events. ``kind`` distinguishes an ordinary generation from a memory-compaction LLM call (both ride this one Custom event as sibling kinds).

### JSON Schema

```json
{
  "description": "Cost/latency metadata that doesn't fit AG-UI's lifecycle events.\n\n``kind`` distinguishes an ordinary generation from a memory-compaction\nLLM call (both ride this one Custom event as sibling kinds).",
  "properties": {
    "completion_tokens": {
      "anyOf": [
        {
          "type": "integer"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "title": "Completion Tokens"
    },
    "finish_reason": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "title": "Finish Reason"
    },
    "kind": {
      "default": "generation",
      "title": "Kind",
      "type": "string"
    },
    "model": {
      "title": "Model",
      "type": "string"
    },
    "prompt_tokens": {
      "anyOf": [
        {
          "type": "integer"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "title": "Prompt Tokens"
    },
    "provider": {
      "title": "Provider",
      "type": "string"
    },
    "reasoning_tokens": {
      "anyOf": [
        {
          "type": "integer"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "title": "Reasoning Tokens"
    }
  },
  "required": [
    "model",
    "provider"
  ],
  "title": "GenerationMetadataValue",
  "type": "object"
}
```

## `marsys.memory.compaction`

### JSON Schema

```json
{
  "properties": {
    "agent_name": {
      "title": "Agent Name",
      "type": "string"
    },
    "duration": {
      "anyOf": [
        {
          "type": "number"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "title": "Duration"
    },
    "post_tokens": {
      "title": "Post Tokens",
      "type": "integer"
    },
    "pre_tokens": {
      "title": "Pre Tokens",
      "type": "integer"
    },
    "status": {
      "title": "Status",
      "type": "string"
    }
  },
  "required": [
    "agent_name",
    "status",
    "pre_tokens",
    "post_tokens"
  ],
  "title": "MemoryCompactionValue",
  "type": "object"
}
```

## `marsys.parallel.group`

### JSON Schema

```json
{
  "properties": {
    "agent_names": {
      "items": {
        "type": "string"
      },
      "title": "Agent Names",
      "type": "array"
    },
    "completed_count": {
      "title": "Completed Count",
      "type": "integer"
    },
    "group_id": {
      "title": "Group Id",
      "type": "string"
    },
    "status": {
      "enum": [
        "started",
        "executing",
        "converging",
        "completed"
      ],
      "title": "Status",
      "type": "string"
    },
    "total_count": {
      "title": "Total Count",
      "type": "integer"
    }
  },
  "required": [
    "group_id",
    "agent_names",
    "status",
    "completed_count",
    "total_count"
  ],
  "title": "ParallelGroupValue",
  "type": "object"
}
```

## `marsys.resource.limit`

System-level constraint signal (non-terminal).

### JSON Schema

```json
{
  "description": "System-level constraint signal (non-terminal).",
  "properties": {
    "current_value": {
      "anyOf": [
        {},
        {
          "type": "null"
        }
      ],
      "default": null,
      "title": "Current Value"
    },
    "limit_value": {
      "anyOf": [
        {},
        {
          "type": "null"
        }
      ],
      "default": null,
      "title": "Limit Value"
    },
    "pool_name": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "title": "Pool Name"
    },
    "resource_type": {
      "title": "Resource Type",
      "type": "string"
    },
    "suggestion": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "title": "Suggestion"
    }
  },
  "required": [
    "resource_type"
  ],
  "title": "ResourceLimitValue",
  "type": "object"
}
```

## `marsys.stream.lagged`

Emitted on the next successful enqueue after queue overflow. ``count`` is the cumulative number of dropped events since the last lagged notification (drop-newest policy preserves prefix coherence of ``TextMessageStart``/``Content``/``End`` triples).

### JSON Schema

```json
{
  "description": "Emitted on the next successful enqueue after queue overflow.\n\n``count`` is the cumulative number of dropped events since the last\nlagged notification (drop-newest policy preserves prefix coherence of\n``TextMessageStart``/``Content``/``End`` triples).",
  "properties": {
    "count": {
      "title": "Count",
      "type": "integer"
    }
  },
  "required": [
    "count"
  ],
  "title": "StreamLaggedValue",
  "type": "object"
}
```

## `marsys.user_interaction.pending`

### JSON Schema

```json
{
  "properties": {
    "agent_name": {
      "title": "Agent Name",
      "type": "string"
    },
    "options": {
      "anyOf": [
        {
          "items": {
            "type": "string"
          },
          "type": "array"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "title": "Options"
    },
    "prompt_summary": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "title": "Prompt Summary"
    }
  },
  "required": [
    "agent_name"
  ],
  "title": "UserInteractionPendingValue",
  "type": "object"
}
```

## `marsys.user_interaction.resolved`

### JSON Schema

```json
{
  "properties": {
    "agent_name": {
      "title": "Agent Name",
      "type": "string"
    }
  },
  "required": [
    "agent_name"
  ],
  "title": "UserInteractionResolvedValue",
  "type": "object"
}
```

## `marsys.user_interaction.timeout`

### JSON Schema

```json
{
  "properties": {
    "agent_name": {
      "title": "Agent Name",
      "type": "string"
    }
  },
  "required": [
    "agent_name"
  ],
  "title": "UserInteractionTimeoutValue",
  "type": "object"
}
```

