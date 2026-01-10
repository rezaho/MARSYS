"""Core types for task planning - Plan and PlanItem dataclasses."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional
import time
import uuid


@dataclass
class PlanItem:
    """A single item in a task plan."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    title: str = ""
    content: str = ""
    active_form: str = ""  # Present continuous form (e.g., "Running tests")
    status: Literal["pending", "in_progress", "completed", "blocked"] = "pending"
    priority: Literal["high", "medium", "low"] = "medium"
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    blocked_reason: Optional[str] = None
    parent_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def start(self) -> None:
        """Mark item as in progress."""
        self.status = "in_progress"
        self.started_at = time.time()

    def complete(self) -> None:
        """Mark item as completed."""
        self.status = "completed"
        self.completed_at = time.time()

    def block(self, reason: str) -> None:
        """Mark item as blocked with reason."""
        self.status = "blocked"
        self.blocked_reason = reason

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for persistence."""
        return {
            "id": self.id,
            "title": self.title,
            "content": self.content,
            "active_form": self.active_form,
            "status": self.status,
            "priority": self.priority,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "blocked_reason": self.blocked_reason,
            "parent_id": self.parent_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PlanItem':
        """Deserialize from persistence."""
        return cls(**data)


@dataclass
class Plan:
    """A complete task plan containing multiple items."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    items: List[PlanItem] = field(default_factory=list)
    goal: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    version: int = 1
    _item_index: Dict[str, PlanItem] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        """Rebuild the item lookup index."""
        self._item_index = {item.id: item for item in self.items}

    def get_item(self, item_id: str) -> Optional[PlanItem]:
        """Get item by ID."""
        return self._item_index.get(item_id)

    def add_item(self, item: PlanItem, after_id: Optional[str] = None) -> None:
        """Add item to plan, optionally after a specific item."""
        if after_id:
            idx = next(
                (i for i, it in enumerate(self.items) if it.id == after_id),
                None
            )
            if idx is not None:
                self.items.insert(idx + 1, item)
            else:
                self.items.append(item)
        else:
            self.items.append(item)
        self._item_index[item.id] = item
        self.version += 1

    def remove_item(self, item_id: str) -> bool:
        """Remove item by ID. Returns True if removed."""
        item = self._item_index.pop(item_id, None)
        if item:
            self.items.remove(item)
            self.version += 1
            return True
        return False

    @property
    def current_item(self) -> Optional[PlanItem]:
        """Get the currently in-progress item."""
        for item in self.items:
            if item.status == "in_progress":
                return item
        return None

    @property
    def pending_items(self) -> List[PlanItem]:
        """Get all pending items."""
        return [i for i in self.items if i.status == "pending"]

    @property
    def completed_items(self) -> List[PlanItem]:
        """Get all completed items."""
        return [i for i in self.items if i.status == "completed"]

    @property
    def is_empty(self) -> bool:
        """Check if plan has no items."""
        return len(self.items) == 0

    @property
    def progress_ratio(self) -> float:
        """Get completion ratio (0.0 to 1.0)."""
        if not self.items:
            return 0.0
        return len(self.completed_items) / len(self.items)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for persistence."""
        return {
            "id": self.id,
            "items": [item.to_dict() for item in self.items],
            "goal": self.goal,
            "created_at": self.created_at,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Plan':
        """Deserialize from persistence."""
        items = [PlanItem.from_dict(item) for item in data.get("items", [])]
        return cls(
            id=data["id"],
            items=items,
            goal=data.get("goal"),
            created_at=data.get("created_at", time.time()),
            version=data.get("version", 1),
        )
