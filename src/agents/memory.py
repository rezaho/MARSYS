from typing import Dict, List

from pydantic import BaseModel


class SingleMsg(BaseModel):
    role: str
    content: str | List[Dict] | Dict
    tool_call: Dict[str, str | Dict] = {}


class MessageMemory(BaseModel):
    messages: List[SingleMsg] = []

    def append(
        self, role: str, content: str, tool_call: Dict[str, str] = dict()
    ) -> None:
        new_message = SingleMsg(role=role, content=content, tool_call=tool_call)
        self.messages.append(new_message)

    def get_all(self) -> List[Dict[str, str]]:
        messages = []
        for m in self.messages:
            msg = dict()
            msg["role"] = m.role
            if m.content is not None:
                msg["content"] = m.content
            if m.tool_call:
                msg["tool_call"] = m.tool_call
            messages.append(msg)
        return messages

    def get_last(self) -> Dict[str, str]:
        return self.messages[-1]

    def reset(self):
        self.messages = []

    def delete_last(self):
        _ = self.messages.pop()

    # New method to remove html_content from any tool messages
    def clean_tool_messages(self) -> None:
        for msg in self.messages:
            if msg.role == "tool" and isinstance(msg.content, dict):
                _ = msg.content.pop("html_content", None)

    def __getitem__(self, index: int) -> SingleMsg:
        return self.messages[index]

    def __len__(self) -> int:
        return len(self.messages)
