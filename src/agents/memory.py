from typing import Dict, List

from pydantic import BaseModel


class SingleMsg(BaseModel):
    role: str
    content: str


class MessageMemory(BaseModel):
    messages: List[SingleMsg] = []

    def append(self, role: str, content: str) -> None:
        new_message = SingleMsg(role=role, content=content)
        self.messages.append(new_message)

    def get_all(self) -> List[Dict[str, str]]:
        return [{"role": m.role, "content": m.content} for m in self.messages]

    def get_last(self) -> Dict[str, str]:
        return self.messages[-1]

    def reset(self):
        self.messages = []

    def delete_last(self):
        _ = self.messages.pop()

    def __getitem__(self, index: int) -> SingleMsg:
        return self.messages[index]

    def __len__(self) -> int:
        return len(self.messages)
