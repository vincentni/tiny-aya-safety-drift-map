from collections.abc import Mapping


class FakeGenerator:
    """``list[str] -> list[str]`` backend: echoes, or gives a prompt its scripted text or raises its scripted error."""

    def __init__(self, scripted: Mapping[str, str | BaseException] | None = None) -> None:
        self.scripted = dict(scripted or {})
        self.calls: list[list[str]] = []

    def __call__(self, prompts: list[str]) -> list[str]:
        self.calls.append(list(prompts))
        replies = []
        for prompt in prompts:
            reply = self.scripted.get(prompt, prompt)
            if isinstance(reply, BaseException):
                raise reply
            replies.append(reply)
        return replies
