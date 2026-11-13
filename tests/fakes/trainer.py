from collections.abc import Callable
from typing import Any

from transformers.trainer_callback import TrainerControl, TrainerState


class PushRecorder:
    """Stand-in for the trainer's ``PushInProgress``."""

    def __init__(self, events: list[str]) -> None:
        self._events = events

    def is_done(self) -> bool:
        return False

    def wait_until_done(self) -> None:
        self._events.append("wait_until_done")


class FakeTrainer:
    """Steps its callbacks the way ``Trainer`` does: ``on_step_end`` first, then the save that starts a push."""

    def __init__(
        self,
        callbacks: list[Any],
        *,
        max_steps: int,
        save_steps: int = 100,
        before_step: Callable[[int], None] | None = None,
    ) -> None:
        self.callbacks = list(callbacks)
        self.max_steps = max_steps
        self.save_steps = save_steps
        self.before_step = before_step
        self.state = TrainerState()
        self.control = TrainerControl()
        self.push_in_progress: PushRecorder | None = None
        self.events: list[str] = []
        self.train_calls: list[str | None] = []

    def train(self, resume_from_checkpoint: str | None = None) -> None:
        self.train_calls.append(resume_from_checkpoint)
        while self.state.global_step < self.max_steps and not self.control.should_training_stop:
            self.state.global_step += 1
            if self.before_step is not None:
                self.before_step(self.state.global_step)
            for callback in self.callbacks:
                self.control = callback.on_step_end(None, self.state, self.control) or self.control
            if self.state.global_step % self.save_steps == 0:
                self.events.append(f"save@{self.state.global_step}")
                self.push_in_progress = PushRecorder(self.events)
