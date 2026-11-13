"""Reader-side inference integration API (capture/source adapter intentionally external).

The stream consumes fully segmented ATQA frames. Applications must supply an
actual SDR or NFC-reader capture source and an enforceable reader controller.
This module cannot access or block NFC transactions without that integration.
"""
from __future__ import annotations
from collections.abc import Iterable, Callable
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
from .predictor import NFCPredictor


@dataclass
class SignalEvent:
    trace: np.ndarray
    signal_id: str


class NFCGuardStream:
    """Thread-agnostic synchronous stream adapter with optional host callback.

    callback receives a prediction record and may implement an external policy.
    The callback's actions are not verified or implied by this library.
    """
    def __init__(self, checkpoint, callback: Callable[[dict], None] | None = None,
                 log_path: str | Path | None = None, device='cpu'):
        self.predictor = NFCPredictor(checkpoint, device=device)
        self.callback = callback
        self.log_path = Path(log_path) if log_path else None

    def process(self, events: Iterable[SignalEvent]):
        for event in events:
            if not isinstance(event, SignalEvent):
                raise TypeError('Source must yield SignalEvent objects containing segmented waveforms')
            result = self.predictor.predict(event.trace, signal_id=event.signal_id)
            if self.log_path:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                with self.log_path.open('a', encoding='utf-8') as f:
                    f.write(json.dumps(result) + '\n')
            if self.callback is not None:
                # Caller may implement hardware control; never claim it happened.
                self.callback(result)
            yield result
