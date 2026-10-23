"""Per-generation admission shared by gateways and direct model clients."""
from __future__ import annotations

from contextlib import contextmanager
import time

from .generation_stop_observer import generation_stopped


class BatchClientAdmission:
    def __init__(self, config, *, pool, agent_id, deadline, admission_client=None):
        from .document_batch_admission import AdmissionClient
        self.client = admission_client or AdmissionClient(config)
        self.pool, self.agent_id, self.deadline = pool, agent_id, deadline
        self.active = None
        self.history = []

    def check_active(self):
        if self.active is None:
            raise RuntimeError('Generation has no batch admission lease')
        if time.monotonic() >= self.active['deadline']:
            raise TimeoutError('Batch generation expired before dispatch')
        valid = self.active.get('is_valid')
        if valid is not None and not valid():
            raise RuntimeError('Batch generation caller is no longer active')

    @contextmanager
    def slot(self, *, deadline=None, is_valid=None):
        if self.active is not None:
            self.check_active()
            yield self.active
            return
        limit = min(self.deadline, deadline if deadline is not None else self.deadline)
        if time.monotonic() >= limit or (is_valid is not None and not is_valid()):
            raise TimeoutError('Batch generation expired or caller closed before queuing')
        started = time.monotonic()
        lease = self.client.acquire(self.pool, self.agent_id, limit)
        limit = min(limit, lease.get('deadline', limit))
        state = {'lease': lease, 'deadline': limit, 'is_valid': is_valid,
                 'queue_seconds': time.monotonic() - started,
                 'transport_entered': False, 'receipt': None}
        self.active = state
        error = None
        try:
            self.check_active()
            yield state
        except BaseException as exc:
            error = exc
            raise
        finally:
            receipt = state['receipt']
            if not state['transport_entered'] or (receipt is not None and receipt.get('generation_attempted') is False):
                disposition = 'not_sent'
            elif receipt is not None and (receipt.get('done_received') is True or generation_stopped(receipt)):
                disposition = 'complete'
            else:
                disposition = 'unknown'
            evidence = {key: receipt.get(key) for key in ('call_id', 'request_id', 'response_id', 'model',
                'done_received', 'finish_reason', 'cancel_status', 'generation_attempted', 'stop_observation',
                'native_stop_observation') if receipt is not None and key in receipt}
            note = {'pool': self.pool, 'agent_id': self.agent_id,
                    'queue_seconds': state['queue_seconds'], 'state': disposition}
            try:
                note['settlement'] = self.client.settle(lease, disposition, evidence)
            except Exception as exc:
                note['settlement_error'] = type(exc).__name__
                # The broker retains its active/unknown slot. A lost local
                # acknowledgement is never evidence that upstream stopped.
                if error is None:
                    raise RuntimeError('Batch admission settlement could not be confirmed') from exc
            finally:
                if receipt is not None:
                    receipt['batch_admission'] = note
                self.history.append(note)
                self.active = None

    def transport_started(self):
        self.check_active()
        self.active['transport_entered'] = True

    def observe(self, receipt):
        if self.active is None:
            raise RuntimeError('Cannot attach response without admission')
        self.active['receipt'] = receipt
