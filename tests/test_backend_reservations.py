import pytest

from bridge_evidence.contest.backends import AccountedBackend
from bridge_evidence.contest.timing import TimingLedger


def test_backend_keeps_trusted_reserved_id(tmp_path):
    from bridge_evidence.contest.rsi_runtime import REQUEST_CALL_ID
    ledger = TimingLedger(tmp_path / 'ledger.sqlite')
    backend = AccountedBackend(lambda ids, maximum: {'text': '{}', 'finish_reason': 'stop'}, ledger)
    token = REQUEST_CALL_ID.set('reserved-1')
    try:
        result = backend([1], 10)
    finally:
        REQUEST_CALL_ID.reset(token)
    assert result['call_id'] == 'reserved-1'
    with ledger.connect() as db:
        assert db.execute('SELECT id FROM calls').fetchall() == [('reserved-1',)]


def test_duplicate_reservation_never_reissues_model_request(tmp_path):
    from bridge_evidence.contest.rsi_runtime import REQUEST_CALL_ID
    calls = []
    def generate(ids, maximum):
        calls.append(1)
        return {'text': '{}', 'finish_reason': 'stop'}
    backend = AccountedBackend(generate, TimingLedger(tmp_path / 'ledger.sqlite'))
    token = REQUEST_CALL_ID.set('reserved-1')
    try:
        backend([1], 10)
        with pytest.raises((ValueError, RuntimeError)):
            backend([1], 10)
    finally:
        REQUEST_CALL_ID.reset(token)
    assert calls == [1]


def test_batch_claim_rolls_back_all_new_ids_on_any_duplicate(tmp_path):
    ledger = TimingLedger(tmp_path / 'ledger.sqlite')
    ledger.claim('existing', role='extraction')
    with pytest.raises(ValueError, match='already dispatched'):
        ledger.claim_many(['new', 'existing'], role='extraction')
    with ledger.connect() as db:
        assert db.execute('SELECT id FROM calls').fetchall() == [('existing',)]


def test_concurrent_claim_has_exactly_one_owner(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    ledger = TimingLedger(tmp_path / 'ledger.sqlite')
    def claim(_):
        try:
            ledger.claim('one', role='extraction')
            return True
        except ValueError:
            return False
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert sum(pool.map(claim, range(4))) == 1


def test_local_batch_bad_sibling_keeps_success_and_settles_all(tmp_path):
    class Partial:
        def batch(self, requests):
            return [{'text': 'ok', 'finish_reason': 'stop', 'output_tokens': 1},
                    {'text': None, 'finish_reason': 'stop', 'output_tokens': 1}]

    ledger = TimingLedger(tmp_path / 'ledger.sqlite')
    result = AccountedBackend(Partial(), ledger).batch([([1], 3), ([2], 3)])
    assert result[0]['text'] == 'ok'
    assert isinstance(result[1], ValueError)
    assert result[1].call_id != result[0]['call_id']
    with ledger.connect() as db:
        assert [r[0] for r in db.execute('SELECT status FROM calls ORDER BY rowid')] == ['success', 'failed']
    assert ledger.summary()['incomplete_call_ids'] == []


@pytest.mark.parametrize('error', [KeyboardInterrupt('stop'), SystemExit('stop')])
def test_terminated_local_batch_records_interrupted_not_failed(tmp_path, error):
    class Interrupted:
        def batch(self, requests):
            raise error

    ledger = TimingLedger(tmp_path / 'ledger.sqlite')
    with pytest.raises(type(error)):
        AccountedBackend(Interrupted(), ledger).batch([([1], 3), ([2], 3)])
    with ledger.connect() as db:
        assert [r[0] for r in db.execute('SELECT status FROM calls')] == ['interrupted', 'interrupted']


def test_batch_settlement_error_does_not_overwrite_prior_success(tmp_path, monkeypatch):
    class Complete:
        def batch(self, requests):
            return [{'text': 'ok', 'finish_reason': 'stop', 'output_tokens': 1} for _ in requests]

    ledger = TimingLedger(tmp_path / 'ledger.sqlite')
    finish = ledger.finish
    count = 0

    def fail_once(*args, **kwargs):
        nonlocal count
        count += 1
        if count == 2:
            raise OSError('transient ledger write error')
        return finish(*args, **kwargs)

    monkeypatch.setattr(ledger, 'finish', fail_once)
    with pytest.raises(OSError, match='ledger write'):
        AccountedBackend(Complete(), ledger).batch([([1], 3), ([2], 3), ([3], 3)])
    with ledger.connect() as db:
        states = [row[0] for row in db.execute('SELECT status FROM calls ORDER BY rowid')]
    assert states == ['success', 'failed', 'failed']


def test_local_vendor_request_identity_is_preserved_and_scoped(tmp_path, monkeypatch):
    import json
    import sys
    from types import ModuleType, SimpleNamespace

    from bridge_evidence.contest.backends import LocalVLLMBackend

    vllm = ModuleType('vllm')
    request = ModuleType('vllm.lora.request')
    class LLM:
        def __init__(self, **kwargs):
            pass
        def generate(self, requests, sampling, **kwargs):
            return [SimpleNamespace(request_id=str(i), outputs=[SimpleNamespace(
                text='ok', finish_reason='stop', token_ids=[1])]) for i, _ in enumerate(requests)]
    vllm.LLM, vllm.SamplingParams = LLM, lambda **kwargs: kwargs
    request.LoRARequest = lambda *args: args
    monkeypatch.setitem(sys.modules, 'vllm', vllm)
    monkeypatch.setitem(sys.modules, 'vllm.lora.request', request)
    manifest = SimpleNamespace(validate=lambda: None, merged='fixed', base='base', adapter=None)
    first, second = LocalVLLMBackend(manifest), LocalVLLMBackend(manifest)
    ledger = TimingLedger(tmp_path / 'ledger.sqlite')
    a = AccountedBackend(first, ledger)([1], 4)
    b = AccountedBackend(second, ledger)([1], 4)
    assert a['backend_request_id'] == b['backend_request_id'] == '0'
    assert a['backend_instance_id'] != b['backend_instance_id']
    with ledger.connect() as db:
        receipts = [json.loads(row[0]) for row in db.execute('SELECT receipt FROM calls ORDER BY rowid')]
    assert receipts[0]['backend_instance_id'] == a['backend_instance_id']
    assert receipts[1]['backend_instance_id'] == b['backend_instance_id']


def test_api_batch_preserves_report_deadline_in_request_threads(tmp_path):
    from bridge_evidence.contest.backends import APIBackend
    from bridge_evidence.contest.rsi_runtime import REPORT_DEADLINE

    deadlines = []
    class API(APIBackend):
        def __init__(self):
            self.workers = 2
        def __call__(self, ids, maximum):
            deadlines.append(REPORT_DEADLINE.get())
            return {'text': 'ok', 'finish_reason': 'stop'}
    token = REPORT_DEADLINE.set(12345.0)
    try:
        AccountedBackend(API(), TimingLedger(tmp_path / 'ledger.sqlite')).batch([([1], 3), ([2], 3)])
    finally:
        REPORT_DEADLINE.reset(token)
    assert deadlines == [12345.0, 12345.0]
