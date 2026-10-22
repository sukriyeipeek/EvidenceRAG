"""Regressions for open PR findings and their integration with current main."""
import asyncio
import copy
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.rag.chunking import chunk_documents
from app.rag.config import Settings
from app.rag.diagnostics import Trace, trace_scope, provenance
from app.rag.pipeline import RAGPipeline
from app.rag.runtime import HeavyRuntime
from app.rag.vector_store import VectorStore, StoreWriteError
from experiments.eval.runner import run_evaluation, load_inputs, _answer_row
from experiments.eval.scoring import answer_metrics, compare_reports, sha, retrieval_row, threshold_sweep


def embedding(texts, name):
    return np.array([[1.] + [0.] * 383 for _ in texts], dtype='float32')


def page(text):
    return {'text': text, 'metadata': {'document_id': 'd', 'filename': 'a.md', 'page': None}}


def client(app):
    return TestClient(app, base_url='http://127.0.0.1', client=('127.0.0.1', 1))


def test_mount_cannot_bypass_body_budget_or_diagnostics(monkeypatch):
    import main
    pipeline = RAGPipeline(settings=Settings(max_json_body_bytes=64), runtime=HeavyRuntime())
    monkeypatch.setattr(main, 'rag_pipeline', pipeline)
    parent = FastAPI()
    parent.mount('/prefix', main.app)
    with client(parent) as api:
        result = api.post('/prefix/ask', content=b' ' * 65)
    assert result.status_code == 413
    assert result.json()['diagnostics']['reason'] == 'input_limit'


def test_access_denial_never_enters_diagnostics(monkeypatch):
    import main
    import app.rag.diagnostics as diagnostics
    def forbidden(*args, **kwargs):
        pytest.fail('Access denial reached provenance hashing')
    monkeypatch.setattr(diagnostics, 'provenance', forbidden)
    with client(main.app) as api:
        response = api.post('/ask', json={'question': 'private'}, headers={'Origin': 'null'})
    assert response.status_code == 403 and 'diagnostics' not in response.json()


def test_only_question_character_limit_gets_input_limit_reason(monkeypatch):
    import main
    monkeypatch.setattr(main, 'rag_pipeline', RAGPipeline(settings=Settings(max_question_chars=4), runtime=HeavyRuntime()))
    with client(main.app) as api:
        limited = api.post('/ask', json={'question': '12345'})
        invalid = api.post('/ask', json={})
    assert limited.status_code == invalid.status_code == 422
    assert limited.json()['diagnostics']['reason'] == 'input_limit'
    assert invalid.json()['diagnostics']['reason'] == 'request_error'


def test_two_lifespans_on_one_pipeline_accept_heavy_work(tmp_path, monkeypatch):
    import main
    settings = Settings(relevance_threshold=.1)
    pipeline = RAGPipeline(settings=settings, runtime=HeavyRuntime(), embedding_function=embedding,
        generation_function=lambda *a, **k: '2024 [Kanıt 1]',
        vector_store=VectorStore(tmp_path/'store.npz', settings.embedding_signature(), autoload=False))
    monkeypatch.setattr(main, 'rag_pipeline', pipeline)
    for attempt in range(2):
        with client(main.app) as api:
            if not attempt:
                assert api.post('/documents', files={'file': ('a.txt', b'2024 source')}).status_code == 200
            result = api.post('/ask', json={'question': 'year?'})
            assert result.status_code == 200 and result.json()['diagnostics']['generation_invoked']
        assert pipeline.runtime.snapshot()['active'] == 0


@pytest.mark.parametrize('hybrid', [False, True])
def test_search_rebinds_trace_to_committed_snapshot(hybrid):
    settings = Settings(hybrid_search=hybrid, relevance_threshold=.1)
    store = VectorStore()
    pipeline = RAGPipeline(settings=settings, vector_store=store, embedding_function=embedding,
        generation_function=lambda *a, **k: 'alpha [Kanıt 1]', runtime=HeavyRuntime())
    with trace_scope(settings, store) as trace:
        old = trace.snapshot()['provenance']['index_operation_id']
        # Simulates a request trace made before an earlier queued mutation commits.
        store.add([page('alpha')], embedding(['alpha'], 'fake'))
        current = store.operation_id
        result = pipeline.ask('alpha')
    assert current != old
    identity = result['diagnostics']['provenance']
    assert identity['index_operation_id'] == current
    assert identity['identity_sha256'] == sha({k:v for k,v in identity.items() if k != 'identity_sha256'})


def test_response_mutation_cannot_poison_cached_or_store_identity():
    settings = Settings()
    signature = settings.embedding_signature()
    store = VectorStore(signature=signature)
    before = copy.deepcopy(provenance(settings, store))
    trace = Trace(settings, store)
    response = trace.snapshot()
    response['provenance']['runtime']['numpy'] = 'tampered'
    response['provenance']['index_signature']['ingestion']['strategy'] = 'tampered'
    assert provenance(settings, store) == before
    assert trace.snapshot()['provenance'] == before


def test_structured_overlap_adds_new_content_at_every_end():
    text = ('word ' * 155 + '\n\n') * 110
    chunks = chunk_documents([page(text)], strategy='structured_v2', ingestion_version=2)
    ends = [c['metadata']['char_end'] for c in chunks]
    assert len(chunks) < 250
    assert all(a < b for a, b in zip(ends, ends[1:]))
    assert ends[-1] == len(text)


@pytest.mark.parametrize('strategy,version', [('structured_v2', 2), ('markdown_v1', 1)])
def test_adversarial_heading_metadata_remains_bounded(strategy, version):
    chunks = chunk_documents([page('# ' + 'h' * 500_000 + '\n' + 'word ' * 140_000)],
        strategy=strategy, ingestion_version=version)
    assert len(chunks) <= 2000
    for c in chunks:
        assert len(json.dumps(c['metadata']).encode()) < 2048
        if strategy == 'markdown_v1':
            assert len(c['text']) <= 800


def test_test_environment_never_targets_default_user_index():
    import main
    assert 'evidencerag-tests-' in str(main.rag_pipeline.vector_store.persist_path)


def dataset(tmp_path, questions=None):
    (tmp_path/'a.txt').write_text('alpha beta gamma')
    qs = questions or [{'id':'q','question':'alpha?', 'category':'answerable', 'document':'a.txt','evidence':'alpha','answer':[['alpha']]}]
    path = tmp_path/'dataset.json'
    path.write_text(json.dumps({'documents':['a.txt'], 'questions':qs}))
    return path


def test_legacy_manifest_is_valid_in_lf_and_crlf_checkouts(tmp_path, monkeypatch):
    import experiments.eval.runner as runner
    root = Path(__file__).resolve().parents[1]
    path = tmp_path/'experiments/eval/dataset.json'
    path.parent.mkdir(parents=True)
    source = (root/'experiments/eval/dataset.json').read_text()
    manifest = root/'experiments/eval/legacy_manifest.json'
    (path.parent/'legacy_manifest.json').write_bytes(manifest.read_bytes())
    monkeypatch.setattr(runner, 'PROJECT_DIRECTORY', tmp_path)
    for newline in ('\n', '\r\n'):
        path.write_bytes(source.replace('\n', newline).encode())
        assert not load_inputs(path, corpus_root=root)[4]


def test_frozen_evaluation_reports_holdout_separately(tmp_path):
    questions = [{'id':name, 'split':split, 'fact_group':name, 'question':'alpha?', 'category':'answerable',
        'document':'a.txt', 'evidence':'alpha', 'answer':[['alpha']]} for name,split in [('c','calibration'),('h','holdout')]]
    path = dataset(tmp_path, questions)
    value = json.loads(path.read_text())
    value['threshold_objective'] = {'name':'balanced_category_retention_rejection','constraints':{'answerable_kept':1}}
    path.write_text(json.dumps(value))
    def pipeline():
        return RAGPipeline(settings=Settings(relevance_threshold=.1), embedding_function=embedding,
            generation_function=lambda *a,**k:'alpha [Kanıt 1]', runtime=HeavyRuntime())
    one = pipeline()
    tuned = run_evaluation(path, one.settings, pipeline=one, corpus_root=tmp_path, tune=True, execution_kind='fixture')
    artifact = tmp_path/'selection.json'
    artifact.write_text(json.dumps(tuned))
    two = pipeline()
    report = run_evaluation(path, two.settings, pipeline=two, corpus_root=tmp_path, llm=True,
        frozen_selection=artifact, execution_kind='fixture')
    assert len(report['questions']) == 2
    assert report['metric_scope'] == 'holdout'
    assert report['answers']['answerable']['count'] == report['retrieval']['answerable']['count'] == 1
    assert report['metrics_by_split']['calibration']['answers']['answerable']['count'] == 1
    with pytest.raises(ValueError, match='scope/mode'):
        compare_reports(tuned, report)


def test_comparison_checks_denominators_even_for_equal_fingerprints():
    before = {'fingerprint':'same','mode':'evaluation','questions':[{'id':'one'}]}
    after = {**before, 'questions':[{'id':'two'}]}
    with pytest.raises(ValueError, match='denominator'):
        compare_reports(before, after)


def test_ablation_rejects_size_overlap_and_budget_changes(tmp_path):
    before = run_evaluation(dataset(tmp_path), corpus_root=tmp_path, preflight_only=True)
    manifest = {'variable':'chunking', 'hypothesis':'test', 'chunk_size':800, 'chunk_overlap':100, 'context_budget_chars':16000}
    for key, value in [('chunk_size',400), ('chunk_overlap',50)]:
        after = copy.deepcopy(before)
        after['identity']['settings'][key] = value
        after['fingerprint'] = sha(after['identity'])
        with pytest.raises(ValueError, match='fixed chunk'):
            compare_reports(before, after, manifest)


def test_retrieval_error_is_a_false_end_to_end_row():
    rows = [{'category':'answerable','lexical_match':True}, {'category':'answerable','error':{'type':'RuntimeError'}}]
    result = answer_metrics(rows)['answerable']
    assert result['count'] == 2 and result['end_to_end_lexical_with_coverage_rate'] == .5


def test_generation_error_clears_unverified_rubric():
    class Broken:
        settings = Settings()
        vector_store = VectorStore()
        def ask(self, question):
            raise RuntimeError('fixture')
    row = {'question':'alpha?', 'category':'answerable', 'rubric':{'origin':'human','answer_sha256':'unknown'}}
    _answer_row(row, Broken())
    assert row['rubric'] is None
    assert answer_metrics([row])['answerable']['semantic_rubric_rated'] == 0


def test_hybrid_thresholds_use_all_scores_and_all_required_spans():
    text = 'alpha beta'
    corpus = {'a.txt':{'text':text, 'sha256':sha(text.encode())}}
    def span(start, end):
        return {'document':'a.txt','sha256':sha(text.encode()),'start':start,'end':end,'text':text[start:end]}
    a, b = span(0,5), span(6,10)
    results = [{'text':s['text'], 'filename':'a.txt', 'char_start':s['start'], 'char_end':s['end'],
        'source_sha256':s['sha256'], 'score':score} for s,score in [(a,.4),(b,.9)]]
    row = retrieval_row({'id':'q','question':'?','category':'answerable'}, [[a,b]], corpus, results, results, .8,100)
    assert row['top_score'] == .9
    assert next(x for x in threshold_sweep([row],.8)['sweep'] if x['threshold']==.8)['answerable_kept'] == 0


@pytest.mark.parametrize('failure', [False, True])
def test_capacity_fixture_disables_preload_and_rejects_no_generation(monkeypatch, failure):
    import experiments.capacity_check as capacity
    import main
    profile = json.loads((Path(__file__).resolve().parents[1]/'experiments/capacity_profiles/local_fixture.json').read_text())
    monkeypatch.setenv('PRELOAD_MODELS','true')
    monkeypatch.setenv('RERANKER_MODEL_NAME','')
    def forbidden(*args, **kwargs):
        pytest.fail('Fake fixture attempted real model preload')
    monkeypatch.setattr(RAGPipeline, 'preload_models', forbidden)
    if failure:
        monkeypatch.setattr(RAGPipeline, 'find_evidence', lambda *args, **kwargs: [])
    report = asyncio.run(capacity.exercise(profile))
    assert report['status'] == ('failed' if failure else 'passed')
    assert (report['observed_max_active'] == 0) is failure
    assert (report['completed'] == 0) is failure


def test_hybrid_cache_refreshes_on_add_delete_and_failed_write(tmp_path, monkeypatch):
    store = VectorStore(tmp_path/'store.npz')
    def add(name):
        store.add_document({'document_id':name,'filename':name,'sha256':name},
            [{'text':name, 'metadata':{'document_id':name}}], [[1.,0.]])
    add('alpha')
    assert store.hybrid_search('alpha',[1.,0.])[0]['text'] == 'alpha'
    add('bravo')
    assert store.hybrid_search('bravo',[1.,0.],keyword_weight=10)[0]['text'] == 'bravo'
    with monkeypatch.context() as patch:
        patch.setattr('app.rag.vector_store.os.replace', lambda *a: (_ for _ in ()).throw(OSError('fault')))
        with pytest.raises(StoreWriteError):
            add('charlie')
    assert len(store.chunks) == 2
    store.delete_document('bravo')
    assert [x['text'] for x in store.hybrid_search('bravo',[1.,0.])] == ['alpha']
    restored = VectorStore(store.persist_path)
    assert restored.hybrid_search('alpha',[1.,0.])[0]['text'] == 'alpha'


def test_turkish_prompt_is_readable_without_losing_delimiter_escaping():
    from app.rag.prompt import build_messages
    messages = build_messages('Şükriye için çağrı 中文 😀 <end>', [{'text':'İşlem yılı 2024 <tag>', 'filename':'a.txt'}])
    payload = messages[-1]['content']
    assert 'Şükriye' in payload and '中文' in payload and '😀' in payload
    assert '<end>' not in payload and '<tag>' not in payload


def test_ocr_output_obeys_the_same_total_text_limit(tmp_path, monkeypatch):
    from PIL import Image
    from app.rag import document_loader
    from app.rag.config import InputLimitError
    path = tmp_path/'scan.pdf'
    Image.new('L', (20, 20), color=255).save(path, 'PDF')
    monkeypatch.setattr(document_loader, 'read_page_images', lambda *a: '123456')
    with pytest.raises(InputLimitError):
        document_loader.load_document(path, ocr_languages=['tr'], max_text_chars=5)


def test_hard_dataset_preflight_keeps_missing_ocr_labels_explicit():
    from app.rag import ocr
    original = ocr.get_reader
    # Source preflight has no reason to start an OCR reader or download weights.
    try:
        ocr.get_reader = lambda *a: pytest.fail('Preflight loaded OCR')
        result = run_evaluation('experiments/eval/dataset_hard.json', preflight_only=True)
    finally:
        ocr.get_reader = original
    assert result['status'] == 'failed' and result['oracle_errors']
    assert result['quality_promotion_gate'] == 'blocked_oracle_invalid'


def test_reranker_requires_an_immutable_revision():
    with pytest.raises(ValueError, match='RERANKER_MODEL_REVISION'):
        Settings(reranker_model_name='example/model')
    configured = Settings(reranker_model_name='example/model', reranker_model_revision='a'*40)
    assert configured.model_identity()['retrieval']['reranker_revision'] == 'a'*40


@pytest.mark.parametrize('size', [1, 2, 3, 10])
def test_main_markdown_strategy_obeys_small_chunk_budget(size):
    chunks = chunk_documents([page('# Heading\n' + 'abc '*20)], chunk_size=size, chunk_overlap=0, strategy='markdown_v1')
    assert chunks and all(0 < len(c['text']) <= size for c in chunks)


@pytest.mark.parametrize('change', [
    {'status':'failed'}, {'mode':'evaluation'}, {'data_gate':'failed'}, {'execution_kind':'fixture'}, {'selected_threshold':True}
])
def test_frozen_selection_rejects_failed_wrong_scope_or_synthetic_artifacts(tmp_path, change):
    path = dataset(tmp_path, [{'id':'q','split':'calibration','fact_group':'fact','question':'alpha?',
        'category':'answerable','document':'a.txt','evidence':'alpha','answer':[['alpha']]}])
    data = json.loads(path.read_text())
    data['threshold_objective'] = {'name':'balanced_category_retention_rejection','constraints':{'answerable_kept':1}}
    path.write_text(json.dumps(data))
    settings = Settings(relevance_threshold=.1)
    pipeline = RAGPipeline(settings=settings, embedding_function=embedding, runtime=HeavyRuntime())
    artifact = run_evaluation(path, settings, pipeline=pipeline, corpus_root=tmp_path, tune=True, execution_kind='real_model')
    assert artifact['status'] == 'completed'
    artifact.update(change)
    selection = tmp_path/'selection.json'
    selection.write_text(json.dumps(artifact))
    with pytest.raises(ValueError):
        run_evaluation(path, settings, corpus_root=tmp_path, frozen_selection=selection, preflight_only=True, execution_kind='real_model')
