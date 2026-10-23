import json
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from conftest import pdf_bytes, upload


@pytest.mark.parametrize('filename,mime,data,status,code', [
    ('x.exe', 'application/octet-stream', b'abc', 415, 'unsupported_file'),
    ('x.txt', 'application/pdf', b'abc', 415, 'invalid_mime'),
    ('x.pdf', 'application/pdf', b'not pdf', 422, 'invalid_pdf'),
    ('x.pdf', 'application/pdf', b'%PDF-1.4 broken', 422, 'extraction_failed'),
    ('x.txt', 'text/plain', b'', 422, 'empty_file'),
    ('x.txt', 'text/plain', b'   \n', 422, 'no_text'),
    ('x.txt', 'text/plain', b'\xff\xfe', 422, 'extraction_failed'),
])
def test_validation(client, filename, mime, data, status, code):
    result = upload(client, data, filename, mime)
    assert result.status_code == status
    assert result.json()['error']['code'] == code
    assert client.get('/api/documents').json() == []


def test_size_limit(client):
    client.app.state.settings.max_upload_mb = 1
    result = upload(client, b'a' * (1024 * 1024 + 1))
    assert result.status_code == 413


def test_safe_filename_duplicate_and_metadata(client):
    result = upload(client, filename='../../policy.txt')
    assert result.status_code == 201
    document = result.json()['document']
    assert document['filename'] == 'policy.txt'
    assert client.get(f"/api/documents/{document['id']}").json() == document
    duplicate = upload(client, filename='renamed.txt')
    assert duplicate.status_code == 200
    assert duplicate.json()['duplicate'] is True
    assert len(client.get('/api/documents').json()) == 1


def test_chunking(client):
    result = upload(client, ('Vacation policy. ' * 100).encode())
    assert result.status_code == 201
    assert result.json()['document']['chunk_count'] > 1
    assert all(len(chunk.text) <= 200 for chunk in client.app.state.store.chunks)


def test_pdf_extract_and_page(client):
    result = upload(client, pdf_bytes('Vacation allowance is 20 days.'), 'policy.pdf', 'application/pdf')
    assert result.status_code == 201
    chunk = client.app.state.store.chunks[0]
    assert '20 days' in chunk.text
    assert chunk.page == 1


def test_extracted_text_limit(client):
    client.app.state.settings.max_extracted_chars = 1000
    assert upload(client, b'vacation ' * 200).status_code == 413


def test_chunk_limit(client):
    client.app.state.settings.max_chunks_per_document = 1
    assert upload(client, b'vacation ' * 200).status_code == 413


def test_restart_and_delete_no_stale_chunks(client, settings, providers):
    first = upload(client).json()['document']
    second = upload(client, b'Finance budget is 5 million.', 'budget.txt').json()['document']
    with TestClient(create_app(settings, providers)) as restarted:
        assert len(restarted.get('/api/documents').json()) == 2
        store = restarted.app.state.store
        assert 'Vacation' in store.search('vacation')[0][0].page_content
        assert restarted.delete(f"/api/documents/{first['id']}").status_code == 204
        assert store.search('vacation') == []
        assert 'Finance' in store.search('budget')[0][0].page_content
    with TestClient(create_app(settings, providers)) as restarted:
        assert [d['id'] for d in restarted.get('/api/documents').json()] == [second['id']]
        assert restarted.app.state.store.search('vacation') == []
        assert restarted.delete(f"/api/documents/{second['id']}").status_code == 204
    with TestClient(create_app(settings, providers)) as restarted:
        assert restarted.app.state.store.index is None
        assert restarted.get('/api/documents').json() == []


def test_embedding_failure_leaves_no_partial_document(client, monkeypatch):
    def fail(texts):
        raise RuntimeError('secret provider detail')
    monkeypatch.setattr(client.app.state.providers.embeddings, 'embed_documents', fail)
    result = upload(client)
    assert result.status_code == 502
    assert 'secret' not in result.text
    assert client.get('/api/documents').json() == []


@pytest.mark.parametrize('vectors', [[[float('nan'), 1]], [[0, 0]], [[1, 2], [1, 2]]])
def test_invalid_provider_vectors(client, monkeypatch, vectors):
    monkeypatch.setattr(client.app.state.providers.embeddings, 'embed_documents', lambda texts: vectors)
    assert upload(client).status_code == 502
    assert client.app.state.store.index is None


def test_corrupt_index_fails_closed(client, settings, providers):
    upload(client)
    root = settings.data_dir
    generation = json.loads((root / 'current.json').read_text())['generation']
    (root / 'snapshots' / generation / 'index.faiss').write_bytes(b'corrupt')
    with TestClient(create_app(settings, providers)) as restarted:
        assert restarted.get('/api/health').json()['storage_ready'] is False
        assert restarted.get('/api/documents').status_code == 503
        assert upload(restarted).status_code == 503


def test_embedding_model_change_requires_new_index(client, settings, providers):
    upload(client)
    changed = settings.model_copy(update={'embedding_model': 'different-model'})
    with TestClient(create_app(changed, providers)) as restarted:
        assert restarted.get('/api/health').json()['storage_ready'] is False


def test_snapshot_path_traversal_rejected(settings, providers):
    settings.data_dir.mkdir()
    (settings.data_dir / 'current.json').write_text('{"generation":"../../elsewhere"}')
    with TestClient(create_app(settings, providers)) as client:
        assert client.get('/api/health').json()['storage_ready'] is False


def test_missing_document_and_bad_identifier(client):
    assert client.get(f'/api/documents/{uuid4()}').status_code == 404
    assert client.delete('/api/documents/not-a-uuid').status_code == 422


def test_atomic_pointer_failure_preserves_previous_snapshot(client, settings, providers, monkeypatch):
    import app.services.storage as storage
    first = upload(client).json()['document']
    original_atomic = storage.atomic_json

    def fail_pointer(path, value):
        if path.name == 'current.json':
            raise OSError('disk failure')
        return original_atomic(path, value)

    monkeypatch.setattr(storage, 'atomic_json', fail_pointer)
    result = upload(client, b'Finance budget is 5 million.', 'finance.txt')
    assert result.status_code == 503
    with TestClient(create_app(settings, providers)) as restarted:
        assert restarted.get('/api/health').json()['storage_ready'] is True
        assert [d['id'] for d in restarted.get('/api/documents').json()] == [first['id']]
        assert restarted.app.state.store.search('budget') == []


def test_invalid_metadata_fails_closed(client, settings, providers):
    upload(client)
    root = settings.data_dir
    generation = json.loads((root / 'current.json').read_text())['generation']
    metadata = root / 'snapshots' / generation / 'metadata.json'
    state = json.loads(metadata.read_text())
    state['chunks'][0]['document_id'] = str(uuid4())
    metadata.write_text(json.dumps(state))
    with TestClient(create_app(settings, providers)) as restarted:
        assert restarted.get('/api/documents').status_code == 503


def test_cleanup_retains_committed_snapshot_not_orphan(client, settings):
    upload(client)
    previous = client.app.state.store.generation
    orphan = settings.data_dir / 'snapshots' / ('f' * 32)
    orphan.mkdir()
    upload(client, b'Finance budget is 5 million.', 'finance.txt')
    current = client.app.state.store.generation
    assert {p.name for p in (settings.data_dir / 'snapshots').iterdir()} == {previous, current}
