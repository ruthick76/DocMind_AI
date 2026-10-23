from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.schemas.models import GroundedAnswer
from conftest import upload


def test_document_to_answer_integration(client, providers, settings):
    """Real extraction, LangChain splitting, FAISS, prompt, API, and disk history."""
    info = upload(client).json()['document']
    result = client.post('/api/chat', json={'question': 'What is the vacation allowance?'})
    assert result.status_code == 200
    answer = result.json()
    assert '20 days' in answer['answer']
    assert answer['sources'][0]['document_id'] == info['id']
    assert '20 days' in answer['sources'][0]['excerpt']
    assert answer['insufficient_context'] is False
    prompt = providers.llm.inputs[0]
    assert prompt[0].type == 'system'
    assert 'Vacation policy' not in prompt[0].content
    assert 'Vacation policy' in prompt[1].content
    conversation = client.get(f"/api/chats/{answer['chat_id']}").json()
    assert [m['role'] for m in conversation['messages']] == ['user', 'assistant']
    followup = client.post('/api/chat', json={'question': 'Can you repeat the allowance?', 'chat_id': answer['chat_id']})
    assert followup.status_code == 200
    assert len(client.get(f"/api/chats/{answer['chat_id']}").json()['messages']) == 4
    with TestClient(create_app(settings, providers)) as restarted:
        assert restarted.get('/api/chats').json()[0]['message_count'] == 4
        assert restarted.delete(f"/api/chats/{answer['chat_id']}").status_code == 204
        assert restarted.get(f"/api/chats/{answer['chat_id']}").status_code == 404
    with TestClient(create_app(settings, providers)) as restarted:
        assert restarted.get('/api/chats').json() == []


def test_no_documents_honest_insufficient_answer(client, providers):
    result = client.post('/api/chat', json={'question': 'What is the vacation allowance?'}).json()
    assert result['insufficient_context'] is True
    assert result['sources'] == []
    assert providers.llm.inputs == []


def test_unrelated_question_no_context(client, providers):
    upload(client)
    result = client.post('/api/chat', json={'question': 'What planet is Mars?'}).json()
    assert result['insufficient_context'] is True
    assert providers.llm.inputs == []


def test_llm_insufficient_answer(client, providers):
    upload(client)
    providers.llm.result = GroundedAnswer(answer='No evidence', sufficient_context=False, citation_ids=[])
    result = client.post('/api/chat', json={'question': 'vacation'}).json()
    assert result['insufficient_context'] is True
    assert result['sources'] == []
    assert 'enough information' in result['answer']


@pytest.mark.parametrize('ids', [[99], []])
def test_invalid_citations_fail_without_saving(client, providers, ids):
    upload(client)
    providers.llm.result = GroundedAnswer(answer='An answer', sufficient_context=True, citation_ids=ids)
    result = client.post('/api/chat', json={'question': 'vacation'})
    assert result.status_code == 502
    assert client.get('/api/chats').json() == []


def test_generation_failure_does_not_save(client, monkeypatch):
    upload(client)
    def fail(*args, **kwargs):
        raise RuntimeError('provider key and details')
    monkeypatch.setattr(client.app.state.providers.llm, 'with_structured_output', fail)
    result = client.post('/api/chat', json={'question': 'vacation'})
    assert result.status_code == 502
    assert 'provider key' not in result.text
    assert client.get('/api/chats').json() == []


@pytest.mark.parametrize('payload', [{'question': ''}, {'question': '   '}, {'question': 'x' * 4001}, {}, {'question': 'hi', 'chat_id': 'bad'}])
def test_chat_validation(client, payload):
    assert client.post('/api/chat', json=payload).status_code == 422


def test_nonexistent_chat(client):
    assert client.post('/api/chat', json={'question': 'hi', 'chat_id': str(uuid4())}).status_code == 404


def test_health_missing_configuration(settings):
    settings.api_key = None
    with TestClient(create_app(settings)) as client:
        data = client.get('/api/health').json()
        assert data['status'] == 'configuration_required'
        assert data['configured'] is False
        assert 'test-only' not in str(data)
        assert upload(client).status_code == 503


def test_cors(client):
    response = client.options('/api/chat', headers={
        'Origin': 'http://localhost:5173', 'Access-Control-Request-Method': 'POST',
        'Access-Control-Request-Headers': 'Content-Type'})
    assert response.headers['access-control-allow-origin'] == 'http://localhost:5173'
    denied = client.options('/api/chat', headers={
        'Origin': 'https://untrusted.example', 'Access-Control-Request-Method': 'POST'})
    assert denied.status_code == 400


def test_inline_citation_must_match_structured_ids(client, providers):
    upload(client)
    providers.llm.result = GroundedAnswer(answer='An answer [99]', sufficient_context=True, citation_ids=[1])
    assert client.post('/api/chat', json={'question': 'vacation'}).status_code == 502
