"""Real LangChain OpenAI adapters; external HTTPS responses simulated in-process."""
import json
from types import SimpleNamespace

import httpx
from fastapi.testclient import TestClient
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.main import create_app
from app.services.providers import Providers
from conftest import upload


def test_real_adapters_and_structured_rag(settings):
    calls = []

    def transport(request):
        payload = json.loads(request.content)
        calls.append((request.url.path, payload))
        if request.url.path.endswith('/embeddings'):
            inputs = payload['input']
            if isinstance(inputs, str):
                inputs = [inputs]
            return httpx.Response(200, json={
                'object': 'list', 'model': 'test-embedding',
                'data': [{'object': 'embedding', 'index': i, 'embedding': [1.0, 0.0, 0.0]} for i, _ in enumerate(inputs)],
                'usage': {'prompt_tokens': 10, 'total_tokens': 10},
            })
        assert request.url.path.endswith('/chat/completions')
        assert payload['response_format']['type'] == 'json_schema'
        assert '20 days' in payload['messages'][1]['content']
        return httpx.Response(200, json={
            'id': 'chatcmpl-test', 'object': 'chat.completion', 'created': 1,
            'model': 'test-chat', 'choices': [{'index': 0, 'finish_reason': 'stop',
                'message': {'role': 'assistant', 'content': json.dumps({
                    'answer': 'Employees receive 20 days of vacation [1].',
                    'sufficient_context': True, 'citation_ids': [1]})}}],
            'usage': {'prompt_tokens': 10, 'completion_tokens': 10, 'total_tokens': 20},
        })

    with httpx.Client(transport=httpx.MockTransport(transport)) as http:
        provider = SimpleNamespace(
            embeddings=OpenAIEmbeddings(api_key='test-only', http_client=http,
                                        check_embedding_ctx_length=False),
            llm=ChatOpenAI(api_key='test-only', model='gpt-4o-mini', http_client=http),
            missing=lambda: [],
        )
        with TestClient(create_app(settings, provider)) as client:
            assert upload(client).status_code == 201
            response = client.post('/api/chat', json={'question': 'What is the vacation allowance?'})
            assert response.status_code == 200, response.text
            assert response.json()['sources'][0]['filename'] == 'policy.txt'
            assert response.json()['insufficient_context'] is False
    assert [path for path, _ in calls] == ['/v1/embeddings', '/v1/embeddings', '/v1/chat/completions']


def test_empty_specific_key_falls_back(settings):
    from pydantic import SecretStr
    settings.llm_api_key = SecretStr('')
    assert settings.key('llm') == 'test-only'


def test_runtime_provider_configuration(settings):
    provider = Providers(settings)
    assert isinstance(provider.embeddings, OpenAIEmbeddings)
    assert isinstance(provider.llm, ChatOpenAI)
    assert provider.embeddings.model == settings.embedding_model
    assert provider.llm.model_name == settings.llm_model
