from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import RunnableLambda

from app.core.config import Settings
from app.main import create_app
from app.schemas.models import GroundedAnswer


class TestEmbeddings(Embeddings):
    """Controlled vectors ONLY in tests; runtime always uses a real provider."""
    __test__ = False

    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        text = text.lower()
        # Orthogonal topic vectors make ranking and insufficient-context tests exact.
        return [float('vacation' in text or 'leave' in text),
                float('budget' in text or 'finance' in text),
                float('planet' in text or 'mars' in text), 0.01]


class TestModel:
    __test__ = False

    def __init__(self):
        self.inputs = []
        self.result = GroundedAnswer(answer="The vacation allowance is 20 days [1].",
                                     sufficient_context=True, citation_ids=[1])

    def with_structured_output(self, schema, **kwargs):
        def respond(value):
            self.inputs.append(value.to_messages())
            return self.result
        return RunnableLambda(respond)


@pytest.fixture
def settings(tmp_path):
    return Settings(_env_file=None, data_dir=tmp_path / 'data',
                    api_key='test-only', chunk_size=200, chunk_overlap=30,
                    min_similarity=0.5)


@pytest.fixture
def providers():
    return SimpleNamespace(embeddings=TestEmbeddings(), llm=TestModel(), missing=lambda: [])


@pytest.fixture
def client(settings, providers):
    with TestClient(create_app(settings, providers)) as client:
        yield client


def upload(client, text=b'Vacation policy: employees receive 20 days of leave per year.', filename='policy.txt', mime='text/plain'):
    return client.post('/api/documents/upload', files={'file': (filename, text, mime)})


def pdf_bytes(text):
    """A small valid text PDF generated without another PDF framework."""
    stream = f'BT /F1 12 Tf 50 700 Td ({text}) Tj ET'.encode()
    objects = [b'<< /Type /Catalog /Pages 2 0 R >>',
               b'<< /Type /Pages /Kids [3 0 R] /Count 1 >>',
               b'<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>',
               b'<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>',
               b'<< /Length ' + str(len(stream)).encode() + b' >>\nstream\n' + stream + b'\nendstream']
    output = b'%PDF-1.4\n'
    offsets = [0]
    for i, obj in enumerate(objects, 1):
        offsets.append(len(output))
        output += f'{i} 0 obj\n'.encode() + obj + b'\nendobj\n'
    xref = len(output)
    output += b'xref\n0 6\n0000000000 65535 f \n'
    output += b''.join(f'{offset:010} 00000 n \n'.encode() for offset in offsets[1:])
    output += f'trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF'.encode()
    return output
