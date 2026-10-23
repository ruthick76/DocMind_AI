"""Browser integration with real HTTP, loaders, FAISS, and test-only model doubles.

Run from repository root: .venv/bin/python scripts/browser_smoke.py
Needs Chromium (CHROMIUM_PATH, /usr/bin/chromium, or Playwright's own install).
Never uses the production data directory or external model credentials.
"""
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time

import httpx
from playwright.sync_api import sync_playwright, expect
import uvicorn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'backend'))
sys.path.insert(0, str(ROOT / 'backend' / 'tests'))
from app.core.config import Settings
from app.main import create_app
from conftest import TestEmbeddings, TestModel, pdf_bytes
from types import SimpleNamespace


def port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def wait_ready(url, process=None):
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError('Development server exited before becoming ready')
        try:
            if httpx.get(url, timeout=1).is_success:
                return
        except httpx.HTTPError:
            pass
        time.sleep(.1)
    raise RuntimeError('Server readiness timeout')


def main():
    backend_port, frontend_port = port(), port()
    with tempfile.TemporaryDirectory(prefix='docmind-browser-') as directory:
        settings = Settings(_env_file=None, data_dir=Path(directory), api_key='test-only', min_similarity=.5)
        providers = SimpleNamespace(embeddings=TestEmbeddings(), llm=TestModel(), missing=lambda: [])
        server = uvicorn.Server(uvicorn.Config(create_app(settings, providers), host='127.0.0.1', port=backend_port, log_level='warning'))
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        log = tempfile.TemporaryFile()
        frontend = None
        try:
            wait_ready(f'http://127.0.0.1:{backend_port}/api/health')
            frontend = subprocess.Popen(
                ['npm', 'run', 'dev', '--', '--port', str(frontend_port)],
                cwd=ROOT / 'frontend', stdout=log, stderr=log, start_new_session=True,
                env={**os.environ, 'DOCMIND_API_PROXY': f'http://127.0.0.1:{backend_port}'})
            wait_ready(f'http://127.0.0.1:{frontend_port}', frontend)
            with sync_playwright() as playwright:
                executable = os.environ.get('CHROMIUM_PATH') or shutil.which('chromium')
                browser = playwright.chromium.launch(headless=True, **({'executable_path': executable} if executable else {}))
                page = browser.new_page(viewport={'width': 1440, 'height': 1000})
                errors = []
                page.on('pageerror', lambda error: errors.append(str(error)))
                page.goto(f'http://127.0.0.1:{frontend_port}')
                expect(page.get_by_text('Workspace connected', exact=True)).to_be_visible()
                expect(page.get_by_role('heading', name='Your documents. A clearer perspective.')).to_be_visible()
                # Multiple upload selection -> sequential genuine API ingestion.
                page.locator('input[type=file]').set_input_files([
                    {'name': 'policy.txt', 'mimeType': 'text/plain', 'buffer': b'Vacation policy: employees receive 20 days of leave per year.'},
                    {'name': 'handbook.pdf', 'mimeType': 'application/pdf', 'buffer': pdf_bytes('Vacation policy allows 20 days.')},
                ])
                expect(page.get_by_role('button', name='Remove handbook.pdf', exact=True)).to_be_enabled()
                page.get_by_label('Ask a question about your documents').fill('What is the vacation allowance?')
                page.get_by_role('button', name='Send question').click()
                expect(page.get_by_text('The vacation allowance is 20 days [1].', exact=True)).to_be_visible()
                page.locator('summary').click()
                expect(page.locator('.source > p')).to_contain_text('20 days')
                # Refresh proves the history is read back via HTTP, not just React state.
                page.reload()
                page.get_by_role('button', name='What is the vacation allowance?', exact=True).click()
                expect(page.get_by_text('The vacation allowance is 20 days [1].', exact=True)).to_be_visible()
                page.get_by_role('button', name='New conversation').click()
                expect(page.get_by_role('heading', name='Your documents. A clearer perspective.')).to_be_visible()
                page.on('dialog', lambda dialog: dialog.accept())
                page.get_by_role('button', name='Delete conversation What is the vacation allowance?', exact=True).click()
                expect(page.get_by_text('Your conversations will appear here.', exact=True)).to_be_visible()
                page.get_by_role('button', name='Remove policy.txt', exact=True).click()
                expect(page.get_by_role('button', name='Remove policy.txt', exact=True)).to_have_count(0)
                page.get_by_role('button', name='Remove handbook.pdf', exact=True).click()
                expect(page.get_by_text('Add your first document to give your conversations a source of truth.', exact=True)).to_be_visible()
                page.locator('input[type=file]').set_input_files({'name': 'bad.txt', 'mimeType': 'text/plain', 'buffer': b''})
                expect(page.get_by_role('alert')).to_contain_text('empty')
                page.get_by_role('button', name='Dismiss error').click()
                page.set_viewport_size({'width': 390, 'height': 844})
                page.get_by_role('button', name='Open navigation').click()
                expect(page.locator('.sidebar')).to_be_in_viewport()
                page.get_by_role('button', name='Close navigation', exact=True).last.click()
                assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
                assert not errors, errors
                browser.close()
                print('Browser smoke passed: TXT + PDF upload, grounded answer, citation expansion, persisted history, new chat, deletion, upload error, mobile navigation; no browser runtime errors.')
        finally:
            if frontend is not None:
                import signal
                os.killpg(frontend.pid, signal.SIGTERM)
                frontend.wait(timeout=10)
            log.close()
            server.should_exit = True
            thread.join(timeout=10)


if __name__ == '__main__':
    main()
