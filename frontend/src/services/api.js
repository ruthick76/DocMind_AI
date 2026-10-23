const base = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '');

export async function api(path, options = {}) {
  // Several sequential embedding batches can take longer than a single model call.
  const timeout = options.timeout || 180_000;
  const { timeout: _, ...requestOptions } = options;
  let response;
  try {
    response = await fetch(`${base}/api${path}`, {
      ...requestOptions,
      signal: AbortSignal.timeout(timeout),
    });
  } catch (error) {
    throw new Error(error.name === 'TimeoutError'
      ? 'The request timed out. Refresh the library or conversation before retrying; server processing may still finish.'
      : 'Cannot reach DocMind. Check that the backend is running.');
  }
  if (response.status === 204) return null;
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error?.message || `Request failed (${response.status}).`);
  return data;
}

export const sendQuestion = (question, chatId) => api('/chat', {
  method: 'POST', headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ question, chat_id: chatId }),
});
