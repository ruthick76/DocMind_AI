import { useEffect, useRef, useState } from 'react';
import { api, sendQuestion } from './services/api';
import Icon from './components/Icons';
import Message from './components/Message';
import Sidebar from './components/Sidebar';

export default function App() {
  const [documents, setDocuments] = useState([]);
  const [chats, setChats] = useState([]);
  const [health, setHealth] = useState(null);
  const [chatId, setChatId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [question, setQuestion] = useState('');
  const [busy, setBusy] = useState(false);
  const [sending, setSending] = useState(false);
  const [error, setError] = useState('');
  const [uploadState, setUploadState] = useState('');
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const end = useRef(null);
  const textarea = useRef(null);

  async function refresh() {
    const [nextHealth, nextDocuments, nextChats] = await Promise.all([
      api('/health'), api('/documents'), api('/chats'),
    ]);
    setHealth(nextHealth); setDocuments(nextDocuments); setChats(nextChats);
  }
  useEffect(() => { refresh().catch(error => setError(error.message)); }, []);
  useEffect(() => { end.current?.scrollIntoView({ behavior: 'smooth', block: 'end' }); }, [messages, sending]);

  function newChat() {
    setChatId(null); setMessages([]); setQuestion(''); setError(''); setSidebarOpen(false);
    textarea.current?.focus();
  }
  async function openChat(id) {
    setBusy(true); setError('');
    try { const chat = await api(`/chats/${id}`); setChatId(chat.id); setMessages(chat.messages); setQuestion(''); setSidebarOpen(false); }
    catch (error) { setError(error.message); }
    finally { setBusy(false); }
  }
  async function upload(files) {
    if (!files.length) return;
    setBusy(true); setError(''); let completed = 0;
    try {
      for (const file of files) {
        setUploadState(`Processing ${completed + 1}/${files.length}: ${file.name}`);
        const form = new FormData(); form.append('file', file);
        const result = await api('/documents/upload', { method: 'POST', body: form });
        completed++;
        setUploadState(result.duplicate ? `${file.name} is already indexed.` : `${completed} document(s) indexed.`);
        await refresh();
      }
    } catch (error) { setError(error.message); setUploadState(`${completed}/${files.length} files processed. Remaining files were not uploaded.`); }
    finally { setBusy(false); }
  }
  async function removeDocument(doc) {
    if (!window.confirm(`Remove ${doc.filename} and all of its searchable chunks? Saved chat citations will remain as historical excerpts.`)) return;
    setBusy(true); setError('');
    try { await api(`/documents/${doc.id}`, { method: 'DELETE' }); await refresh(); }
    catch (error) { setError(error.message); }
    finally { setBusy(false); }
  }
  async function deleteChat(id) {
    if (!window.confirm('Permanently delete this conversation and its history?')) return;
    setBusy(true); setError('');
    try { await api(`/chats/${id}`, { method: 'DELETE' }); if (chatId === id) newChat(); await refresh(); }
    catch (error) { setError(error.message); }
    finally { setBusy(false); }
  }
  async function submit(event) {
    event.preventDefault(); const text = question.trim();
    if (!text || busy) return;
    setBusy(true); setSending(true); setError(''); setQuestion('');
    const prior = messages;
    setMessages([...prior, { role: 'user', content: text }]);
    try {
      const result = await sendQuestion(text, chatId);
      setChatId(result.chat_id);
      setMessages([...prior, { role: 'user', content: text }, { role: 'assistant', content: result.answer, sources: result.sources }]);
      // A refresh failure must not roll back a successfully saved answer.
      await refresh().catch(error => setError(error.message));
    } catch (error) { setMessages(prior); setQuestion(text); setError(error.message); }
    finally { setBusy(false); setSending(false); textarea.current?.focus(); }
  }
  const suggestions = ['Summarize the key points', 'What are the main risks?', 'Find deadlines and action items'];
  return <div className="app-shell">
    <Sidebar {...{documents, chats, chatId, busy, uploadState, health}} onUpload={upload} onNew={newChat} onOpen={openChat} onDeleteDocument={removeDocument} onDeleteChat={deleteChat} open={sidebarOpen} onClose={() => setSidebarOpen(false)}/>
    <main className="main-panel">
      <header className="topbar"><div><button className="icon-button mobile-only" aria-label="Open navigation" onClick={() => setSidebarOpen(true)}><Icon name="menu"/></button><span className="breadcrumb">Workspace</span><span className="breadcrumb-divider">/</span><strong>{chatId ? 'Conversation' : 'New conversation'}</strong></div><span className="grounded-badge"><Icon name="check" size={14}/>Grounded in your documents</span></header>
      {error && <div className="error-banner" role="alert"><span>{error}</span><button className="icon-button" onClick={() => setError('')} aria-label="Dismiss error"><Icon name="close" size={16}/></button></div>}
      {health && (!health.configured || !health.storage_ready) && <div className="setup-banner" role="status">{!health.storage_ready ? health.message : 'Connect your model provider in backend/.env to enable document indexing and AI answers. API keys stay on the backend.'}<button className="text-button" disabled={busy} onClick={() => refresh().catch(error => setError(error.message))}>Check again</button></div>}
      <div className={`conversation-area ${!messages.length ? 'empty' : ''}`}>
        {!messages.length ? <section className="welcome">
          <div className="welcome-icon"><Icon name="spark" size={32}/></div>
          <div className="eyebrow welcome-eyebrow">LESS SEARCHING. MORE UNDERSTANDING.</div>
          <h1>Your documents.<br/><em>A clearer perspective.</em></h1>
          <p>Turn the information you have into the answers you need.<br className="desktop-break"/> Add a document, ask a question, and follow the sources.</p>
          <div className="workflow"><div><span>01</span><strong>Add your documents</strong><small>PDFs and text files, in one place.</small></div><div><span>02</span><strong>Ask in your own words</strong><small>Explore the details that matter.</small></div><div><span>03</span><strong>See where it came from</strong><small>Every answer connects to evidence.</small></div></div>
          <div className="suggestions-label">A FEW WAYS TO GET STARTED</div>
          <div className="suggestions">{suggestions.map(text => <button key={text} disabled={busy} onClick={() => { setQuestion(text); textarea.current?.focus(); }}>{text}<span>↗</span></button>)}</div>
        </section> : <div className="message-list" aria-live="polite" aria-relevant="additions">{messages.map((message, i) => <Message key={i} message={message}/>)}{sending && <div className="thinking" role="status"><span className="spinner"/>Finding evidence and preparing an answer…</div>}<div ref={end}/></div>}
      </div>
      <div className="composer-container">
        <form className="composer" onSubmit={submit}>
          <label className="sr-only" htmlFor="question">Ask a question about your documents</label>
          <textarea id="question" ref={textarea} value={question} maxLength={4000} rows={2} disabled={busy} placeholder="Ask anything about your documents…" onChange={event => setQuestion(event.target.value)} onKeyDown={event => { if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) { event.preventDefault(); submit(event); } }}/>
          <div className="composer-bottom"><span><Icon name="file" size={14}/>{documents.length ? `${documents.length} document${documents.length > 1 ? 's' : ''} in your library` : 'Add documents to get started'}</span><div><small>Enter to send</small><button className="send-button" type="submit" disabled={busy || !question.trim()} aria-label="Send question">{sending ? <span className="spinner"/> : <Icon name="arrow" size={20}/>}</button></div></div>
        </form>
        <p className="composer-note">AI can make mistakes. Review the source excerpts for important decisions.</p>
      </div>
    </main>
  </div>;
}
