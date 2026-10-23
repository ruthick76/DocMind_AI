import { useRef } from 'react';
import Icon from './Icons';

export default function Sidebar({ documents, chats, chatId, busy, uploadState, onUpload, onNew, onOpen, onDeleteDocument, onDeleteChat, open, onClose, health }) {
  const input = useRef(null);
  return <>
    {open && <button className="sidebar-scrim" aria-label="Close navigation" onClick={onClose}/>}
    <aside className={`sidebar ${open ? 'is-open' : ''}`}>
      <div className="brand"><span className="brand-mark"><Icon name="book" size={26}/></span><span>DocMind<span className="brand-ai">AI</span></span><button className="icon-button mobile-only" onClick={onClose} aria-label="Close navigation"><Icon name="close"/></button></div>
      <div className="workspace-label">YOUR KNOWLEDGE WORKSPACE</div>
      <button className="new-chat" disabled={busy} onClick={onNew}><Icon name="plus"/>New conversation<span>↗</span></button>
      <div className="section-heading"><h2>Document library</h2><span>{documents.length}</span></div>
      <input ref={input} type="file" accept=".pdf,.txt,application/pdf,text/plain" multiple hidden onChange={event => { onUpload(Array.from(event.target.files)); event.target.value = ''; }}/>
      <button className="upload-button" disabled={busy} onClick={() => input.current.click()}><Icon name="upload"/><span>Add documents<small>PDF or TXT · up to 10 MB by default</small></span></button>
      {uploadState && <div className="upload-status" role="status">{uploadState}</div>}
      <div className="document-list">{documents.length ? documents.map(doc => <div className="document-row" key={doc.id}>
        <span className="file-icon"><Icon name="file" size={19}/></span><div className="row-text"><strong title={doc.filename}>{doc.filename}</strong><small>{doc.chunk_count} chunks · {(doc.size_bytes / 1024).toFixed(1)} KB</small></div>
        <button className="icon-button row-action" disabled={busy} onClick={() => onDeleteDocument(doc)} aria-label={`Remove ${doc.filename}`}><Icon name="trash" size={16}/></button>
      </div>) : <p className="sidebar-empty">Add your first document to give your conversations a source of truth.</p>}</div>
      <div className="section-heading history-heading"><h2>Conversations</h2><span>{chats.length}</span></div>
      <nav className="chat-list" aria-label="Saved conversations">{chats.length ? chats.map(chat => <div key={chat.id} className={`chat-row ${chatId === chat.id ? 'active' : ''}`}>
        <button disabled={busy} onClick={() => onOpen(chat.id)}><Icon name="chat" size={16}/><span>{chat.title}</span></button>
        <button className="icon-button row-action" disabled={busy} onClick={() => onDeleteChat(chat.id)} aria-label={`Delete conversation ${chat.title}`}><Icon name="trash" size={15}/></button>
      </div>) : <p className="sidebar-empty">Your conversations will appear here.</p>}</nav>
      <div className="workspace-footer"><span className={`status-dot ${health?.configured && health?.storage_ready ? '' : 'warning'}`}/><div><strong>{health?.configured && health?.storage_ready ? 'Workspace connected' : 'Setup required'}</strong><small>Local storage · private workspace</small></div><span className="version">v1.0</span></div>
    </aside>
  </>;
}
