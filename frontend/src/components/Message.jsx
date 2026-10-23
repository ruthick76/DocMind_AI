import Icon from './Icons';

export default function Message({ message }) {
  const assistant = message.role === 'assistant';
  return <article className={`message ${assistant ? 'assistant' : 'user'}`}>
    <div className="avatar">{assistant ? <Icon name="spark" size={18} /> : 'Y'}</div>
    <div className="message-body">
      <div className="message-label">{assistant ? 'DocMind' : 'You'}<span>{assistant ? 'DOCUMENT ASSISTANT' : ''}</span></div>
      <p className="answer-text">{message.content}</p>
      {!!message.sources?.length && <div className="sources">
        <div className="eyebrow">SUPPORTING SOURCES</div>
        {message.sources.map(source => <details key={source.citation_id} className="source">
          <summary><span className="citation-number">{source.citation_id}</span><Icon name="file" size={15}/><span>{source.filename}</span>{source.page && <small>Page {source.page}</small>}</summary>
          <p>{source.excerpt}</p>
          <small>Cosine similarity: {source.similarity.toFixed(3)} · Retrieval score, not answer confidence</small>
        </details>)}
      </div>}
    </div>
  </article>;
}
