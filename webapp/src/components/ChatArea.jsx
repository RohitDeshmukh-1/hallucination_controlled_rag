import { useState, useRef, useEffect } from 'react';
import './ChatArea.css';

const SendIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
    <line x1="22" y1="2" x2="11" y2="13" /><polygon points="22 2 15 22 11 13 2 9 22 2" />
  </svg>
);
const UploadIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
    <polyline points="17 8 12 3 7 8" /><line x1="12" y1="3" x2="12" y2="15" />
  </svg>
);
const PinIcon = () => (
  <svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor">
    <path d="M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7zm0 9.5c-1.38 0-2.5-1.12-2.5-2.5s1.12-2.5 2.5-2.5 2.5 1.12 2.5 2.5-1.12 2.5-2.5 2.5z"/>
  </svg>
);
const ChevronIcon = ({ open }) => (
  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"
    style={{ transform: open ? 'rotate(180deg)' : 'rotate(0)', transition: '200ms' }}>
    <polyline points="6 9 12 15 18 9" />
  </svg>
);
const BotIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <rect x="3" y="11" width="18" height="10" rx="2" /><path d="M12 11V7" />
    <circle cx="12" cy="5" r="2" /><line x1="8" y1="16" x2="8" y2="16" strokeWidth="3" strokeLinecap="round" />
    <line x1="12" y1="16" x2="12" y2="16" strokeWidth="3" strokeLinecap="round" />
    <line x1="16" y1="16" x2="16" y2="16" strokeWidth="3" strokeLinecap="round" />
  </svg>
);

function VerdictBadge({ verdict }) {
  const map = {
    supported: ['badge-success', '✓ Supported'],
    partially_supported: ['badge-warning', '⚬ Partial'],
    refused: ['badge-error', '✕ Refused'],
    loading: ['badge-muted', '…'],
  };
  const [cls, label] = map[verdict] || ['badge-muted', verdict];
  return <span className={`badge ${cls}`}>{label}</span>;
}

function ConfidenceBar({ value }) {
  const pct = Math.round((value || 0) * 100);
  return (
    <div className="confidence-row">
      <span className="conf-label">Confidence</span>
      <div className="progress-bar conf-bar">
        <div className="progress-fill" style={{ width: `${pct}%`,
          background: pct > 75 ? 'linear-gradient(90deg,#10b981,#34d399)'
            : pct > 45 ? 'linear-gradient(90deg,#f59e0b,#fbbf24)'
            : 'linear-gradient(90deg,#ef4444,#f87171)' }} />
      </div>
      <span className="conf-pct">{pct}%</span>
    </div>
  );
}

function CitationList({ citations }) {
  if (!citations?.length) return null;
  return (
    <div className="citation-list">
      {citations.map((c, i) => (
        <span key={i} className="citation-chip" title={c.text_preview}>
          [{c.evidence_id}] p.{c.pages}
        </span>
      ))}
    </div>
  );
}

function MessageBubble({ turn, onTurnClick, onPin }) {
  const [showEvidence, setShowEvidence] = useState(false);
  const isLoading = turn.verdict === 'loading';

  const handlePin = (e) => {
    e.stopPropagation();
    const text = window.getSelection()?.toString() || turn.answer?.slice(0, 120) + '…';
    onPin(text, turn.question);
  };

  return (
    <div className="message-group fade-up" onClick={() => onTurnClick(turn)}>
      {/* User */}
      <div className="user-row">
        <div className="user-bubble">{turn.question}</div>
        {turn.rewritten_question && (
          <div className="rewrite-hint">↻ Context-expanded: "{turn.rewritten_question.slice(0, 60)}…"</div>
        )}
      </div>

      {/* AI */}
      <div className="ai-row">
        <div className="ai-avatar"><BotIcon /></div>
        <div className="ai-card">
          {isLoading ? (
            <div className="ai-loading">
              <span className="spinner" /> Analysing evidence…
            </div>
          ) : (
            <>
              <div className="ai-header">
                <VerdictBadge verdict={turn.verdict} />
                {turn.verdict !== 'refused' && turn.confidence != null && (
                  <ConfidenceBar value={turn.confidence} />
                )}
                <span className="ai-ts">{turn.timestamp}</span>
                <button className="pin-btn" onClick={handlePin} data-tooltip="Pin selection">
                  <PinIcon />
                </button>
              </div>

              <div className="ai-answer">
                {(turn.answer || '').split('\n').map((line, i) => (
                  <p key={i}>{highlightCitations(line)}</p>
                ))}
              </div>

              {turn.citations?.length > 0 && (
                <>
                  <CitationList citations={turn.citations} />
                  <button
                    className="evidence-toggle"
                    onClick={e => { e.stopPropagation(); setShowEvidence(v => !v); }}
                  >
                    <ChevronIcon open={showEvidence} />
                    {showEvidence ? 'Hide' : 'View'} Evidence ({turn.citations.length} sources)
                  </button>
                  {showEvidence && (
                    <div className="evidence-preview">
                      {turn.citations.map((c, i) => (
                        <div key={i} className="ev-chip">
                          <span className="ev-id">[{c.evidence_id}]</span>
                          <span className="ev-preview">{c.text_preview}</span>
                          <span className="ev-page">p.{c.pages}</span>
                        </div>
                      ))}
                    </div>
                  )}
                </>
              )}

              {turn.citation_coverage != null && turn.verdict !== 'refused' && (
                <div className="coverage-bar-row">
                  <span className="conf-label">Citation coverage</span>
                  <div className="progress-bar" style={{ flex: 1 }}>
                    <div className="progress-fill" style={{ width: `${Math.round(turn.citation_coverage * 100)}%` }} />
                  </div>
                  <span className="conf-pct">{Math.round(turn.citation_coverage * 100)}%</span>
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}

function highlightCitations(text) {
  const parts = text.split(/(\[E\d+\])/g);
  return parts.map((p, i) =>
    /\[E\d+\]/.test(p)
      ? <span key={i} className="inline-cite">{p}</span>
      : p
  );
}

function EmptyState({ onUpload, uploadLoading }) {
  const fileRef = useRef(null);
  return (
    <div className="empty-state">
      <div className="empty-icon">
        <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="var(--primary)" strokeWidth="1.2">
          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
          <polyline points="14 2 14 8 20 8" />
          <line x1="12" y1="11" x2="12" y2="17" /><line x1="9" y1="14" x2="15" y2="14" />
        </svg>
      </div>
      <h2 className="empty-title">Upload a research paper to begin</h2>
      <p className="empty-sub">
        Upload any academic PDF and ask questions. Answers are grounded in evidence with inline citations and hallucination detection.
      </p>
      <input ref={fileRef} type="file" accept=".pdf" style={{ display: 'none' }}
        onChange={e => { const f = e.target.files?.[0]; if (f) onUpload(f); e.target.value = ''; }} />
      <button className="btn btn-primary empty-upload" onClick={() => fileRef.current?.click()} disabled={uploadLoading}>
        {uploadLoading ? <><span className="spinner" /> Indexing…</> : <><UploadIcon /> Upload PDF</>}
      </button>
      <p className="empty-hint">Supports multi-document indexing • Conversation memory • Citation tracking</p>
    </div>
  );
}

export default function ChatArea({
  sessionName, turns, indexStatus, activeDocs, isLoading, error,
  onAsk, onTurnClick, onPin, onClearSession, onUpload, uploadLoading,
}) {
  const [question, setQuestion] = useState('');
  const [useMemory, setUseMemory] = useState(true);
  const [enableNli, setEnableNli] = useState(false);
  const fileRef = useRef(null);
  const bottomRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [turns.length, isLoading]);

  const submit = () => {
    const q = question.trim();
    if (!q || isLoading) return;
    setQuestion('');
    onAsk(q, { useMemory, enableNli });
  };

  const handleKey = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); submit(); }
  };

  const hasIndex = indexStatus.chunk_count > 0;

  return (
    <div className="chat-area">
      {/* Top Bar */}
      <div className="chat-topbar">
        <div className="chat-title-block">
          <h1 className="chat-session-name">{sessionName}</h1>
          <div className="chat-badges">
            {activeDocs.length > 0
              ? activeDocs.map((d, i) => (
                  <span key={i} className="badge badge-muted">{d.filename.replace('.pdf', '')}</span>
                ))
              : <span className="badge badge-muted">No papers loaded</span>}
            {hasIndex && <span className="badge badge-primary">{indexStatus.chunk_count} chunks</span>}
          </div>
        </div>
        <div className="chat-actions">
          <label className="toggle-label">
            <input type="checkbox" checked={useMemory} onChange={e => setUseMemory(e.target.checked)} />
            <span>Memory</span>
          </label>
          <label className="toggle-label">
            <input type="checkbox" checked={enableNli} onChange={e => setEnableNli(e.target.checked)} />
            <span>NLI</span>
          </label>
          {turns.length > 0 && (
            <button className="btn btn-ghost btn-sm" onClick={onClearSession}>Clear</button>
          )}
        </div>
      </div>

      {/* Error Banner */}
      {error && (
        <div className="error-banner fade-up">⚠ {error}</div>
      )}

      {/* Messages */}
      <div className="chat-messages">
        {!hasIndex && turns.length === 0 ? (
          <EmptyState onUpload={onUpload} uploadLoading={uploadLoading} />
        ) : (
          <>
            {turns.map((turn) => (
              <MessageBubble
                key={turn.id}
                turn={turn}
                onTurnClick={onTurnClick}
                onPin={onPin}
              />
            ))}
            {isLoading && turns[turns.length - 1]?.verdict !== 'loading' && (
              <div className="typing-indicator fade-up">
                <div className="ai-avatar"><BotIcon /></div>
                <div className="typing-dots">
                  <span /><span /><span />
                </div>
              </div>
            )}
            <div ref={bottomRef} />
          </>
        )}
      </div>

      {/* Input Bar */}
      <div className="chat-input-bar">
        <input
          ref={fileRef} type="file" accept=".pdf" style={{ display: 'none' }}
          onChange={e => { const f = e.target.files?.[0]; if (f) onUpload(f); e.target.value = ''; }}
        />
        <button className="input-icon-btn" onClick={() => fileRef.current?.click()}
          disabled={uploadLoading} data-tooltip="Upload PDF">
          {uploadLoading ? <span className="spinner" /> : <UploadIcon />}
        </button>
        <textarea
          ref={inputRef}
          className="chat-input"
          placeholder={hasIndex ? 'Ask a question about your papers…' : 'Upload a paper first, then ask questions…'}
          value={question}
          onChange={e => setQuestion(e.target.value)}
          onKeyDown={handleKey}
          rows={1}
          disabled={!hasIndex}
        />
        <button
          className="btn btn-primary send-btn"
          onClick={submit}
          disabled={!question.trim() || isLoading || !hasIndex}
        >
          {isLoading ? <span className="spinner" /> : <SendIcon />}
        </button>
      </div>
    </div>
  );
}
