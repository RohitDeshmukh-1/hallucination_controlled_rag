import { useState, useRef } from 'react';
import './Sidebar.css';

const AtomIcon = () => (
  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
    <circle cx="12" cy="12" r="2" fill="currentColor" />
    <ellipse cx="12" cy="12" rx="10" ry="4" />
    <ellipse cx="12" cy="12" rx="10" ry="4" transform="rotate(60 12 12)" />
    <ellipse cx="12" cy="12" rx="10" ry="4" transform="rotate(120 12 12)" />
  </svg>
);

const DocIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
    <polyline points="14 2 14 8 20 8" />
  </svg>
);

const PinIcon = () => (
  <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor">
    <path d="M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7zm0 9.5c-1.38 0-2.5-1.12-2.5-2.5s1.12-2.5 2.5-2.5 2.5 1.12 2.5 2.5-1.12 2.5-2.5 2.5z" />
  </svg>
);

const HistoryIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <polyline points="1 4 1 10 7 10" />
    <path d="M3.51 15a9 9 0 1 0 .49-4.51" />
  </svg>
);

const PlusIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
    <line x1="12" y1="5" x2="12" y2="19" /><line x1="5" y1="12" x2="19" y2="12" />
  </svg>
);

const TrashIcon = () => (
  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <polyline points="3 6 5 6 21 6" /><path d="M19 6l-1 14H6L5 6" /><path d="M10 11v6M14 11v6" />
    <path d="M9 6V4h6v2" />
  </svg>
);

export default function Sidebar({
  user,
  sessions, currentSessionId, sessionName, activeDocs,
  pins, indexStatus, onNewSession, onSelectSession,
  onUpload, onUnpin, onClearIndex, onLogout, uploadLoading,
}) {
  const fileRef = useRef(null);
  const [expandPins, setExpandPins] = useState(true);
  const [expandDocs, setExpandDocs] = useState(true);
  const [expandHistory, setExpandHistory] = useState(false);

  const handleFile = (e) => {
    const f = e.target.files?.[0];
    if (f) onUpload(f);
    e.target.value = '';
  };

  return (
    <aside className="sidebar">
      {/* Logo */}
      <div className="sidebar-logo">
        <span className="logo-icon"><AtomIcon /></span>
        <div className="logo-copy">
          <span className="logo-text">ResearchMind</span>
          <span className="logo-user">@{user?.username}</span>
        </div>
      </div>

      {/* Upload Button */}
      <div className="sidebar-section">
        <input ref={fileRef} type="file" accept=".pdf" onChange={handleFile} style={{ display: 'none' }} />
        <button className="btn btn-primary sidebar-upload-btn" onClick={() => fileRef.current?.click()} disabled={uploadLoading}>
          {uploadLoading ? <span className="spinner" /> : <PlusIcon />}
          {uploadLoading ? 'Indexing…' : 'Upload Paper'}
        </button>
      </div>

      {/* Index Stats Chips */}
      <div className="sidebar-chips">
        <span className="badge badge-primary">{indexStatus.chunk_count} chunks</span>
        <span className="badge badge-muted">{indexStatus.document_count} docs</span>
      </div>

      <div className="sidebar-scroll">
        {/* Library — uploaded docs */}
        <div className="sidebar-section-header" onClick={() => setExpandDocs(v => !v)}>
          <DocIcon />
          <span>Library</span>
          <span className="chevron">{expandDocs ? '▾' : '▸'}</span>
        </div>
        {expandDocs && (
          <div className="sidebar-items">
            {activeDocs.length === 0 ? (
              <p className="sidebar-empty">No papers indexed yet</p>
            ) : (
              activeDocs.map((d, i) => (
                <div key={i} className="doc-card">
                  <DocIcon />
                  <div className="doc-info">
                    <p className="doc-name truncate">{d.filename}</p>
                    <p className="doc-meta">{d.chunk_count} chunks · {d.indexed_at}</p>
                  </div>
                </div>
              ))
            )}
            {activeDocs.length > 0 && (
              <button className="btn btn-danger btn-sm" onClick={onClearIndex}>
                <TrashIcon /> Clear Index
              </button>
            )}
          </div>
        )}

        {/* Memory Pins */}
        <div className="sidebar-section-header" onClick={() => setExpandPins(v => !v)}>
          <PinIcon />
          <span>Memory Pins</span>
          <span className="pin-count">{pins.length}</span>
          <span className="chevron">{expandPins ? '▾' : '▸'}</span>
        </div>
        {expandPins && (
          <div className="sidebar-items">
            {pins.length === 0 ? (
              <p className="sidebar-empty">Pin key insights from answers</p>
            ) : (
              pins.map(pin => (
                <div key={pin.id} className="pin-card">
                  <p className="pin-text">{pin.text}</p>
                  <div className="pin-footer">
                    <span className="pin-source truncate">Q: {pin.source_question}</span>
                    <button className="pin-remove" onClick={() => onUnpin(pin.id)} title="Remove pin">×</button>
                  </div>
                </div>
              ))
            )}
          </div>
        )}

        {/* Session History */}
        <div className="sidebar-section-header" onClick={() => setExpandHistory(v => !v)}>
          <HistoryIcon />
          <span>Sessions</span>
          <span className="chevron">{expandHistory ? '▾' : '▸'}</span>
        </div>
        {expandHistory && (
          <div className="sidebar-items">
            {sessions.map(s => (
              <div
                key={s.session_id}
                className={`session-item ${s.session_id === currentSessionId ? 'active' : ''}`}
                onClick={() => onSelectSession(s.session_id, s.session_name)}
              >
                <p className="session-name truncate">{s.session_name}</p>
                <p className="session-meta">{s.turn_count} Q&amp;A · {s.doc_count} docs</p>
              </div>
            ))}
            {sessions.length === 0 && <p className="sidebar-empty">No past sessions</p>}
          </div>
        )}
      </div>

      {/* Bottom: New Session */}
      <div className="sidebar-footer">
        <button className="btn btn-ghost btn-full" onClick={onNewSession}>
          <PlusIcon /> New Session
        </button>
        <button className="btn btn-ghost btn-full" onClick={onLogout}>
          Sign Out
        </button>
      </div>
    </aside>
  );
}
