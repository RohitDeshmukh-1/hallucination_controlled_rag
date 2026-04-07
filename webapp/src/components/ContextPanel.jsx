import { useState } from 'react';
import './ContextPanel.css';

const ChevronIcon = ({ open }) => (
  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"
    style={{ transform: open ? 'rotate(180deg)' : 'rotate(0)', transition: '200ms' }}>
    <polyline points="6 9 12 15 18 9" />
  </svg>
);
const PinIcon = () => (
  <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor">
    <path d="M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7zm0 9.5c-1.38 0-2.5-1.12-2.5-2.5s1.12-2.5 2.5-2.5 2.5 1.12 2.5 2.5-1.12 2.5-2.5 2.5z"/>
  </svg>
);

function Section({ title, icon, children, defaultOpen = true }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="ctx-section">
      <button className="ctx-section-header" onClick={() => setOpen(v => !v)}>
        <span className="ctx-icon">{icon}</span>
        <span>{title}</span>
        <ChevronIcon open={open} />
      </button>
      {open && <div className="ctx-section-body">{children}</div>}
    </div>
  );
}

function EvidenceChunk({ chunk, idx }) {
  const [expanded, setExpanded] = useState(false);
  return (
    <div className="ev-card">
      <div className="ev-card-header">
        <span className="ev-badge">[E{idx + 1}]</span>
        <span className="ev-doc truncate">{chunk.doc_id}</span>
        <span className="ev-pages">p.{chunk.page_start}–{chunk.page_end}</span>
        {chunk.cross_score != null && (
          <span className="ev-score" title="Reranker score">
            {(chunk.cross_score * 100).toFixed(0)}%
          </span>
        )}
      </div>
      <p className={`ev-text ${expanded ? 'expanded' : ''}`}>{chunk.text}</p>
      {chunk.text?.length > 180 && (
        <button className="ev-expand" onClick={() => setExpanded(v => !v)}>
          {expanded ? 'Show less' : 'Show more'}
        </button>
      )}
    </div>
  );
}

function CitationCard({ citation, idx }) {
  return (
    <div className="ev-card">
      <div className="ev-card-header">
        <span className="ev-badge">[{citation.evidence_id}]</span>
        <span className="ev-doc truncate">{citation.doc_id}</span>
        <span className="ev-pages">p.{citation.pages}</span>
      </div>
      {citation.text_preview && (
        <p className="ev-text">{citation.text_preview}</p>
      )}
    </div>
  );
}

function VerificationBlock({ turn }) {
  if (!turn) return <p className="ctx-empty">Select a response to view verification</p>;

  const confidence = turn.confidence ?? 0;
  const coverage = turn.citation_coverage ?? 0;
  const support = turn.support_ratio ?? 0;

  const verdictColors = {
    supported: 'var(--success)',
    partially_supported: 'var(--warning)',
    refused: 'var(--error)',
  };

  return (
    <div className="verification-block">
      <div className="veri-row">
        <span className="veri-label">Verdict</span>
        <span className="veri-value" style={{ color: verdictColors[turn.verdict] || 'var(--text-muted)' }}>
          {turn.verdict?.replace(/_/g, ' ')}
        </span>
      </div>
      <div className="veri-metric">
        <div className="veri-metric-header">
          <span>Confidence</span><span>{Math.round(confidence * 100)}%</span>
        </div>
        <div className="progress-bar">
          <div className="progress-fill" style={{ width: `${confidence * 100}%`,
            background: confidence > 0.75 ? 'linear-gradient(90deg,#10b981,#34d399)'
              : confidence > 0.45 ? 'linear-gradient(90deg,#f59e0b,#fbbf24)'
              : 'linear-gradient(90deg,#ef4444,#f87171)' }} />
        </div>
      </div>
      <div className="veri-metric">
        <div className="veri-metric-header">
          <span>Support Ratio</span><span>{Math.round(support * 100)}%</span>
        </div>
        <div className="progress-bar">
          <div className="progress-fill" style={{ width: `${support * 100}%` }} />
        </div>
      </div>
      <div className="veri-metric">
        <div className="veri-metric-header">
          <span>Citation Coverage</span><span>{Math.round(coverage * 100)}%</span>
        </div>
        <div className="progress-bar">
          <div className="progress-fill" style={{ width: `${coverage * 100}%` }} />
        </div>
      </div>
    </div>
  );
}

export default function ContextPanel({ activeEvidence, activeTurn, pins, onPin, onUnpin }) {
  const evidence = activeEvidence?.evidence || [];
  const citations = activeEvidence?.citations || [];

  return (
    <aside className="context-panel">
      <div className="ctx-title">Context &amp; Evidence</div>

      {/* Evidence chunks (raw, when available) */}
      <Section title="Evidence Chunks" icon="⟨E⟩">
        {evidence.length > 0
          ? evidence.map((chunk, i) => <EvidenceChunk key={i} chunk={chunk} idx={i} />)
          : citations.length > 0
          ? citations.map((c, i) => <CitationCard key={i} citation={c} idx={i} />)
          : <p className="ctx-empty">Ask a question to see retrieved evidence</p>}
      </Section>

      {/* Verification metrics */}
      <Section title="Verification" icon="✓">
        <VerificationBlock turn={activeTurn} />
      </Section>

      {/* Memory Pins */}
      <Section title="Memory Pins" icon={<PinIcon />} defaultOpen={false}>
        {pins.length === 0
          ? <p className="ctx-empty">No pins yet. Select text in responses and click 📌</p>
          : pins.map(pin => (
              <div key={pin.id} className="ctx-pin-card">
                <p className="ctx-pin-text">{pin.text}</p>
                <div className="ctx-pin-meta">
                  <span className="ctx-pin-q truncate">"{pin.source_question}"</span>
                  <button className="pin-remove-btn" onClick={() => onUnpin(pin.id)}>×</button>
                </div>
              </div>
            ))}
      </Section>
    </aside>
  );
}
