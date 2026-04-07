import { useState, useEffect, useRef, useCallback } from 'react';
import { api } from './api';
import Sidebar from './components/Sidebar';
import ChatArea from './components/ChatArea';
import ContextPanel from './components/ContextPanel';
import './App.css';

export default function App() {
  const [sessionId, setSessionId] = useState('default');
  const [sessionName, setSessionName] = useState('Research Session');
  const [sessions, setSessions] = useState([]);
  const [turns, setTurns] = useState([]);
  const [pins, setPins] = useState([]);
  const [activeDocs, setActiveDocs] = useState([]);
  const [indexStatus, setIndexStatus] = useState({ chunk_count: 0, document_count: 0 });
  const [activeEvidence, setActiveEvidence] = useState(null); // evidence for context panel
  const [activeTurn, setActiveTurn] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [uploadLoading, setUploadLoading] = useState(false);
  const [error, setError] = useState(null);
  const pollRef = useRef(null);

  const refreshSession = useCallback(async (sid) => {
    try {
      const data = await api.getSession(sid);
      setTurns(data.turns || []);
      setPins(data.pins || []);
      setActiveDocs(data.active_docs || []);
    } catch {}
  }, []);

  const refreshIndex = useCallback(async () => {
    try {
      const status = await api.indexStatus();
      setIndexStatus(status);
    } catch {}
  }, []);

  const refreshSessions = useCallback(async () => {
    try {
      const list = await api.listSessions();
      setSessions(list);
    } catch {}
  }, []);

  useEffect(() => {
    refreshIndex();
    refreshSessions();
  }, []);

  useEffect(() => {
    refreshSession(sessionId);
  }, [sessionId]);

  const handleNewSession = async () => {
    const name = `Session ${new Date().toLocaleTimeString()}`;
    try {
      const data = await api.createSession(name);
      setSessionId(data.session_id);
      setSessionName(data.session_name);
      setTurns([]);
      setPins([]);
      setActiveEvidence(null);
      setActiveTurn(null);
      await refreshSessions();
    } catch (e) {
      setError(e.message);
    }
  };

  const handleSelectSession = async (sid, sname) => {
    setSessionId(sid);
    setSessionName(sname);
    setActiveEvidence(null);
    setActiveTurn(null);
    await refreshSession(sid);
  };

  const handleUpload = async (file) => {
    setUploadLoading(true);
    setError(null);
    try {
      await api.uploadPdf(file, sessionId);
      await refreshIndex();
      await refreshSession(sessionId);
    } catch (e) {
      setError(e.message);
    } finally {
      setUploadLoading(false);
    }
  };

  const handleAsk = async (question, opts = {}) => {
    setIsLoading(true);
    setError(null);
    // Optimistically add user turn
    const tempId = `temp-${Date.now()}`;
    const userTurn = { id: tempId, question, answer: null, verdict: 'loading', timestamp: new Date().toLocaleTimeString() };
    setTurns(prev => [...prev, userTurn]);
    try {
      const result = await api.ask(question, sessionId, opts);
      // Replace temp turn with real result
      const newTurn = {
        id: result.id || tempId,
        question,
        answer: result.answer,
        verdict: result.verdict,
        confidence: result.confidence,
        citations: result.citations || [],
        citation_coverage: result.citation_coverage,
        support_ratio: result.support_ratio,
        timestamp: new Date().toLocaleTimeString(),
        rewritten_question: result.rewritten_question,
      };
      setTurns(prev => prev.map(t => t.id === tempId ? newTurn : t));
      // Update context panel with evidence from this turn
      if (result.evidence?.length) {
        setActiveEvidence({ evidence: result.evidence, turn: newTurn });
        setActiveTurn(newTurn);
      }
      await refreshSessions();
    } catch (e) {
      setError(e.message);
      setTurns(prev => prev.filter(t => t.id !== tempId));
    } finally {
      setIsLoading(false);
    }
  };

  const handleTurnClick = (turn) => {
    setActiveTurn(turn);
    // Show citations as context if no raw evidence
    if (turn.citations?.length) {
      setActiveEvidence({ citations: turn.citations, turn });
    }
  };

  const handlePin = async (text, sourceQuestion, fromDoc) => {
    try {
      const data = await api.addPin(sessionId, text, sourceQuestion, fromDoc);
      setPins(prev => [...prev, data.pin]);
    } catch (e) {
      setError(e.message);
    }
  };

  const handleUnpin = async (pinId) => {
    try {
      await api.removePin(sessionId, pinId);
      setPins(prev => prev.filter(p => p.id !== pinId));
    } catch (e) {
      setError(e.message);
    }
  };

  const handleClearIndex = async () => {
    try {
      await api.clearIndex();
      setActiveDocs([]);
      await refreshIndex();
    } catch (e) {
      setError(e.message);
    }
  };

  const handleClearSession = async () => {
    try {
      await api.clearSession(sessionId);
      setTurns([]);
      setPins([]);
      setActiveEvidence(null);
      setActiveTurn(null);
    } catch (e) {
      setError(e.message);
    }
  };

  return (
    <div className="app-shell">
      <Sidebar
        sessions={sessions}
        currentSessionId={sessionId}
        sessionName={sessionName}
        activeDocs={activeDocs}
        pins={pins}
        indexStatus={indexStatus}
        onNewSession={handleNewSession}
        onSelectSession={handleSelectSession}
        onUpload={handleUpload}
        onUnpin={handleUnpin}
        onClearIndex={handleClearIndex}
        uploadLoading={uploadLoading}
      />

      <ChatArea
        sessionName={sessionName}
        turns={turns}
        indexStatus={indexStatus}
        activeDocs={activeDocs}
        isLoading={isLoading}
        error={error}
        onAsk={handleAsk}
        onTurnClick={handleTurnClick}
        onPin={handlePin}
        onClearSession={handleClearSession}
        onUpload={handleUpload}
        uploadLoading={uploadLoading}
      />

      <ContextPanel
        activeEvidence={activeEvidence}
        activeTurn={activeTurn}
        pins={pins}
        onPin={handlePin}
        onUnpin={handleUnpin}
      />
    </div>
  );
}
