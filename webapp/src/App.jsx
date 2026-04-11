import { useCallback, useEffect, useState } from 'react';
import { api } from './api';
import Sidebar from './components/Sidebar';
import ChatArea from './components/ChatArea';
import ContextPanel from './components/ContextPanel';
import './App.css';

function AuthScreen({ mode, loading, error, onSubmit, onToggleMode }) {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');

  const submit = async (e) => {
    e.preventDefault();
    await onSubmit(username, password);
  };

  return (
    <div className="auth-shell">
      <div className="auth-panel fade-up">
        <div className="auth-copy">
          <span className="auth-kicker">Private Research Workspace</span>
          <h1>Each login gets its own papers, answers, and session history.</h1>
          <p>
            Sign in to keep uploads and retrieval results isolated per user. No more shared default session.
          </p>
        </div>

        <form className="auth-form" onSubmit={submit}>
          <label className="auth-label">
            <span>Username</span>
            <input
              className="input-field"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder="researcher01"
              autoComplete="username"
              required
            />
          </label>

          <label className="auth-label">
            <span>Password</span>
            <input
              className="input-field"
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="At least 6 characters"
              autoComplete={mode === 'login' ? 'current-password' : 'new-password'}
              required
            />
          </label>

          {error && <div className="auth-error">{error}</div>}

          <button className="btn btn-primary auth-submit" type="submit" disabled={loading}>
            {loading ? <span className="spinner" /> : null}
            {mode === 'login' ? 'Log In' : 'Create Account'}
          </button>
        </form>

        <button className="auth-switch" type="button" onClick={onToggleMode}>
          {mode === 'login' ? 'Need an account? Register' : 'Already have an account? Log in'}
        </button>
      </div>
    </div>
  );
}

export default function App() {
  const [user, setUser] = useState(null);
  const [authMode, setAuthMode] = useState('login');
  const [authLoading, setAuthLoading] = useState(true);
  const [authError, setAuthError] = useState(null);

  const [sessionId, setSessionId] = useState(null);
  const [sessionName, setSessionName] = useState('Research Session');
  const [sessions, setSessions] = useState([]);
  const [turns, setTurns] = useState([]);
  const [pins, setPins] = useState([]);
  const [activeDocs, setActiveDocs] = useState([]);
  const [indexStatus, setIndexStatus] = useState({ chunk_count: 0, document_count: 0 });
  const [activeEvidence, setActiveEvidence] = useState(null);
  const [activeTurn, setActiveTurn] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [uploadLoading, setUploadLoading] = useState(false);
  const [error, setError] = useState(null);

  const resetWorkspace = useCallback(() => {
    setSessionId(null);
    setSessionName('Research Session');
    setSessions([]);
    setTurns([]);
    setPins([]);
    setActiveDocs([]);
    setIndexStatus({ chunk_count: 0, document_count: 0 });
    setActiveEvidence(null);
    setActiveTurn(null);
    setError(null);
    setIsLoading(false);
    setUploadLoading(false);
  }, []);

  const refreshSession = useCallback(async (sid) => {
    if (!sid) return;
    const data = await api.getSession(sid);
    setTurns(data.turns || []);
    setPins(data.pins || []);
    setActiveDocs(data.active_docs || []);
    setSessionName(data.session_name || 'Research Session');
  }, []);

  const refreshIndex = useCallback(async () => {
    const status = await api.indexStatus();
    setIndexStatus(status);
  }, []);

  const refreshSessions = useCallback(async () => {
    const list = await api.listSessions();
    setSessions(list);
    return list;
  }, []);

  const initializeWorkspace = useCallback(async () => {
    const [status, existingSessions] = await Promise.all([
      api.indexStatus(),
      api.listSessions(),
    ]);

    setIndexStatus(status);
    setSessions(existingSessions);

    let active = existingSessions[0];
    if (!active) {
      active = await api.createSession('Research Session');
      setSessions([{ ...active, turn_count: 0, pin_count: 0, doc_count: 0 }]);
    }

    setSessionId(active.session_id);
    setSessionName(active.session_name || 'Research Session');
    await refreshSession(active.session_id);
  }, [refreshSession]);

  useEffect(() => {
    const token = api.getToken();
    if (!token) {
      setAuthLoading(false);
      return;
    }

    let cancelled = false;

    const restore = async () => {
      try {
        const data = await api.me();
        if (cancelled) return;
        setUser(data.user);
        await initializeWorkspace();
      } catch (e) {
        if (cancelled) return;
        api.setToken('');
        setUser(null);
        resetWorkspace();
        setAuthError(e.message);
      } finally {
        if (!cancelled) {
          setAuthLoading(false);
        }
      }
    };

    restore();
    return () => {
      cancelled = true;
    };
  }, [initializeWorkspace, resetWorkspace]);

  const handleAuth = async (username, password) => {
    setAuthLoading(true);
    setAuthError(null);

    try {
      const data = authMode === 'login'
        ? await api.login(username, password)
        : await api.register(username, password);

      api.setToken(data.token);
      setUser(data.user);
      resetWorkspace();
      await initializeWorkspace();
    } catch (e) {
      setAuthError(e.message);
    } finally {
      setAuthLoading(false);
    }
  };

  const handleLogout = () => {
    api.setToken('');
    setUser(null);
    setAuthError(null);
    resetWorkspace();
  };

  const handleNewSession = async () => {
    const name = `Session ${new Date().toLocaleTimeString()}`;
    try {
      const data = await api.createSession(name);
      setSessionId(data.session_id);
      setSessionName(data.session_name || name);
      setTurns([]);
      setPins([]);
      setActiveDocs([]);
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
    try {
      await refreshSession(sid);
    } catch (e) {
      setError(e.message);
    }
  };

  const handleUpload = async (file) => {
    if (!sessionId) return;
    setUploadLoading(true);
    setError(null);
    try {
      await api.uploadPdf(file, sessionId);
      await refreshIndex();
      await refreshSession(sessionId);
      await refreshSessions();
    } catch (e) {
      setError(e.message);
    } finally {
      setUploadLoading(false);
    }
  };

  const handleAsk = async (question, opts = {}) => {
    if (!sessionId) return;
    setIsLoading(true);
    setError(null);

    const tempId = `temp-${Date.now()}`;
    const userTurn = {
      id: tempId,
      question,
      answer: null,
      verdict: 'loading',
      timestamp: new Date().toLocaleTimeString(),
    };
    setTurns((prev) => [...prev, userTurn]);

    try {
      const result = await api.ask(question, sessionId, opts);
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

      setTurns((prev) => prev.map((turn) => (turn.id === tempId ? newTurn : turn)));

      if (result.evidence?.length) {
        setActiveEvidence({ evidence: result.evidence, turn: newTurn });
        setActiveTurn(newTurn);
      }

      await refreshSessions();
    } catch (e) {
      setError(e.message);
      setTurns((prev) => prev.filter((turn) => turn.id !== tempId));
    } finally {
      setIsLoading(false);
    }
  };

  const handleTurnClick = (turn) => {
    setActiveTurn(turn);
    if (turn.citations?.length) {
      setActiveEvidence({ citations: turn.citations, turn });
    }
  };

  const handlePin = async (text, sourceQuestion, fromDoc) => {
    if (!sessionId) return;
    try {
      const data = await api.addPin(sessionId, text, sourceQuestion, fromDoc);
      setPins((prev) => [...prev, data.pin]);
      await refreshSessions();
    } catch (e) {
      setError(e.message);
    }
  };

  const handleUnpin = async (pinId) => {
    if (!sessionId) return;
    try {
      await api.removePin(sessionId, pinId);
      setPins((prev) => prev.filter((pin) => pin.id !== pinId));
      await refreshSessions();
    } catch (e) {
      setError(e.message);
    }
  };

  const handleClearIndex = async () => {
    try {
      await api.clearIndex();
      setActiveDocs([]);
      await refreshIndex();
      if (sessionId) {
        await refreshSession(sessionId);
      }
      await refreshSessions();
    } catch (e) {
      setError(e.message);
    }
  };

  const handleClearSession = async () => {
    if (!sessionId) return;
    try {
      await api.clearSession(sessionId);
      setTurns([]);
      setPins([]);
      setActiveDocs([]);
      setActiveEvidence(null);
      setActiveTurn(null);
      await refreshSessions();
    } catch (e) {
      setError(e.message);
    }
  };

  if (!user) {
    return (
      <AuthScreen
        mode={authMode}
        loading={authLoading}
        error={authError}
        onSubmit={handleAuth}
        onToggleMode={() => {
          setAuthMode((prev) => (prev === 'login' ? 'register' : 'login'));
          setAuthError(null);
        }}
      />
    );
  }

  return (
    <div className="app-shell">
      <Sidebar
        user={user}
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
        onLogout={handleLogout}
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
