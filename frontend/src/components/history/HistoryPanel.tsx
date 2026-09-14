import React, { useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  AlertTriangle,
  Check,
  ChevronRight,
  ListFilter,
  MoreHorizontal,
  Pencil,
  SquarePen,
  Trash2,
  X,
} from 'lucide-react';
import { useApp } from '../../context/AppState';
import type { Session } from '../../lib/types';
import { cn } from '../../lib/utils';

/* -------------------------------------------------------------------------- */
/* Grouping                                                                    */
/* -------------------------------------------------------------------------- */

const DAY_MS = 86_400_000;
/** Older conversations collapse behind a "Show more" so the rail stays short. */
const COLLAPSED_LIMIT = 20;

function bucketFor(iso: string): string {
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return 'Older';

  const now = new Date();
  const startOfToday = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();

  if (then >= startOfToday) return 'Today';
  if (then >= startOfToday - DAY_MS) return 'Yesterday';
  if (then >= startOfToday - 7 * DAY_MS) {
    return new Date(then).toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
  }
  return 'Older';
}

/** Sessions arrive newest-first, so first-seen order gives the bucket order. */
function groupByDate(sessions: Session[]): Array<[string, Session[]]> {
  const groups = new Map<string, Session[]>();
  for (const session of sessions) {
    const key = bucketFor(session.startedAt);
    const bucket = groups.get(key);
    if (bucket) bucket.push(session);
    else groups.set(key, [session]);
  }
  return Array.from(groups.entries());
}

function matches(session: Session, needle: string): boolean {
  if (!needle) return true;
  const q = needle.toLowerCase();
  return (
    session.title.toLowerCase().includes(q) ||
    Boolean(session.datasetName?.toLowerCase().includes(q)) ||
    session.messages.some((message) => message.content.toLowerCase().includes(q))
  );
}

/* -------------------------------------------------------------------------- */
/* Row                                                                         */
/* -------------------------------------------------------------------------- */

interface RowProps {
  title: string;
  active: boolean;
  onOpen: () => void;
  /** Omitted for the live conversation, which cannot be renamed or deleted. */
  sessionId?: string;
}

const Row: React.FC<RowProps> = ({ title, active, onOpen, sessionId }) => {
  const { renameSession, deleteSession } = useApp();
  const [menuOpen, setMenuOpen] = useState(false);
  const [editing, setEditing] = useState(false);
  const [confirming, setConfirming] = useState(false);
  const [draft, setDraft] = useState(title);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (editing) inputRef.current?.select();
  }, [editing]);

  useEffect(() => {
    if (!menuOpen) return;
    const close = () => setMenuOpen(false);
    window.addEventListener('click', close);
    return () => window.removeEventListener('click', close);
  }, [menuOpen]);

  const commit = () => {
    if (sessionId) renameSession(sessionId, draft);
    setEditing(false);
  };

  if (editing) {
    return (
      <div className="flex items-center gap-1 rounded-lg border border-accent/45 bg-surface-2 px-2 py-1">
        <input
          ref={inputRef}
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === 'Enter') commit();
            if (event.key === 'Escape') {
              setDraft(title);
              setEditing(false);
            }
          }}
          className="min-w-0 flex-1 bg-transparent py-1 text-[13px] text-ink outline-none"
          aria-label="Conversation title"
        />
        <button
          type="button"
          aria-label="Save title"
          onClick={commit}
          className="cursor-pointer rounded p-1 text-accent hover:bg-surface-3"
        >
          <Check size={12} />
        </button>
      </div>
    );
  }

  return (
    <div
      className={cn(
        'group relative flex items-center rounded-lg transition-colors',
        active ? 'bg-surface-3' : 'hover:bg-surface-2'
      )}
    >
      <button
        type="button"
        onClick={onOpen}
        className="flex min-w-0 flex-1 cursor-pointer items-center gap-2.5 px-2.5 py-[7px] text-left"
      >
        <span
          className={cn(
            'h-1.5 w-1.5 shrink-0 rounded-full border transition-colors',
            active ? 'border-accent bg-accent' : 'border-ink-faint bg-transparent'
          )}
          aria-hidden
        />
        <span
          className={cn(
            'truncate text-[13px] leading-snug',
            active ? 'text-ink' : 'text-ink-muted group-hover:text-ink'
          )}
          title={title}
        >
          {title}
        </span>
      </button>

      {sessionId && (
        <div className="relative pr-1">
          <button
            type="button"
            aria-label="Conversation options"
            onClick={(event) => {
              event.stopPropagation();
              setMenuOpen((open) => !open);
              setConfirming(false);
            }}
            className={cn(
              'flex h-6 w-6 cursor-pointer items-center justify-center rounded text-ink-faint transition-all hover:bg-surface-4 hover:text-ink',
              menuOpen ? 'opacity-100' : 'opacity-0 group-hover:opacity-100 focus-visible:opacity-100'
            )}
          >
            <MoreHorizontal size={14} />
          </button>

          {menuOpen && (
            <div
              onClick={(event) => event.stopPropagation()}
              className="absolute right-0 top-[calc(100%+4px)] z-40 w-36 overflow-hidden rounded-lg border border-line-strong bg-surface-2 py-1 shadow-2xl"
            >
              <button
                type="button"
                onClick={() => {
                  setDraft(title);
                  setEditing(true);
                  setMenuOpen(false);
                }}
                className="flex w-full cursor-pointer items-center gap-2 px-3 py-1.5 text-[12.5px] text-ink-muted transition-colors hover:bg-surface-3 hover:text-ink"
              >
                <Pencil size={12} aria-hidden />
                Rename
              </button>
              {confirming ? (
                <button
                  type="button"
                  onClick={() => deleteSession(sessionId)}
                  className="flex w-full cursor-pointer items-center gap-2 px-3 py-1.5 text-[12.5px] text-danger transition-colors hover:bg-danger/10"
                >
                  <AlertTriangle size={12} aria-hidden />
                  Confirm
                </button>
              ) : (
                <button
                  type="button"
                  onClick={() => setConfirming(true)}
                  className="flex w-full cursor-pointer items-center gap-2 px-3 py-1.5 text-[12.5px] text-ink-muted transition-colors hover:bg-surface-3 hover:text-danger"
                >
                  <Trash2 size={12} aria-hidden />
                  Delete
                </button>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

/* -------------------------------------------------------------------------- */
/* Group                                                                       */
/* -------------------------------------------------------------------------- */

interface GroupProps {
  label: string;
  sessions: Session[];
  activeId: string | null;
  onOpen: (session: Session) => void;
  headerAction?: React.ReactNode;
}

const Group: React.FC<GroupProps> = ({ label, sessions, activeId, onOpen, headerAction }) => {
  const [collapsed, setCollapsed] = useState(false);
  const [expanded, setExpanded] = useState(false);

  const limit = expanded ? sessions.length : COLLAPSED_LIMIT;
  const visible = sessions.slice(0, limit);
  const hidden = sessions.length - visible.length;

  return (
    <section className="mb-2">
      <div className="flex items-center justify-between gap-1 pr-1">
        <button
          type="button"
          onClick={() => setCollapsed((value) => !value)}
          aria-expanded={!collapsed}
          className="group flex cursor-pointer items-center gap-1 rounded px-2 py-1.5 text-ink-faint transition-colors hover:text-ink-muted"
        >
          <span className="label-caps">{label}</span>
          <ChevronRight
            size={11}
            className={cn(
              'opacity-0 transition-all group-hover:opacity-100',
              !collapsed && 'rotate-90'
            )}
            aria-hidden
          />
        </button>
        {headerAction}
      </div>

      {!collapsed && (
        <div className="space-y-px">
          {visible.map((session) => (
            <Row
              key={session.id}
              sessionId={session.id}
              title={session.title}
              active={activeId === session.id}
              onOpen={() => onOpen(session)}
            />
          ))}

          {hidden > 0 && (
            <button
              type="button"
              onClick={() => setExpanded(true)}
              className="w-full cursor-pointer rounded-lg px-2.5 py-1.5 text-left text-[12.5px] text-ink-faint transition-colors hover:bg-surface-2 hover:text-ink-muted"
            >
              Show {hidden} more
            </button>
          )}
        </div>
      )}
    </section>
  );
};

/* -------------------------------------------------------------------------- */
/* Panel                                                                       */
/* -------------------------------------------------------------------------- */

const CURRENT_ID = '__current__';

/**
 * Conversation rail that sits beside the workspace rather than replacing it —
 * picking a conversation loads it into the thread on the right, so the list
 * and the transcript are visible at once.
 */
export const HistoryPanel: React.FC = () => {
  const navigate = useNavigate();
  const { sessions, restoreSession, resetConversation, messages, setHistoryOpen } = useApp();

  const [query, setQuery] = useState('');
  const [searchOpen, setSearchOpen] = useState(false);
  const [activeId, setActiveId] = useState<string>(CURRENT_ID);
  const searchRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (searchOpen) searchRef.current?.focus();
  }, [searchOpen]);

  const currentTitle = useMemo(() => {
    const firstUser = messages.find((message) => message.role === 'user');
    return firstUser?.content.slice(0, 120) ?? null;
  }, [messages]);

  const filtered = useMemo(
    () => sessions.filter((session) => matches(session, query.trim())),
    [sessions, query]
  );
  const grouped = useMemo(() => groupByDate(filtered), [filtered]);

  const openSession = (session: Session) => {
    restoreSession(session.id);
    setActiveId(session.id);
    navigate('/console');
  };

  const startNew = () => {
    resetConversation();
    setActiveId(CURRENT_ID);
    navigate('/console');
  };

  const showCurrent =
    currentTitle !== null && (!query.trim() || currentTitle.toLowerCase().includes(query.toLowerCase()));

  return (
    <aside
      className="flex w-[268px] shrink-0 flex-col border-r border-line bg-surface"
      aria-label="Conversation history"
    >
      <header className="flex h-11 shrink-0 items-center justify-between gap-1 border-b border-line px-2">
        <button
          type="button"
          onClick={startNew}
          className="flex min-w-0 flex-1 cursor-pointer items-center gap-2 rounded-lg px-2 py-1.5 text-[13px] text-ink transition-colors hover:bg-surface-2"
        >
          <SquarePen size={14} className="shrink-0 text-accent" aria-hidden />
          New conversation
        </button>

        <button
          type="button"
          aria-label={searchOpen ? 'Hide search' : 'Search conversations'}
          aria-pressed={searchOpen}
          onClick={() => {
            setSearchOpen((open) => !open);
            if (searchOpen) setQuery('');
          }}
          className={cn(
            'flex h-7 w-7 shrink-0 cursor-pointer items-center justify-center rounded-lg transition-colors',
            searchOpen ? 'bg-accent/15 text-accent' : 'text-ink-faint hover:bg-surface-2 hover:text-ink'
          )}
        >
          <ListFilter size={14} />
        </button>

        <button
          type="button"
          aria-label="Close history"
          onClick={() => setHistoryOpen(false)}
          className="flex h-7 w-7 shrink-0 cursor-pointer items-center justify-center rounded-lg text-ink-faint transition-colors hover:bg-surface-2 hover:text-ink"
        >
          <X size={14} />
        </button>
      </header>

      {searchOpen && (
        <div className="border-b border-line p-2">
          <input
            ref={searchRef}
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search conversations"
            aria-label="Search conversations"
            className="h-8 w-full rounded-lg border border-line bg-surface-2 px-2.5 text-[13px] text-ink outline-none transition-colors placeholder:text-ink-faint focus:border-accent/45"
          />
        </div>
      )}

      <div className="scrollbar-slim min-h-0 flex-1 overflow-y-auto px-2 py-2">
        {showCurrent && currentTitle && (
          <section className="mb-2">
            <div className="px-2 py-1.5">
              <span className="label-caps text-ink-faint">Current</span>
            </div>
            <Row
              title={currentTitle}
              active={activeId === CURRENT_ID}
              onOpen={() => {
                setActiveId(CURRENT_ID);
                navigate('/console');
              }}
            />
          </section>
        )}

        {grouped.map(([label, bucket]) => (
          <Group
            key={label}
            label={label}
            sessions={bucket}
            activeId={activeId}
            onOpen={openSession}
          />
        ))}

        {grouped.length === 0 && !showCurrent && (
          <p className="px-2 py-8 text-center text-[12.5px] leading-relaxed text-ink-faint">
            {query.trim()
              ? `Nothing matches “${query.trim()}”.`
              : 'No conversations yet. Ask something and it will show up here.'}
          </p>
        )}
      </div>
    </aside>
  );
};

export default HistoryPanel;
