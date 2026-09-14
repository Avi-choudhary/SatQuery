import React from 'react';
import { Info } from 'lucide-react';
import { Badge } from '../components/ui/Badge';
import { useApp } from '../context/AppState';

/**
 * There is no auth backend yet, so this shows the local session rather than
 * inventing an account. Wire it to real user data once /auth exists.
 */
const ProfilePage: React.FC = () => {
  const { sessions, messages, dataset } = useApp();

  const answered = messages.filter((m) => m.role === 'assistant' && m.status === 'complete').length;

  return (
    <div className="scrollbar-slim min-h-0 flex-1 overflow-y-auto">
      <div className="mx-auto w-full max-w-3xl space-y-5 px-6 py-8">
        <header>
          <h1 className="text-xl font-semibold tracking-tight text-ink">Session</h1>
          <p className="mt-1 text-[13px] text-ink-muted">
            Local workspace state. Nothing here leaves this browser.
          </p>
        </header>

        <section className="grid gap-3 sm:grid-cols-3">
          {[
            { label: 'Answers this session', value: answered },
            { label: 'Archived conversations', value: sessions.length },
            { label: 'Active scene', value: dataset ? '1' : '0' },
          ].map((stat) => (
            <div key={stat.label} className="rounded-panel border border-line bg-surface p-4">
              <p className="label-caps text-ink-faint">{stat.label}</p>
              <p className="mt-2 font-mono text-2xl text-ink">{stat.value}</p>
            </div>
          ))}
        </section>

        {dataset && (
          <section className="rounded-panel border border-line bg-surface p-5">
            <h2 className="mb-3 text-sm font-medium text-ink">Loaded scene</h2>
            <div className="flex flex-wrap items-center gap-2">
              <Badge variant="info">{dataset.sensor}</Badge>
              <Badge variant="quiet">{dataset.mode}</Badge>
              <Badge variant="quiet">{dataset.crs}</Badge>
            </div>
            <p className="mt-3 break-all font-mono text-[12px] text-ink-muted">{dataset.name}</p>
          </section>
        )}

        <section className="flex items-start gap-3 rounded-panel border border-line bg-surface-2 p-4">
          <Info size={15} className="mt-0.5 shrink-0 text-ink-faint" aria-hidden />
          <p className="text-[12.5px] leading-relaxed text-ink-muted">
            Accounts are not implemented. The sign-in screens are UI only and do
            not authenticate against anything — add a real auth endpoint before
            treating them as a security boundary.
          </p>
        </section>
      </div>
    </div>
  );
};

export default ProfilePage;
