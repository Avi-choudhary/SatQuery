import { useState } from 'react';
import { Panel } from '../components/ui/Panel';
import { Button } from '../components/ui/Button';
import { API_BASE_URL, fetchSystemStatus, type SystemStatusResponse } from '../lib/api';
import { CheckCircle2, XCircle, Loader2, Cpu, Globe } from 'lucide-react';

const SettingsPage = () => {
  const [testing, setTesting] = useState(false);
  const [status, setStatus] = useState<SystemStatusResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleTestConnection = async () => {
    setTesting(true);
    setError(null);
    try {
      const data = await fetchSystemStatus();
      setStatus(data);
    } catch (err: any) {
      setError(err?.message || 'Failed to connect to backend.');
      setStatus(null);
    } finally {
      setTesting(false);
    }
  };

  return (
    <div className="p-8 max-w-4xl mx-auto">
      <h1 className="text-3xl font-bold text-white mb-8">System Settings</h1>

      <div className="grid gap-6">
        <Panel variant="glass" className="p-6">
          <h2 className="text-lg font-semibold text-white mb-4">General Configuration</h2>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-sm text-slate-400">Dark Mode</span>
              <div className="w-10 h-5 bg-accent-cyan rounded-full relative">
                <div className="absolute right-1 top-1 w-3 h-3 bg-space-black rounded-full" />
              </div>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm text-slate-400">Auto-save Trace Logs</span>
              <div className="w-10 h-5 bg-accent-cyan rounded-full relative">
                <div className="absolute right-1 top-1 w-3 h-3 bg-space-black rounded-full" />
              </div>
            </div>
          </div>
        </Panel>

        <Panel variant="glass" className="p-6">
          <h2 className="text-lg font-semibold text-white mb-4">AI Backend & Remote GPU Engine</h2>
          <div className="space-y-4">
            <div className="space-y-2">
              <label className="text-xs font-mono text-slate-500 uppercase">Backend Endpoint</label>
              <input
                type="text"
                readOnly
                value={API_BASE_URL}
                className="w-full bg-space-black border border-white/10 rounded p-2 text-sm font-mono text-accent-cyan"
              />
            </div>

            <Button
              variant="outline"
              onClick={handleTestConnection}
              disabled={testing}
              className="text-xs uppercase tracking-widest flex items-center gap-2 cursor-pointer"
            >
              {testing ? <Loader2 size={14} className="animate-spin" /> : null}
              {testing ? 'Pinging Host...' : 'Test Connection & GPU'}
            </Button>

            {error && (
              <div className="p-3 bg-red-500/10 border border-red-500/30 rounded flex items-start gap-2 text-xs text-red-300">
                <XCircle size={16} className="text-red-400 mt-0.5 shrink-0" />
                <div>
                  <p className="font-semibold">Backend Unreachable</p>
                  <p>{error}</p>
                </div>
              </div>
            )}

            {status && (
              <div className="p-4 bg-space-black/60 border border-white/10 rounded-lg space-y-3 font-mono text-xs">
                <div className="flex items-center gap-2 text-green-400">
                  <CheckCircle2 size={16} />
                  <span className="font-semibold">Backend Online ({status.service})</span>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-2 text-slate-300 pt-2 border-t border-white/10">
                  <div>
                    <span className="text-slate-500">Execution Mode: </span>
                    <span className="text-accent-cyan font-bold">{status.execution_mode}</span>
                  </div>
                  {status.model_engine.remote_service_url && (
                    <div>
                      <span className="text-slate-500 flex items-center gap-1">
                        <Globe size={12} /> Remote GPU Host:
                      </span>
                      <span className="text-yellow-400 break-all">{status.model_engine.remote_service_url}</span>
                    </div>
                  )}
                  {status.model_engine.remote_host_status && (
                    <div>
                      <span className="text-slate-500">Host Status: </span>
                      <span>{status.model_engine.remote_host_status}</span>
                    </div>
                  )}
                  <div>
                    <span className="text-slate-500 flex items-center gap-1">
                      <Cpu size={12} /> Hardware:
                    </span>
                    <span>
                      {status.model_engine.remote_vram
                        ? status.model_engine.remote_vram
                        : `${status.model_engine.local_device} (${status.model_engine.local_vram_gb} GB VRAM)`}
                    </span>
                  </div>
                </div>
              </div>
            )}
          </div>
        </Panel>
      </div>
    </div>
  );
};

export default SettingsPage;
