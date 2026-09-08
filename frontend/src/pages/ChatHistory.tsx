import { motion } from 'framer-motion';
import { Panel } from '../components/ui/Panel';
import { Calendar, Clock, ChevronRight, Cpu } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { useTrace } from '../context/TraceContext';

const ChatHistory = () => {
  const navigate = useNavigate();
  const { historySessions, loadSession } = useTrace();

  const handleSelectSession = (sessionId: string) => {
    loadSession(sessionId);
    navigate('/console');
  };

  return (
    <div className="p-8 max-w-5xl mx-auto">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-12"
      >
        <h1 className="text-4xl font-bold text-white mb-4">Query & Trace History</h1>
        <p className="text-slate-400">Review, inspect, and reload previous analysis sessions and auditable traces.</p>
      </motion.div>

      <div className="grid gap-4">
        {historySessions.map((session, index) => (
          <motion.div
            key={session.id}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: index * 0.08 }}
          >
            <div
              onClick={() => handleSelectSession(session.id)}
              className="block group cursor-pointer"
            >
              <Panel
                variant="glass"
                className="p-6 border border-white/10 group-hover:border-accent-cyan/50 group-hover:bg-white/[0.07] transition-all"
              >
                <div className="flex items-start justify-between">
                  <div className="space-y-2">
                    <div className="flex items-center gap-3 text-xs font-mono text-accent-cyan uppercase tracking-wider">
                      <Calendar size={14} />
                      {new Date(session.timestamp).toLocaleDateString()}
                      <span className="text-white/30">|</span>
                      <span className="text-white/60">{session.datasetName || 'Sentinel-2 Scene'}</span>
                    </div>
                    <h3 className="text-lg font-semibold text-white group-hover:text-accent-cyan transition-colors">
                      {session.query}
                    </h3>
                    <p className="text-sm text-slate-300 line-clamp-1 font-sans">
                      {session.textAnswer}
                    </p>
                    <div className="flex items-center gap-4 text-xs text-slate-500 pt-1">
                      <div className="flex items-center gap-1">
                        <Clock size={12} />
                        {new Date(session.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                      </div>
                      <div className="flex items-center gap-1 text-accent-teal font-mono">
                        <span>{session.trace.length} Trace Stages</span>
                      </div>
                      {session.hardware && (
                        <div className="flex items-center gap-1 text-slate-400 font-mono hidden sm:flex">
                          <Cpu size={12} />
                          <span>{session.hardware}</span>
                        </div>
                      )}
                    </div>
                  </div>
                  <ChevronRight className="text-white/20 group-hover:text-accent-cyan group-hover:translate-x-1 transition-all mt-3" size={24} />
                </div>
              </Panel>
            </div>
          </motion.div>
        ))}
      </div>

      {historySessions.length === 0 && (
        <div className="text-center py-20 text-slate-500 border-2 border-dashed border-white/5 rounded-2xl">
          <p>No previous sessions found.</p>
        </div>
      )}
    </div>
  );
};

export default ChatHistory;
