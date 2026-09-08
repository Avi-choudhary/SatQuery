// TODO(backend): Connect to FastAPI SSE streaming endpoint (POST /api/v1/query/stream).
// Request Payload:
//   - query: string
//   - dataset_id: string (from UploadDropzone)
//   - spatial_extent?: [minLng, minLat, maxLng, maxLat]
//   - session_id: string
// Server-Sent Events (SSE) stream will emit:
//   - { event: "intent", data: { tool: "Change Detection", confidence: 0.94 } }
//   - { event: "token", data: { text: "..." } }
//   - { event: "evidence", data: { geojson: {...}, metrics: {...} } }

import { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Panel } from '../ui/Panel';
import { Button } from '../ui/Button';
import { Badge } from '../ui/Badge';
import { UploadDropzone, type UploadedDataset } from './UploadDropzone';
import type { ChatMessage } from '../../mocks/chatHistory';
import { useTrace, type TraceSession } from '../../context/TraceContext';
import { submitSatQuery } from '../../lib/api';
import { Send, Bot, User, Sparkles, ChevronDown, ChevronUp, RotateCcw } from 'lucide-react';

const SUGGESTED_QUERIES = [
  'What is the best area for growing sugarcane in this image?',
  'Identify industrial storage tanks and clusters in the south.',
  'Analyze vegetation canopy health change via NDVI/SAR.',
];

interface QueryPanelProps {
  onQuerySuccess?: (query: string, data: any, dataset?: UploadedDataset | null) => void;
  onDatasetChange?: (dataset: UploadedDataset) => void;
  activeDatasetName?: string;
  activeSession?: TraceSession | null;
}

export const QueryPanel: React.FC<QueryPanelProps> = ({
  onQuerySuccess,
  onDatasetChange,
  activeDatasetName,
  activeSession,
}) => {
  const {
    chatMessages,
    setChatMessages,
    clearChat,
    activeUploadedDataset,
    setActiveUploadedDataset
  } = useTrace();

  const [query, setQuery] = useState('');
  const messages = chatMessages;
  const setMessages = setChatMessages;
  const activeDataset = activeUploadedDataset;

  const [isProcessing, setIsProcessing] = useState(false);
  const [showUploader, setShowUploader] = useState(true);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isProcessing]);

  // Synchronize chat messages with loaded historical session from TraceContext
  useEffect(() => {
    if (!activeSession) return;
    setMessages((prev) => {
      const alreadyHasSession = prev.some(
        (m) => m.content === activeSession.query || m.content === activeSession.textAnswer
      );
      if (alreadyHasSession) return prev;

      const userMsg: ChatMessage = {
        id: `hist-user-${activeSession.id}`,
        role: 'user',
        content: activeSession.query,
        timestamp: activeSession.timestamp,
        datasetName: activeSession.datasetName || activeDatasetName || 'Active Scene',
        sensorType: (activeSession.sensor as any) || 'Optical (Sentinel-2)',
      };

      const assistantMsg: ChatMessage = {
        id: `hist-asst-${activeSession.id}`,
        role: 'assistant',
        content: activeSession.textAnswer,
        timestamp: activeSession.timestamp,
        datasetName: activeSession.datasetName || activeDatasetName,
        metadata: {
          confidence: 0.95,
          toolsUsed: activeSession.trace.slice(0, 3).map((t) => t.action),
          executionTimeMs: activeSession.trace.reduce((acc, t) => acc + t.latencyMs, 0),
        },
      };

      return [...prev, userMsg, assistantMsg];
    });
  }, [activeSession, activeDatasetName]);

  const handleSend = async (textToSend?: string) => {
    const queryText = (textToSend ?? query).trim();
    if (!queryText || isProcessing) return;

    const currentDatasetName = activeDataset?.name || activeDatasetName || 'Uploaded Scene';
    const userMessage: ChatMessage = {
      id: `msg-${Date.now()}`,
      role: 'user',
      content: queryText,
      timestamp: new Date().toISOString(),
      datasetName: currentDatasetName,
      sensorType: activeDataset?.sensor ?? 'Optical (Sentinel-2)',
    };

    setMessages((prev) => [...prev, userMessage]);
    setQuery('');
    setIsProcessing(true);

    const startTime = performance.now();
    try {
      // Dispatch query to central backend (which routes to remote PC GPU or local GPU)
      const filesToSend = activeDataset?.files && activeDataset.files.length > 1
        ? activeDataset.files
        : (activeDataset?.file || null);
      const data = await submitSatQuery(queryText, filesToSend, currentDatasetName);
      const elapsedMs = Math.round(performance.now() - startTime);

      // Notify parent Console with the exact user query text and active dataset
      onQuerySuccess?.(queryText, data, activeDataset);

      const rawSteps = data.execution_trace?.steps || data.execution_trace?.logs || data.trace_log;
      const formattedSteps: string[] = Array.isArray(rawSteps)
        ? (rawSteps
            .map((s: string): string | null => {
              const lower = s.toLowerCase();
              if (lower.includes('routed to specialist tool 1') || lower.includes('vqa')) return 'VQA Specialist';
              if (lower.includes('routed to specialist tool 2') || lower.includes('grounding')) return 'Visual Grounding';
              if (lower.includes('change detection') || lower.includes('changeformer') || lower.includes('change detective')) return 'ChangeFormer (CUDA)';
              if (lower.includes('agentic controller') || lower.includes('classified intent')) return 'Agentic Controller';
              if (lower.includes('inference complete')) return 'Qwen3-VL (CUDA)';
              if (lower.includes('gis pre-processing') || lower.includes('co-registration')) return 'GIS Engine';
              return null;
            })
            .filter((val): val is string => typeof val === 'string')
            .filter((val, idx, arr) => arr.indexOf(val) === idx))
        : ['Agentic Controller', 'Qwen3-VL (CUDA)'];

      const finalToolsUsed = formattedSteps.length > 0 ? formattedSteps : ['Agentic Controller', 'Qwen3-VL (CUDA)'];

      const assistantMessage: ChatMessage = {
        id: `msg-${Date.now() + 1}`,
        role: 'assistant',
        content: data.text_answer || data.response || 'SatQuery AI analysis complete.',
        timestamp: new Date().toISOString(),
        datasetName: activeDataset?.name,
        metadata: {
          confidence: data.confidence ?? 0.965,
          toolsUsed: finalToolsUsed,
          detectionCount: Array.isArray(data.visual_evidence) ? data.visual_evidence.length : (data.visual_evidence ? 1 : 0),
          executionTimeMs: elapsedMs,
        },
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (err: any) {
      console.warn('Backend API notice, falling back:', err);
      const elapsedMs = Math.round(performance.now() - startTime);
      const fallbackMessage: ChatMessage = {
        id: `msg-${Date.now() + 1}`,
        role: 'assistant',
        content: `Notice: ${err?.message || 'Backend unreachable'}. If running on your laptop, make sure your desktop PC GPU host is online and MODEL_SERVICE_URL is set in backend/.env.`,
        timestamp: new Date().toISOString(),
        metadata: {
          confidence: 0.5,
          toolsUsed: ['Network Status Check'],
          executionTimeMs: elapsedMs,
        },
      };
      setMessages((prev) => [...prev, fallbackMessage]);
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <Panel variant="glass" className="h-full flex flex-col overflow-hidden border border-white/10">
      {/* Header */}
      <div className="px-4 py-3 border-b border-white/10 flex items-center justify-between bg-space-black/50">
        <div className="flex items-center gap-2">
          <Bot size={16} className="text-accent-cyan" />
          <h3 className="text-xs font-mono uppercase tracking-widest text-accent-cyan font-semibold">
            Agentic Console
          </h3>
        </div>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={clearChat}
            className="text-[11px] font-mono text-slate-400 hover:text-accent-cyan flex items-center gap-1 transition-colors cursor-pointer mr-1"
            title="Clear chat history and start new session"
          >
            <RotateCcw size={12} />
            <span>Reset</span>
          </button>
          <button
            type="button"
            onClick={() => setShowUploader(!showUploader)}
            className="text-[11px] font-mono text-slate-400 hover:text-white flex items-center gap-1 transition-colors cursor-pointer"
          >
            <span>GeoTIFF Ingest</span>
            {showUploader ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
          </button>
          <Badge variant="info">Ready</Badge>
        </div>
      </div>

      {/* Collapsible GeoTIFF Upload Dropzone */}
      <AnimatePresence>
        {showUploader && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.25 }}
            className="overflow-hidden border-b border-white/10 bg-space-black/30 p-3"
          >
            <UploadDropzone
              onDatasetSelect={(ds) => {
                setActiveUploadedDataset(ds);
                onDatasetChange?.(ds);
              }}
            />
          </motion.div>
        )}
      </AnimatePresence>

      {/* Message Thread */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        <AnimatePresence initial={false}>
          {messages.map((msg) => {
            const isUser = msg.role === 'user';
            return (
              <motion.div
                key={msg.id}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className={`flex gap-2.5 ${isUser ? 'justify-end' : 'justify-start'}`}
              >
                {!isUser && (
                  <div className="w-6 h-6 rounded-full bg-accent-cyan/15 border border-accent-cyan/30 flex items-center justify-center text-accent-cyan shrink-0 mt-1">
                    <Bot size={13} />
                  </div>
                )}

                <div
                  className={`max-w-[85%] rounded-xl p-3.5 text-xs leading-relaxed ${
                    isUser
                      ? 'bg-accent-cyan/15 border border-accent-cyan/30 text-white shadow-sm'
                      : 'bg-space-navy/90 border border-white/10 text-slate-200 shadow-md'
                  }`}
                >
                  <div className="flex items-center justify-between gap-3 mb-1.5 pb-1 border-b border-white/10 text-[10px] font-mono text-slate-400">
                    <span className="font-semibold uppercase text-accent-cyan">
                      {isUser ? 'Analyst' : 'SatQuery Agent'}
                    </span>
                    <span>{new Date(msg.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
                  </div>

                  <p className="whitespace-pre-wrap">{msg.content}</p>

                  {/* Assistant Telemetry & Metadata Footer */}
                  {msg.metadata && (
                    <div className="mt-2.5 pt-2 border-t border-white/10 text-[10px] font-mono space-y-1 text-slate-400 bg-space-black/30 p-2 rounded">
                      <div className="flex justify-between items-center text-accent-cyan font-medium">
                        <span>Confidence: {(msg.metadata.confidence! * 100).toFixed(1)}%</span>
                        <span>{msg.metadata.executionTimeMs} ms</span>
                      </div>
                      {msg.metadata.toolsUsed && (
                        <div className="text-[9px] text-slate-400 truncate">
                          Tools: {msg.metadata.toolsUsed.join(' → ')}
                        </div>
                      )}
                      {msg.metadata.areaChangedHa && (
                        <div className="text-accent-warm text-[9px]">
                          Evidence: Net delta +{msg.metadata.areaChangedHa} hectares
                        </div>
                      )}
                    </div>
                  )}
                </div>

                {isUser && (
                  <div className="w-6 h-6 rounded-full bg-white/10 border border-white/20 flex items-center justify-center text-white shrink-0 mt-1">
                    <User size={13} />
                  </div>
                )}
              </motion.div>
            );
          })}
        </AnimatePresence>

        {/* Processing Indicator */}
        {isProcessing && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="flex items-center gap-2 text-xs font-mono text-accent-cyan p-2 bg-accent-cyan/10 rounded-lg border border-accent-cyan/20 w-fit"
          >
            <div className="w-2 h-2 rounded-full bg-accent-cyan animate-ping" />
            <span>Agentic controller routing query across specialist models...</span>
          </motion.div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Suggested Prompt Chips */}
      <div className="px-3 py-2 border-t border-white/10 bg-space-black/40 flex items-center gap-1.5 overflow-x-auto scrollbar-hide">
        <Sparkles size={13} className="text-accent-cyan shrink-0" />
        <span className="text-[10px] font-mono text-slate-400 shrink-0">Sample:</span>
        {SUGGESTED_QUERIES.map((sq, i) => (
          <button
            key={i}
            type="button"
            onClick={() => handleSend(sq)}
            className="text-[10px] font-mono px-2 py-1 rounded bg-white/5 hover:bg-accent-cyan/20 hover:text-accent-cyan text-slate-300 border border-white/10 transition-colors whitespace-nowrap shrink-0 cursor-pointer"
          >
            {sq}
          </button>
        ))}
      </div>

      {/* Input Area */}
      <div className="p-3 border-t border-white/10 bg-space-black/60">
        <form
          onSubmit={(e) => {
            e.preventDefault();
            handleSend();
          }}
          className="relative"
        >
          <textarea
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Ask a question about your satellite/SAR imagery..."
            rows={2}
            className="w-full bg-space-navy/80 border border-white/15 rounded-xl p-3 pr-12 text-xs text-white placeholder:text-slate-500 focus:outline-none focus:border-accent-cyan focus:ring-1 focus:ring-accent-cyan resize-none transition-all"
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleSend();
              }
            }}
          />
          <Button
            type="submit"
            size="sm"
            disabled={!query.trim() || isProcessing}
            className="absolute bottom-2.5 right-2.5 p-2 h-8 w-8 rounded-lg"
          >
            <Send size={14} className="text-space-black" />
          </Button>
        </form>
      </div>
    </Panel>
  );
};

export default QueryPanel;
