import { Panel } from '../components/ui/Panel';
import MapViewport from '../components/map/MapViewport';
import { motion } from 'framer-motion';
import { useTrace } from '../context/TraceContext';
import { Globe } from 'lucide-react';

const Dashboard = () => {
  const { activeGeoJson, activeDatasetName, activeSensor } = useTrace();

  return (
    <div className="h-full w-full relative flex flex-col overflow-hidden bg-space-black">
      {/* Top Header Overlay */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="absolute top-4 left-4 z-10 pointer-events-none"
      >
        <Panel variant="glass" className="px-3.5 py-2 border border-white/15 pointer-events-auto flex items-center gap-2.5 shadow-2xl backdrop-blur-xl">
          <Globe size={15} className="text-accent-cyan" />
          <div>
            <h2 className="text-xs font-mono font-bold text-white uppercase tracking-widest">
              Global Satellite Map Viewport
            </h2>
            <p className="text-[10px] font-mono text-slate-400">
              {activeDatasetName ? `${activeDatasetName} • Active` : 'Interactive Slippy Exploration Mode'}
            </p>
          </div>
        </Panel>
      </motion.div>

      {/* Main Fullscreen Interactive Map */}
      <div className="flex-1 h-full w-full">
        <MapViewport
          geoJsonData={activeGeoJson}
          datasetName={activeDatasetName}
          sensor={activeSensor}
        />
      </div>
    </div>
  );
};

export default Dashboard;
