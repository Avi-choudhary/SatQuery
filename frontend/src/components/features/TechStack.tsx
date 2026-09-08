import { motion } from 'framer-motion';
import { Cpu, ShieldCheck } from 'lucide-react';

const technologies = [
  { name: 'EVA-02 / ViT-G', category: 'Vision-Language Model', spec: '1.0B params • Remote-CLIP' },
  { name: 'Siam-NestedUNet', category: 'Change Detection Network', spec: 'Dense Skip Connections' },
  { name: 'Grounding-DINO', category: 'Visual Grounding', spec: 'Multi-scale Cross-Attention' },
  { name: 'GDAL & Rasterio', category: 'Geospatial Raster Core', spec: 'Cloud-Optimized GeoTIFF' },
  { name: 'MapLibre GL JS', category: 'Vector/Raster Mapping', spec: 'GPU WebGL Pipeline' },
  { name: 'PyTorch / TensorRT', category: 'Inference Engine', spec: 'FP16 CUDA Acceleration' },
  { name: 'FastAPI', category: 'Async Server Protocol', spec: 'SSE & WebSocket Telemetry' },
  { name: 'Three.js / R3F', category: 'Orbital Visualization', spec: 'WebGL Three-Fiber Canvas' },
];

export const TechStack = () => {
  return (
    <section className="py-16 bg-space-navy/80 border-t border-white/5">
      <div className="container mx-auto px-6">
        <div className="flex flex-col lg:flex-row items-start lg:items-center justify-between gap-8 mb-10">
          <div>
            <div className="flex items-center gap-2 text-xs font-mono text-accent-cyan uppercase tracking-widest mb-1">
              <Cpu size={14} />
              <span>Technical Instrumentation</span>
            </div>
            <h2 className="text-xl md:text-2xl font-bold text-white">
              Underlying Model & Engine Architecture
            </h2>
          </div>

          <div className="flex items-center gap-2 text-xs font-mono text-slate-400 bg-space-black/50 px-3 py-1.5 rounded-lg border border-white/10">
            <ShieldCheck size={14} className="text-accent-teal" />
            <span>Open Geospatial Consortium (OGC) Standards Compliant</span>
          </div>
        </div>

        {/* Minimal Spec Strip Grid */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {technologies.map((tech, index) => (
            <motion.div
              key={tech.name}
              initial={{ opacity: 0, y: 10 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: index * 0.05 }}
              className="p-3.5 rounded-lg bg-space-black/40 border border-white/10 hover:border-accent-cyan/30 transition-colors"
            >
              <span className="text-[10px] font-mono text-accent-cyan/80 block mb-1 uppercase tracking-wider">
                {tech.category}
              </span>
              <span className="text-xs font-bold text-white block mb-1">
                {tech.name}
              </span>
              <span className="text-[10px] font-mono text-slate-500 block truncate">
                {tech.spec}
              </span>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default TechStack;
