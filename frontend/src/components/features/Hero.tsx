import { useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { Stars, Float } from '@react-three/drei';
import * as THREE from 'three';
import { motion } from 'framer-motion';
import { Button } from '../ui/Button';
import { useNavigate } from 'react-router-dom';
import { Radio as RadioIcon, Sparkles as SparklesIcon } from 'lucide-react';

function OrbitingSatellite() {
  const satelliteRef = useRef<THREE.Group>(null);
  const orbitRadius = 3.6;

  useFrame(({ clock }) => {
    const t = clock.getElapsedTime() * 0.4;
    if (satelliteRef.current) {
      satelliteRef.current.position.x = Math.cos(t) * orbitRadius;
      satelliteRef.current.position.z = Math.sin(t) * orbitRadius;
      satelliteRef.current.position.y = Math.sin(t * 1.5) * 0.9;
      satelliteRef.current.rotation.y = -t + Math.PI / 2;
    }
  });

  return (
    <group ref={satelliteRef}>
      <mesh>
        <boxGeometry args={[0.22, 0.22, 0.35]} />
        <meshStandardMaterial color="#00f2ff" roughness={0.2} metalness={0.8} />
      </mesh>
      <mesh position={[-0.45, 0, 0]}>
        <boxGeometry args={[0.55, 0.02, 0.25]} />
        <meshStandardMaterial color="#00dec2" roughness={0.3} metalness={0.7} />
      </mesh>
      <mesh position={[0.45, 0, 0]}>
        <boxGeometry args={[0.55, 0.02, 0.25]} />
        <meshStandardMaterial color="#00dec2" roughness={0.3} metalness={0.7} />
      </mesh>
      <pointLight color="#00f2ff" intensity={1.5} distance={2} />
    </group>
  );
}

function OrbitRing() {
  const points = [];
  const segments = 128;
  const radius = 3.6;
  for (let i = 0; i <= segments; i++) {
    const theta = (i / segments) * Math.PI * 2;
    points.push(
      new THREE.Vector3(
        Math.cos(theta) * radius,
        Math.sin(theta * 1.5) * 0.9,
        Math.sin(theta) * radius
      )
    );
  }
  const lineGeometry = new THREE.BufferGeometry().setFromPoints(points);

  return (
    <primitive object={new THREE.Line(
      lineGeometry,
      new THREE.LineBasicMaterial({ color: '#00f2ff', transparent: true, opacity: 0.15 })
    )} />
  );
}

function EarthGlobe() {
  const globeRef = useRef<THREE.Group>(null);

  useFrame(() => {
    if (globeRef.current) {
      globeRef.current.rotation.y += 0.0025;
    }
  });

  return (
    <group ref={globeRef}>
      <mesh>
        <sphereGeometry args={[2.2, 36, 36]} />
        <meshBasicMaterial
          color="#00f2ff"
          wireframe
          transparent
          opacity={0.1}
        />
      </mesh>
      <mesh>
        <sphereGeometry args={[2.16, 48, 48]} />
        <meshStandardMaterial
          color="#060914"
          roughness={0.9}
          metalness={0.1}
        />
      </mesh>
      <mesh rotation={[Math.PI / 2, 0, 0]}>
        <ringGeometry args={[2.22, 2.24, 64]} />
        <meshBasicMaterial color="#00dec2" transparent opacity={0.2} side={THREE.DoubleSide} />
      </mesh>
    </group>
  );
}

export const Hero = () => {
  const navigate = useNavigate();
  const scrollTo = (id: string) => {
    const el = document.getElementById(id);
    if (el) {
      el.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <section className="relative min-h-screen w-full flex items-center justify-center overflow-hidden bg-space-black">
      <div className="absolute inset-0 z-0">
        <Canvas camera={{ position: [0, 1.2, 7.2], fov: 45 }}>
          <ambientLight intensity={0.4} />
          <directionalLight position={[10, 10, 10]} intensity={1.2} />
          <Stars radius={120} depth={60} count={6000} factor={4} saturation={0} fade speed={0.8} />
          <Float speed={1.2} rotationIntensity={0.3} floatIntensity={0.8}>
            <group rotation={[0.2, 0.4, 0]}>
              <EarthGlobe />
              <OrbitRing />
              <OrbitingSatellite />
            </group>
          </Float>
        </Canvas>
      </div>

      <div
        className="absolute inset-0 pointer-events-none z-[1]"
        style={{
          background: 'radial-gradient(ellipse at 50% 50%, transparent 40%, #05070C 95%)',
        }}
      />

      <div className="relative z-10 container mx-auto px-6 text-center max-w-4xl pt-16 pb-24">
        <motion.div
          initial={{ opacity: 0, y: 25 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, ease: 'easeOut' }}
        >
          <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-white/5 border border-white/15 text-xs font-mono text-accent-cyan mb-8 backdrop-blur-md shadow-lg">
            <RadioIcon size={12} className="animate-pulse text-accent-cyan" />
            <span className="text-white font-medium">Smart India Hackathon 2026</span>
            <span className="text-white/30">•</span>
            <span className="text-slate-300">PS 26167</span>
            <span className="text-white/30">•</span>
            <span className="text-accent-teal">ISRO Space Tech</span>
          </div>

          <h1 className="text-5xl sm:text-6xl md:text-7xl font-extrabold text-white mb-6 tracking-tight leading-none">
            SatQuery <span className="text-accent-cyan text-glow">AI</span>
          </h1>

          <p className="text-lg sm:text-xl md:text-2xl text-slate-300 max-w-2xl mx-auto mb-10 leading-relaxed font-normal">
            Agentic vision-language assistant for satellite and SAR remote sensing analysis — delivering audited answers with verifiable visual evidence.
          </p>

          <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
            <Button
              size="lg"
              onClick={() => navigate('/console')}
              className="w-full sm:w-auto font-mono text-sm uppercase tracking-wider"
            >
              <SparklesIcon size={16} className="mr-2 inline text-space-black" />
              Launch Console
            </Button>
            <Button
              variant="outline"
              size="lg"
              onClick={() => scrollTo('how-it-works')}
              className="w-full sm:w-auto font-mono text-sm uppercase tracking-wider backdrop-blur-sm"
            >
              How It Works
            </Button>
          </div>
        </motion.div>
      </div>

      <motion.button
        type="button"
        onClick={() => scrollTo('console')}
        className="absolute bottom-8 left-1/2 -translate-x-1/2 z-10 flex flex-col items-center gap-2 text-slate-500 hover:text-accent-cyan transition-colors group cursor-pointer"
        animate={{ y: [0, 8, 0] }}
        transition={{ repeat: Infinity, duration: 2.2, ease: 'easeInOut' }}
      >
        <span className="text-[10px] font-mono uppercase tracking-widest text-slate-400 group-hover:text-accent-cyan transition-colors">
          Explore System
        </span>
        <div className="w-5 h-8 border border-white/20 rounded-full flex justify-center p-1 group-hover:border-accent-cyan transition-colors">
          <div className="w-1 h-2 bg-accent-cyan rounded-full" />
        </div>
      </motion.button>
    </section>
  );
};

export default Hero;
