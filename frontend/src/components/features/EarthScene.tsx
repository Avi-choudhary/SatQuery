import { useEffect, useMemo, useRef } from 'react';
import { Canvas, useFrame, useLoader, useThree } from '@react-three/fiber';
import { Stars, useGLTF } from '@react-three/drei';
import * as THREE from 'three';
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js';

/*
 * Photoreal Earth for the landing hero.
 *
 * Textures are NASA-derived imagery bundled in public/textures (see
 * frontend/CLAUDE.md). Nothing is fetched at runtime, so the globe renders the
 * same offline as online.
 *
 * The globe is four concentric shells:
 *   1. surface   — colour + normal + specular, lit by a single directional "sun"
 *   2. night     — city lights, additively blended and masked to the dark side
 *   3. clouds    — semi-transparent, rotating slightly faster than the surface
 *   4. atmosphere — back-facing fresnel shell for limb glow
 */

const EARTH_RADIUS = 2.15;
/** Shared by the spacecraft and the traced path, so the two always agree. */
const ORBIT_RADIUS = 3.4;
const ORBIT_TILT = 0.88;
const SUN_DIRECTION = new THREE.Vector3(1, 0.28, 0.72).normalize();
const TEXTURE_BASE = '/textures';

/* -------------------------------------------------------------------------- */
/* Shaders                                                                     */
/* -------------------------------------------------------------------------- */

const NIGHT_VERTEX = /* glsl */ `
  varying vec2 vUv;
  varying vec3 vWorldNormal;
  void main() {
    vUv = uv;
    // World-space normal: the globe spins, the sun does not.
    vWorldNormal = normalize(mat3(modelMatrix) * normal);
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

const NIGHT_FRAGMENT = /* glsl */ `
  uniform sampler2D lightsMap;
  uniform vec3 sunDirection;
  varying vec2 vUv;
  varying vec3 vWorldNormal;

  void main() {
    float facing = dot(normalize(vWorldNormal), sunDirection);
    // Fade the lights in across the terminator rather than snapping at 0.
    float night = smoothstep(0.12, -0.30, facing);
    vec3 lights = texture2D(lightsMap, vUv).rgb;
    float energy = max(lights.r, max(lights.g, lights.b));
    // Warm the sodium-vapour glow slightly.
    vec3 tinted = lights * vec3(1.0, 0.86, 0.62);
    gl_FragColor = vec4(tinted * night * 2.1, energy * night);
  }
`;

const ATMOSPHERE_VERTEX = /* glsl */ `
  varying vec3 vViewNormal;
  void main() {
    vViewNormal = normalize(normalMatrix * normal);
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

const ATMOSPHERE_FRAGMENT = /* glsl */ `
  uniform vec3 glowColor;
  varying vec3 vViewNormal;
  void main() {
    // Rendered on the back faces, so the rim sits where the normal turns away.
    float rim = pow(0.68 - dot(vViewNormal, vec3(0.0, 0.0, 1.0)), 3.2);
    gl_FragColor = vec4(glowColor, 1.0) * clamp(rim, 0.0, 1.0);
  }
`;

/* -------------------------------------------------------------------------- */
/* Earth                                                                       */
/* -------------------------------------------------------------------------- */

const Earth: React.FC<{ paused: boolean }> = ({ paused }) => {
  const surfaceRef = useRef<THREE.Mesh>(null);
  const nightRef = useRef<THREE.Mesh>(null);
  const cloudsRef = useRef<THREE.Mesh>(null);

  const [colorMap, normalMap, specularMap, cloudMap, lightsMap] = useLoader(THREE.TextureLoader, [
    `${TEXTURE_BASE}/earth_atmos_2048.jpg`,
    `${TEXTURE_BASE}/earth_normal_2048.jpg`,
    `${TEXTURE_BASE}/earth_specular_2048.jpg`,
    `${TEXTURE_BASE}/earth_clouds_1024.png`,
    `${TEXTURE_BASE}/earth_lights_2048.png`,
  ]);

  useMemo(() => {
    // Colour imagery is authored in sRGB; the data maps are not.
    for (const texture of [colorMap, cloudMap, lightsMap]) {
      texture.colorSpace = THREE.SRGBColorSpace;
      texture.anisotropy = 8;
    }
    for (const texture of [normalMap, specularMap]) {
      texture.colorSpace = THREE.NoColorSpace;
      texture.anisotropy = 8;
    }
  }, [colorMap, normalMap, specularMap, cloudMap, lightsMap]);

  const nightMaterial = useMemo(
    () =>
      new THREE.ShaderMaterial({
        uniforms: {
          lightsMap: { value: lightsMap },
          sunDirection: { value: SUN_DIRECTION },
        },
        vertexShader: NIGHT_VERTEX,
        fragmentShader: NIGHT_FRAGMENT,
        blending: THREE.AdditiveBlending,
        transparent: true,
        depthWrite: false,
      }),
    [lightsMap]
  );

  const atmosphereMaterial = useMemo(
    () =>
      new THREE.ShaderMaterial({
        uniforms: { glowColor: { value: new THREE.Color('#2ea8ff') } },
        vertexShader: ATMOSPHERE_VERTEX,
        fragmentShader: ATMOSPHERE_FRAGMENT,
        side: THREE.BackSide,
        blending: THREE.AdditiveBlending,
        transparent: true,
        depthWrite: false,
      }),
    []
  );

  useFrame((_, delta) => {
    if (paused) return;
    const spin = delta * 0.045;
    if (surfaceRef.current) surfaceRef.current.rotation.y += spin;
    if (nightRef.current) nightRef.current.rotation.y += spin;
    // Cloud deck drifts a little faster than the ground beneath it.
    if (cloudsRef.current) cloudsRef.current.rotation.y += spin * 1.28;
  });

  return (
    <group rotation={[0, 0, (23.44 * Math.PI) / 180]}>
      {/* 1. Surface */}
      <mesh ref={surfaceRef}>
        <sphereGeometry args={[EARTH_RADIUS, 96, 96]} />
        <meshPhongMaterial
          map={colorMap}
          normalMap={normalMap}
          normalScale={new THREE.Vector2(0.85, 0.85)}
          specularMap={specularMap}
          specular={new THREE.Color('#2b3d55')}
          shininess={18}
        />
      </mesh>

      {/* 2. City lights on the night side */}
      <mesh ref={nightRef} material={nightMaterial}>
        <sphereGeometry args={[EARTH_RADIUS + 0.004, 96, 96]} />
      </mesh>

      {/* 3. Cloud deck */}
      <mesh ref={cloudsRef}>
        <sphereGeometry args={[EARTH_RADIUS + 0.028, 72, 72]} />
        <meshPhongMaterial
          map={cloudMap}
          transparent
          opacity={0.42}
          depthWrite={false}
          shininess={2}
        />
      </mesh>

      {/* 4. Atmospheric limb */}
      <mesh material={atmosphereMaterial} scale={1.055}>
        <sphereGeometry args={[EARTH_RADIUS, 64, 64]} />
      </mesh>
    </group>
  );
};

/* -------------------------------------------------------------------------- */
/* Environment                                                                 */
/* -------------------------------------------------------------------------- */

/**
 * Physically-based metals reflect their surroundings; with nothing to reflect
 * they render pure black, which is exactly how the NASA spacecraft first
 * appeared. RoomEnvironment ships inside three, so this gives the metal
 * something to catch without pulling an HDRI off a CDN.
 *
 * Only Standard/Physical materials read scene.environment, so the Earth's
 * Phong shells are untouched by it.
 */
const StudioEnvironment: React.FC = () => {
  const scene = useThree((state) => state.scene);
  const gl = useThree((state) => state.gl);

  useEffect(() => {
    const pmrem = new THREE.PMREMGenerator(gl);
    const room = new RoomEnvironment();
    const target = pmrem.fromScene(room, 0.04);

    scene.environment = target.texture;
    // Dialled well down: this is orbit, not a photo studio.
    scene.environmentIntensity = 0.4;

    return () => {
      scene.environment = null;
      target.dispose();
      pmrem.dispose();
      room.clear();
    };
  }, [gl, scene]);

  return null;
};

/* -------------------------------------------------------------------------- */
/* Satellite                                                                   */
/* -------------------------------------------------------------------------- */

/**
 * Real spacecraft geometry: RADARSAT-1, a synthetic-aperture radar imaging
 * satellite, from NASA's 3D Resources collection ("free and without
 * copyright"). It carries the large SAR antenna panel this project's Sentinel-1
 * work is actually about, which no box-and-cylinder stand-in conveys.
 *
 * Swap MODEL_URL for any other GLB in public/models — the geometry is
 * re-centred and scaled to MODEL_TARGET_SIZE automatically, so a different
 * model needs no other change.
 */
const MODEL_URL = '/models/radarsat-1.glb';
const MODEL_TARGET_SIZE = 1.15;

const SatelliteModel: React.FC = () => {
  const { scene } = useGLTF(MODEL_URL);

  // useGLTF caches the parsed document, so clone before touching materials.
  const model = useMemo(() => {
    const clone = scene.clone(true);

    // Normalise: authored units and origin vary between models.
    const box = new THREE.Box3().setFromObject(clone);
    const size = box.getSize(new THREE.Vector3());
    const centre = box.getCenter(new THREE.Vector3());
    const largest = Math.max(size.x, size.y, size.z) || 1;
    const scale = MODEL_TARGET_SIZE / largest;

    clone.position.sub(centre);
    clone.scale.setScalar(scale);
    clone.position.multiplyScalar(scale);

    // Re-material the whole model.
    //
    // This export's MeshPhysicalMaterials render as a black cut-out no matter
    // how the scene is lit — they carry glTF extension state (transmission /
    // specular) that needs a backdrop this scene does not have. Rather than
    // fight it, keep each part's authored base colour and rebuild the surface
    // with a plain standard material we fully control.
    clone.traverse((node) => {
      const mesh = node as THREE.Mesh;
      if (!mesh.isMesh) return;

      mesh.castShadow = false;
      mesh.receiveShadow = false;

      const source = (Array.isArray(mesh.material) ? mesh.material[0] : mesh.material) as
        | THREE.MeshStandardMaterial
        | undefined;
      const baseColor = source?.color?.clone() ?? new THREE.Color('#d8dde5');

      // Solar arrays are the dark blue parts; give them a touch of sheen and
      // let the structure read as brushed white foil.
      const isPanel = baseColor.b > baseColor.r * 1.3 && baseColor.b > 0.15;

      mesh.material = new THREE.MeshStandardMaterial({
        color: baseColor,
        metalness: isPanel ? 0.55 : 0.2,
        roughness: isPanel ? 0.32 : 0.55,
        emissive: isPanel ? new THREE.Color('#0a1a38') : new THREE.Color('#0b0f16'),
        emissiveIntensity: isPanel ? 0.5 : 0.35,
        side: THREE.DoubleSide,
      });
    });

    return clone;
  }, [scene]);

  return <primitive object={model} />;
};

useGLTF.preload(MODEL_URL);

const Satellite: React.FC<{ paused: boolean }> = ({ paused }) => {
  const orbitRef = useRef<THREE.Group>(null);
  const bodyRef = useRef<THREE.Group>(null);

  useFrame(({ clock }) => {
    const group = orbitRef.current;
    if (!group) return;

    const t = paused ? 1.05 : clock.getElapsedTime() * 0.13;
    group.position.set(
      Math.cos(t) * ORBIT_RADIUS,
      Math.sin(t) * ORBIT_RADIUS * Math.sin(ORBIT_TILT),
      Math.sin(t) * ORBIT_RADIUS
    );

    // Nadir pointing: a real imaging satellite keeps its instrument face
    // toward the planet all the way round the orbit.
    bodyRef.current?.lookAt(0, 0, 0);
  });

  return (
    <group ref={orbitRef}>
      {/*
        Local key + fill. The scene's single distant "sun" leaves the
        spacecraft reading as a dark cut-out against the limb, so it gets its
        own rig. These sit outside the hull and are distance-limited, so they
        light the spacecraft without washing the Earth.
      */}
      <pointLight position={[1.6, 1.1, 1.6]} intensity={7} distance={6} decay={2} color="#fff4e2" />
      <pointLight position={[-1.4, -0.6, -1.2]} intensity={2.2} distance={5} decay={2} color="#8fbcff" />

      <group ref={bodyRef}>
        <SatelliteModel />
      </group>
    </group>
  );
};

/** Faint ellipse tracing the satellite's ground track. */
const OrbitPath: React.FC = () => {
  const geometry = useMemo(() => {
    const points: THREE.Vector3[] = [];
    for (let i = 0; i <= 200; i += 1) {
      const theta = (i / 200) * Math.PI * 2;
      points.push(
        new THREE.Vector3(
          Math.cos(theta) * ORBIT_RADIUS,
          Math.sin(theta) * ORBIT_RADIUS * Math.sin(ORBIT_TILT),
          Math.sin(theta) * ORBIT_RADIUS
        )
      );
    }
    return new THREE.BufferGeometry().setFromPoints(points);
  }, []);

  return (
    <primitive
      object={
        new THREE.Line(
          geometry,
          new THREE.LineBasicMaterial({ color: '#00d8f0', transparent: true, opacity: 0.16 })
        )
      }
    />
  );
};

/* -------------------------------------------------------------------------- */
/* Scene                                                                       */
/* -------------------------------------------------------------------------- */

export const EarthScene: React.FC<{ paused?: boolean }> = ({ paused = false }) => (
  <Canvas
    camera={{ position: [0, 0.6, 8.6], fov: 40 }}
    dpr={[1, 2]}
    gl={{ antialias: true, powerPreference: 'high-performance' }}
  >
    <color attach="background" args={['#05070c']} />
    <StudioEnvironment />

    {/* Sun */}
    <directionalLight
      position={SUN_DIRECTION.clone().multiplyScalar(12).toArray()}
      intensity={3.1}
      color="#fff6e8"
    />
    {/* Starlight fill, so the night side is not pure black */}
    <ambientLight intensity={0.09} color="#5f7fb0" />
    {/* Cool bounce from the limb */}
    <pointLight position={[-7, -2.5, -5]} intensity={0.5} color="#1d5fa8" />

    <Stars radius={140} depth={70} count={4200} factor={3.4} saturation={0} fade speed={0.55} />

    {/*
      Centred, with just enough downward offset that the headline above it is
      not fighting the daylit hemisphere for contrast.
    */}
    <group position={[0, -1.35, 0]} rotation={[0.16, 0.4, 0]}>
      <Earth paused={paused} />
      <OrbitPath />
      <Satellite paused={paused} />
    </group>
  </Canvas>
);

export default EarthScene;
