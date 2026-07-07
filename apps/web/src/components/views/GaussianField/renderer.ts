/**
 * GaussianFieldRenderer — the flagship WebGL view (§7). three.js (pinned dep)
 * quads with an analytic Gaussian-falloff fragment shader and additive glow, on
 * the near-black instrument canvas. Two modes behind a toggle:
 *
 *   • 2D (default): the original flat field — every splat's clip position is
 *     computed directly in the vertex shader (orthographic, z = 0).
 *   • 3D RELIEF (S2): the field becomes terrain. Each splat sits on a ground
 *     plane (XZ) at a HEIGHT (Y) equal to its measured `glow` (Chefer
 *     attribution) — z = zForGlow(glow), a pure function of one measured channel
 *     (§7 honesty rule; the pane labels the axis). A slow auto-orbiting
 *     perspective camera reads the terrain; a faint ground grid at Y=0 anchors
 *     depth. Splats are camera-facing billboards so they stay legible from any
 *     angle; color/opacity/glow/halo encodings are unchanged.
 *
 * GEOMETRY: NON-INSTANCED BY DESIGN
 * --------------------------------
 * The field is 197 quads baked into ONE ordinary mesh (197×4 = 788 vertices).
 * We deliberately do NOT use instanced rendering: the software GL used for
 * headless verification (SwiftShader) mis-handles instanced draws with a
 * RawShaderMaterial. 788 vertices is trivially 60 fps.
 *
 * CAMERA / PROJECTION
 * -------------------
 * BOTH modes compute `gl_Position` manually in the shader, so the scene renders
 * with a plain identity camera. In 3D the shader multiplies world positions by
 * `uViewProj`, which the CPU rebuilds each frame from a spherical orbit
 * (relief.ts `orbitEye`) via a helper PerspectiveCamera used purely for matrix
 * math. The same matrix drives the ground-grid material and the hover pick.
 *
 * IDLE MICRO-MOTION (S1)
 * ----------------------
 * A global `uShimmer` scalar (a tiny sine breath, identical for every splat)
 * multiplies body brightness so the field reads as "alive" even while the loop
 * is paused. It is NOT a data channel — a uniform gain on all splats encodes
 * nothing; it only signals liveness. The 3D auto-orbit is the 3D equivalent.
 *
 * INTERPOLATION / HIGHLIGHT
 * -------------------------
 * The 12 channels of every token are LERP'd on the CPU (interp.ts) into vertex
 * buffers each frame `t` changes; `aHighlight` uploads only on selection change.
 */
import * as THREE from "three";
import type { LoadedGaussians } from "@/src/lib/pack/types";
import { easeInOut } from "@/src/lib/loop/schedule";
import {
  RELIEF_Z_GAIN,
  nearestScreenIndex,
  orbitEye,
  type ScreenPoint,
} from "./relief";
import {
  CH,
  CLS_INDEX,
  FIELD_MARGIN,
  SIGMA_EXTENT,
  bracketSteps,
  lerp,
  lerpAngle,
} from "./interp";

// Instrument palette (mirrors tailwind.config: gauss/image/signal), as vec3s.
const GLOW_COLOR = new THREE.Color(0xb5179e); // attribution — the view's accent
const HALO_COLOR = new THREE.Color(0x4cc9f0); // attention-in — cool cyan ring
const HIGHLIGHT_COLOR = new THREE.Color(0xe8eefc); // hot selection outline
const CLS_COLOR = new THREE.Color(0x9fb4ff);
const GRID_COLOR = new THREE.Color(0x1c2333); // faint edge hue for the ground grid

// 3D orbit constants.
const ORBIT_RADIUS = 2.7;
const ORBIT_CENTER: [number, number, number] = [0, 0.16, 0];
const TOPDOWN_POLAR = 0.12; // near straight-down → reads like the flat 2D view
const ORBIT_POLAR = 1.02; // settled ~58° tilt showing the terrain relief
const AUTO_ORBIT_SPEED = 0.16; // rad/sec — the "always-alive" slow spin
const PICK_RADIUS_PX = 20;

// The four quad corners (unit square in [-1,1]), reused for every token.
const CORNERS = [
  [-1, -1],
  [1, -1],
  [-1, 1],
  [1, 1],
];

const VERT = /* glsl */ `
precision highp float;
in vec3 position;    // quad corner in [-1,1] (z unused)
in vec4 aGeom;       // x, y, rx, ry
in vec4 aColT;       // r, g, b, theta
in vec4 aParam;      // opacity, glow, halo, isCls
in float aHighlight; // 0/1 selection flag

uniform float uSigma;
uniform vec2 uFit;   // square-aspect correction (2D)
uniform float uMargin;
uniform float uMode;      // 0 = 2D, 1 = 3D relief
uniform float uZGain;     // world height per unit glow
uniform mat4 uViewProj;   // 3D view-projection
uniform vec3 uCamRight;   // 3D billboard basis
uniform vec3 uCamUp;

out vec2 vSig;
out vec3 vColor;
out float vOpacity;
out float vGlow;
out float vHalo;
out float vHighlight;
out float vIsCls;

void main() {
  vColor = aColT.xyz;
  vOpacity = aParam.x;
  vGlow = aParam.y;
  vHalo = aParam.z;
  vIsCls = aParam.w;
  vHighlight = aHighlight;
  vSig = position.xy * uSigma;

  vec2 local = position.xy * uSigma * vec2(aGeom.z, aGeom.w);
  float c = cos(aColT.w), s = sin(aColT.w);
  vec2 rot = vec2(c * local.x - s * local.y, s * local.x + c * local.y);

  if (uMode < 0.5) {
    // -- 2D: direct clip position (unchanged from M6) --
    vec2 world = aGeom.xy + rot;
    vec2 clipField = vec2(world.x * 2.0 - 1.0, 1.0 - world.y * 2.0) * uMargin;
    vec2 clipCls = vec2(-0.92, -0.92) + position.xy * 0.055;
    vec2 clip = mix(clipField, clipCls, aParam.w);
    gl_Position = vec4(clip * uFit, 0.0, 1.0);
  } else {
    // -- 3D relief: ground plane = XZ, height Y = measured glow (honesty rule) --
    float gx = (aGeom.x * 2.0 - 1.0) * uMargin;
    float gz = (aGeom.y * 2.0 - 1.0) * uMargin;
    float gy = uZGain * vGlow;
    vec3 center = vec3(gx, gy, gz);
    vec3 offset = uCamRight * rot.x + uCamUp * rot.y; // camera-facing billboard
    // CLS: parked flat at a ground corner, off the terrain.
    vec3 clsCenter = vec3(-0.92 * uMargin, 0.0, 0.92 * uMargin);
    vec3 clsOffset = (uCamRight * position.x + uCamUp * position.y) * (uSigma * 0.02);
    vec3 world3 = mix(center + offset, clsCenter + clsOffset, aParam.w);
    gl_Position = uViewProj * vec4(world3, 1.0);
  }
}
`;

const FRAG = /* glsl */ `
precision highp float;
in vec2 vSig;
in vec3 vColor;
in float vOpacity;
in float vGlow;
in float vHalo;
in float vHighlight;
in float vIsCls;

uniform vec3 uGlowColor;
uniform vec3 uHaloColor;
uniform vec3 uHighlightColor;
uniform vec3 uClsColor;
uniform float uShimmer; // global liveness breath (NOT a data channel)

out vec4 frag;

void main() {
  float d2 = dot(vSig, vSig);
  float d = sqrt(d2);
  float g = exp(-0.5 * d2);

  // body: patch mean color, brightness ∝ activation. The scalar gains are a
  // fixed display exposure; uShimmer is a uniform breath applied identically to
  // every splat (legibility of "alive", encodes no data — §7 honesty rule).
  vec3 col = vColor * g * (0.55 + vOpacity * 1.3) * 1.7 * uShimmer;

  // additive emissive glow ∝ attribution
  col += uGlowColor * (g * g) * vGlow * 2.3;

  // soft halo ring ∝ attention-in
  float ring = exp(-pow(d - 2.1, 2.0) / (2.0 * 0.45 * 0.45));
  col += uHaloColor * ring * vHalo * 2.8;

  // selection: brighten + rim outline
  float rim = exp(-pow(d - 2.5, 2.0) / (2.0 * 0.22 * 0.22));
  col += uHighlightColor * (rim * 1.4 + g * 0.6) * step(0.5, vHighlight);

  // CLS gutter marker: distinct cool ring
  float m = exp(-pow(d - 1.6, 2.0) / (2.0 * 0.4 * 0.4));
  col += uClsColor * (m * 1.2 + g * 0.4) * step(0.5, vIsCls);

  // additive blend (ONE,ONE); discard near-black fragments
  if (max(col.r, max(col.g, col.b)) < 0.004) discard;
  frag = vec4(col, 1.0);
}
`;

const GRID_VERT = /* glsl */ `
precision highp float;
in vec3 position;
uniform mat4 uViewProj;
void main() { gl_Position = uViewProj * vec4(position, 1.0); }
`;

const GRID_FRAG = /* glsl */ `
precision highp float;
uniform vec3 uColor;
out vec4 frag;
void main() { frag = vec4(uColor, 1.0); }
`;

/** Build the XZ ground-grid line segments at Y=0 spanning ±margin. */
function buildGrid(margin: number, divisions: number): Float32Array {
  const lines: number[] = [];
  for (let i = 0; i <= divisions; i++) {
    const a = -margin + (2 * margin * i) / divisions;
    lines.push(-margin, 0, a, margin, 0, a); // along X
    lines.push(a, 0, -margin, a, 0, margin); // along Z
  }
  return new Float32Array(lines);
}

export class GaussianFieldRenderer {
  private renderer: THREE.WebGLRenderer;
  private scene = new THREE.Scene();
  private camera = new THREE.Camera(); // identity — shaders compute clip directly
  private geometry: THREE.BufferGeometry | null = null;
  private material: THREE.RawShaderMaterial;
  private gridMesh: THREE.LineSegments;

  private loaded: LoadedGaussians | null = null;
  private tokens = 0;
  private channels = 12;
  private verts = 0; // tokens * 4
  private aGeom: Float32Array | null = null; // [verts*4] x,y,rx,ry
  private aColT: Float32Array | null = null; // [verts*4] r,g,b,theta
  private aParam: Float32Array | null = null; // [verts*4] opacity,glow,halo,isCls
  private highlight: Float32Array | null = null; // [verts]
  private lastT = NaN;

  private width = 1;
  private height = 1;
  private startTime = performance.now();
  private lastFrame = performance.now();

  // -- 3D relief state --
  private mode3D = false;
  private camMath = new THREE.PerspectiveCamera(45, 1, 0.01, 100);
  private viewProj = new THREE.Matrix4();
  private camRight = new THREE.Vector3(1, 0, 0);
  private camUp = new THREE.Vector3(0, 1, 0);
  private orbitAzimuth = Math.PI * 0.15;
  private tiltFrac = 0; // 0 = top-down, 1 = full orbit tilt
  private tiltTarget = 0;
  private dragging = false;

  constructor(canvas: HTMLCanvasElement) {
    this.renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: true,
      alpha: false,
      powerPreference: "high-performance",
      preserveDrawingBuffer: true,
    });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.setClearColor(0x05060a, 1);

    this.material = new THREE.RawShaderMaterial({
      glslVersion: THREE.GLSL3,
      vertexShader: VERT,
      fragmentShader: FRAG,
      transparent: true,
      depthTest: false,
      depthWrite: false,
      blending: THREE.CustomBlending,
      blendEquation: THREE.AddEquation,
      blendSrc: THREE.OneFactor,
      blendDst: THREE.OneFactor,
      uniforms: {
        uSigma: { value: SIGMA_EXTENT },
        uFit: { value: new THREE.Vector2(1, 1) },
        uMargin: { value: FIELD_MARGIN },
        uMode: { value: 0 },
        uZGain: { value: RELIEF_Z_GAIN },
        uViewProj: { value: this.viewProj },
        uCamRight: { value: this.camRight },
        uCamUp: { value: this.camUp },
        uShimmer: { value: 1 },
        uGlowColor: { value: GLOW_COLOR },
        uHaloColor: { value: HALO_COLOR },
        uHighlightColor: { value: HIGHLIGHT_COLOR },
        uClsColor: { value: CLS_COLOR },
      },
    });

    // Ground grid (3D only) — additive faint lines sharing the view-projection.
    const gridGeo = new THREE.BufferGeometry();
    gridGeo.setAttribute("position", new THREE.BufferAttribute(buildGrid(FIELD_MARGIN, 10), 3));
    const gridMat = new THREE.RawShaderMaterial({
      glslVersion: THREE.GLSL3,
      vertexShader: GRID_VERT,
      fragmentShader: GRID_FRAG,
      transparent: true,
      depthTest: false,
      depthWrite: false,
      blending: THREE.CustomBlending,
      blendEquation: THREE.AddEquation,
      blendSrc: THREE.OneFactor,
      blendDst: THREE.OneFactor,
      uniforms: {
        uViewProj: { value: this.viewProj },
        uColor: { value: GRID_COLOR },
      },
    });
    this.gridMesh = new THREE.LineSegments(gridGeo, gridMat);
    this.gridMesh.frustumCulled = false;
    this.gridMesh.visible = false;
    this.scene.add(this.gridMesh);
  }

  /** Bind the field data and build the (non-instanced) 197-quad geometry. */
  setData(loaded: LoadedGaussians): void {
    this.loaded = loaded;
    this.tokens = loaded.tokens;
    this.channels = loaded.channelCount || 12;
    this.verts = this.tokens * 4;
    this.lastT = NaN;

    const pos = new Float32Array(this.verts * 3);
    for (let n = 0; n < this.tokens; n++) {
      for (let k = 0; k < 4; k++) {
        const v = (n * 4 + k) * 3;
        pos[v] = CORNERS[k][0];
        pos[v + 1] = CORNERS[k][1];
        pos[v + 2] = 0;
      }
    }
    const index = new Uint16Array(this.tokens * 6);
    for (let n = 0; n < this.tokens; n++) {
      const b = n * 4;
      const o = n * 6;
      index[o] = b;
      index[o + 1] = b + 1;
      index[o + 2] = b + 2;
      index[o + 3] = b + 2;
      index[o + 4] = b + 1;
      index[o + 5] = b + 3;
    }

    this.aGeom = new Float32Array(this.verts * 4);
    this.aColT = new Float32Array(this.verts * 4);
    this.aParam = new Float32Array(this.verts * 4);
    this.highlight = new Float32Array(this.verts);

    this.geometry?.dispose();
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    geo.setAttribute("aGeom", new THREE.BufferAttribute(this.aGeom, 4));
    geo.setAttribute("aColT", new THREE.BufferAttribute(this.aColT, 4));
    geo.setAttribute("aParam", new THREE.BufferAttribute(this.aParam, 4));
    geo.setAttribute("aHighlight", new THREE.BufferAttribute(this.highlight, 1));
    geo.setIndex(new THREE.BufferAttribute(index, 1));
    this.geometry = geo;

    // Keep the grid; only (re)create the splat mesh.
    const stale = this.scene.children.filter((c) => c !== this.gridMesh);
    for (const c of stale) this.scene.remove(c);
    const mesh = new THREE.Mesh(geo, this.material);
    mesh.frustumCulled = false;
    this.scene.add(mesh);

    this.fill(0);
  }

  /**
   * CPU-interpolate all tokens at time `t` into the vertex attribute buffers
   * (same value written to each token's 4 vertices) and flag for re-upload.
   */
  private fill(t: number): void {
    const l = this.loaded;
    if (!l || !this.aGeom || !this.aColT || !this.aParam || !this.geometry) return;
    const { data, steps, tokens } = l;
    const C = this.channels;
    const { s0, s1, f } = bracketSteps(t, steps);
    const base0 = s0 * tokens * C;
    const base1 = s1 * tokens * C;
    for (let n = 0; n < tokens; n++) {
      const o0 = base0 + n * C;
      const o1 = base1 + n * C;
      const li = (c: number) => lerp(data[o0 + c], data[o1 + c], f);
      const x = li(CH.x);
      const y = li(CH.y);
      const rx = li(CH.rx);
      const ry = li(CH.ry);
      const r = li(CH.r);
      const gg = li(CH.g);
      const b = li(CH.b);
      const theta = lerpAngle(data[o0 + CH.theta], data[o1 + CH.theta], f);
      const opacity = li(CH.opacity);
      const glow = li(CH.glow);
      const halo = li(CH.halo);
      const isCls = n === CLS_INDEX ? 1 : 0;
      for (let k = 0; k < 4; k++) {
        const q = (n * 4 + k) * 4;
        this.aGeom[q] = x;
        this.aGeom[q + 1] = y;
        this.aGeom[q + 2] = rx;
        this.aGeom[q + 3] = ry;
        this.aColT[q] = r;
        this.aColT[q + 1] = gg;
        this.aColT[q + 2] = b;
        this.aColT[q + 3] = theta;
        this.aParam[q] = opacity;
        this.aParam[q + 1] = glow;
        this.aParam[q + 2] = halo;
        this.aParam[q + 3] = isCls;
      }
    }
    (this.geometry.getAttribute("aGeom") as THREE.BufferAttribute).needsUpdate = true;
    (this.geometry.getAttribute("aColT") as THREE.BufferAttribute).needsUpdate = true;
    (this.geometry.getAttribute("aParam") as THREE.BufferAttribute).needsUpdate = true;
    this.lastT = t;
  }

  /** Re-upload the per-vertex highlight flags (only on selection change). */
  setHighlight(lit: Set<number>): void {
    if (!this.highlight || !this.geometry) return;
    for (let n = 0; n < this.tokens; n++) {
      const on = lit.has(n) ? 1 : 0;
      for (let k = 0; k < 4; k++) this.highlight[n * 4 + k] = on;
    }
    (this.geometry.getAttribute("aHighlight") as THREE.BufferAttribute).needsUpdate = true;
  }

  /** Set the timeline position; re-interpolates only if it changed. */
  setT(t: number): void {
    if (t !== this.lastT) this.fill(t);
  }

  /** Toggle 3D relief. Entering 3D animates the tilt from top-down → orbit. */
  setMode3D(on: boolean): void {
    if (on === this.mode3D) return;
    this.mode3D = on;
    this.material.uniforms.uMode.value = on ? 1 : 0;
    this.gridMesh.visible = on;
    if (on) {
      // "dissolve into terrain": start near-flat (top-down) and tilt up.
      this.tiltFrac = 0;
      this.tiltTarget = 1;
    }
  }

  isMode3D(): boolean {
    return this.mode3D;
  }

  /** Pointer-drag orbit (3D): dx spins azimuth, dy adjusts the tilt. */
  dragOrbit(dx: number, dy: number): void {
    this.orbitAzimuth += dx * 0.006;
    this.tiltTarget = Math.min(1, Math.max(0, this.tiltTarget - dy * 0.004));
  }

  setDragging(on: boolean): void {
    this.dragging = on;
  }

  /**
   * 3D hover/selection pick: project every splat center with the live
   * view-projection and return the token idx nearest the pointer on screen
   * (relief.ts `nearestScreenIndex`). Chosen over an offscreen ID buffer — no
   * GPU readback — and over the 2D analytic ellipse test, which is plane-only.
   */
  pick3D(px: number, py: number, rectW: number, rectH: number): number {
    if (!this.aGeom || !this.aParam) return -1;
    const zGain = this.material.uniforms.uZGain.value as number;
    const v = new THREE.Vector4();
    const pts: ScreenPoint[] = [];
    for (let n = 1; n < this.tokens; n++) {
      // n === CLS_INDEX (0) is parked off-terrain — excluded.
      const q = n * 4 * 4; // token n's first vertex, 4 floats/vertex
      const gx = (this.aGeom[q] * 2 - 1) * FIELD_MARGIN;
      const gz = (this.aGeom[q + 1] * 2 - 1) * FIELD_MARGIN;
      const gy = zGain * this.aParam[q + 1]; // height = glow (matches shader)
      v.set(gx, gy, gz, 1).applyMatrix4(this.viewProj);
      if (v.w <= 0) continue;
      const sx = (v.x / v.w * 0.5 + 0.5) * rectW;
      const sy = (1 - (v.y / v.w * 0.5 + 0.5)) * rectH;
      pts.push({ idx: n, x: sx, y: sy });
    }
    return nearestScreenIndex(pts, px, py, PICK_RADIUS_PX);
  }

  /** Resize the drawing buffer and recompute square-aspect fit. */
  resize(width: number, height: number): void {
    this.width = Math.max(1, width);
    this.height = Math.max(1, height);
    this.renderer.setSize(this.width, this.height, false);
    const aspect = this.width / this.height;
    const fit = this.material.uniforms.uFit.value as THREE.Vector2;
    if (aspect >= 1) fit.set(1 / aspect, 1);
    else fit.set(1, aspect);
  }

  /** Advance the perspective orbit and refresh the view-projection uniforms. */
  private updateCamera(dt: number): void {
    if (!this.dragging) this.orbitAzimuth += AUTO_ORBIT_SPEED * dt;
    this.tiltFrac += (this.tiltTarget - this.tiltFrac) * Math.min(1, dt * 3.5);
    const polar = TOPDOWN_POLAR + (ORBIT_POLAR - TOPDOWN_POLAR) * easeInOut(this.tiltFrac);
    const eye = orbitEye(ORBIT_RADIUS, this.orbitAzimuth, polar, ORBIT_CENTER);
    this.camMath.position.set(eye[0], eye[1], eye[2]);
    this.camMath.up.set(0, 1, 0);
    this.camMath.lookAt(ORBIT_CENTER[0], ORBIT_CENTER[1], ORBIT_CENTER[2]);
    this.camMath.aspect = this.width / this.height;
    this.camMath.updateProjectionMatrix();
    this.camMath.updateMatrixWorld();
    this.viewProj.multiplyMatrices(this.camMath.projectionMatrix, this.camMath.matrixWorldInverse);
    this.camRight.setFromMatrixColumn(this.camMath.matrixWorld, 0);
    this.camUp.setFromMatrixColumn(this.camMath.matrixWorld, 1);
  }

  render(): void {
    if (!this.geometry) {
      this.renderer.clear();
      return;
    }
    const now = performance.now();
    const dt = Math.min((now - this.lastFrame) / 1000, 0.05);
    this.lastFrame = now;

    // Idle micro-motion: a subtle global breath (legibility, not data).
    this.material.uniforms.uShimmer.value =
      1 + 0.05 * Math.sin((now - this.startTime) / 900);

    if (this.mode3D) this.updateCamera(dt);
    this.renderer.render(this.scene, this.camera);
  }

  dispose(): void {
    this.geometry?.dispose();
    this.material.dispose();
    (this.gridMesh.geometry as THREE.BufferGeometry).dispose();
    (this.gridMesh.material as THREE.Material).dispose();
    this.renderer.dispose();
  }
}
