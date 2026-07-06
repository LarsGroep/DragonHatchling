/**
 * GaussianFieldRenderer — the flagship WebGL view (§7). three.js (pinned dep)
 * quads with an analytic Gaussian-falloff fragment shader and additive glow, on
 * the near-black instrument canvas.
 *
 * GEOMETRY: NON-INSTANCED BY DESIGN
 * --------------------------------
 * The field is 197 quads baked into ONE ordinary mesh (197×4 = 788 vertices,
 * 197×6 indices). We deliberately do NOT use instanced rendering: the software
 * GL used for headless verification (SwiftShader) mis-handles instanced draws
 * with a RawShaderMaterial (only instance 0 rasterizes / gl_InstanceID unusable),
 * which would make the whole field invisible in CI. 788 vertices is trivially
 * 60 fps, so the non-instanced path costs nothing and renders identically on
 * every GL. Each token's 12 channels are written to its 4 quad vertices.
 *
 * INTERPOLATION LOCATION / PERFORMANCE
 * ------------------------------------
 * Each frame the 12 channels of every token are LERP'd between the two
 * bracketing timeline steps ON THE CPU (interp.ts — the same pure, unit-tested
 * code the hit-test uses) into three compact vertex-attribute buffers, which are
 * re-uploaded (~38 KB/frame; skipped entirely while `t` is unchanged). We do not
 * push all 13 steps into a data texture and interpolate in the vertex shader:
 * vertex-texture-fetch also proved unreliable under SwiftShader, and the channel
 * math is unit-tested independently of the GPU either way. See the M6 report.
 *
 * The per-vertex `aHighlight` attribute is uploaded only when the store's
 * hover/pinned selection changes (user-paced, not per-frame).
 *
 * CLS (token 0) is drawn as a fixed corner gutter marker (a per-vertex flag),
 * not on the image plane.
 */
import * as THREE from "three";
import type { LoadedGaussians } from "@/src/lib/pack/types";
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
uniform vec2 uFit;   // square-aspect correction
uniform float uMargin;

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
  vec2 world = aGeom.xy + rot;
  vec2 clipField = vec2(world.x * 2.0 - 1.0, 1.0 - world.y * 2.0) * uMargin;
  // CLS: fixed small marker in the bottom-left gutter, off the image plane.
  vec2 clipCls = vec2(-0.92, -0.92) + position.xy * 0.055;
  vec2 clip = mix(clipField, clipCls, aParam.w);
  gl_Position = vec4(clip * uFit, 0.0, 1.0);
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

out vec4 frag;

void main() {
  float d2 = dot(vSig, vSig);
  float d = sqrt(d2);
  float g = exp(-0.5 * d2);

  // body: patch mean color, brightness ∝ activation (opacity channel). The
  // scalar gains are a fixed display exposure — relative brightness stays
  // proportional to the measured channels (the §7 honesty rule).
  vec3 col = vColor * g * (0.55 + vOpacity * 1.3) * 1.7;

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

export class GaussianFieldRenderer {
  private renderer: THREE.WebGLRenderer;
  private scene = new THREE.Scene();
  private camera = new THREE.Camera();
  private geometry: THREE.BufferGeometry | null = null;
  private material: THREE.RawShaderMaterial;

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
        uGlowColor: { value: GLOW_COLOR },
        uHaloColor: { value: HALO_COLOR },
        uHighlightColor: { value: HIGHLIGHT_COLOR },
        uClsColor: { value: CLS_COLOR },
      },
    });
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

    this.scene.clear();
    const mesh = new THREE.Mesh(geo, this.material);
    mesh.frustumCulled = false;
    this.scene.add(mesh);

    this.fill(0);
  }

  /**
   * CPU-interpolate all tokens at time `t` into the vertex attribute buffers
   * (same value written to each token's 4 vertices) and flag for re-upload.
   * O(tokens); allocation-free.
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

  render(): void {
    if (!this.geometry) {
      this.renderer.clear();
      return;
    }
    this.renderer.render(this.scene, this.camera);
  }

  dispose(): void {
    this.geometry?.dispose();
    this.material.dispose();
    this.renderer.dispose();
  }
}
