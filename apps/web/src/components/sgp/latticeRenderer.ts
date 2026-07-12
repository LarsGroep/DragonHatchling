/**
 * SomLatticeRenderer — the SGP flagship WebGL view (docs/SGP-ARCHITECTURE.md §6).
 *
 * Renders the SOM's real neuron lattice as camera-facing Gaussian splats
 * (analytic falloff fragment shader, additive glow) plus its lattice edges as
 * additive lines, on the same near-dark instrument inset the Gaussian field
 * uses. The whole scene auto-orbits (drag to steer) so the 3-D structure reads
 * at a glance.
 *
 * LAYOUT = THE LATTICE (the honesty upgrade): every node's world position is a
 * pure mapping of its REAL grid coordinate (`lattice.ts latticeWorld`) — no
 * force simulation, no fitted layout. The encoder-depth axis points up and the
 * pane labels it "learned hierarchy".
 *
 * ENCODINGS (all measured):
 *   • body color      — community hue (seeded k-means over neuron weights)
 *   • body size       — sqrt(BMU hit count); dead neurons are tiny + gray
 *   • body brightness — this image's BMU activation at the scrubbed depth,
 *                       EMA-blended across frames for legibility (uniform
 *                       temporal filter; encodes nothing)
 *   • halo ring       — hover/selection highlight
 *   • edge alpha      — weight-space similarity (U-matrix ridges go dark)
 *
 * GEOMETRY: like the Gaussian field, K quads baked into ONE mesh (no
 * instancing — SwiftShader-safe) + one LineSegments for the edges. K ≤ 512 →
 * 2048 verts; trivially 60 fps.
 */
import * as THREE from "three";
import { orbitEye, nearestScreenIndex, type ScreenPoint } from "../views/GaussianField/relief";
import { easeInOut } from "@/src/lib/loop/schedule";
import { communityRgb, latticeWorld, sizeForHits } from "./lattice";
import type { SgpSom } from "@/src/lib/sgp";

// Orbit constants (tuned to frame a ±0.9 cube like the relief mode). A longer
// radius + narrower fov flattens the perspective so lattice rows stay parallel
// and near splats don't balloon.
const ORBIT_RADIUS = 4.1;
const ORBIT_CENTER: [number, number, number] = [0, 0, 0];
const ORBIT_POLAR = 1.08; // settled tilt
const TOPDOWN_POLAR = 0.35;
const AUTO_ORBIT_SPEED = 0.14; // rad/sec — the "always-alive" slow spin
const PICK_RADIUS_PX = 18;

// Splat world radius at size 1 (node size scales this).
const SPLAT_R = 0.062;
const DEAD_RGB: [number, number, number] = [0.42, 0.46, 0.52];
const HIGHLIGHT_COLOR = new THREE.Color(0xeaf1ff);
const EDGE_COLOR = new THREE.Color(0x5a708c);

const CORNERS = [
  [-1, -1],
  [1, -1],
  [-1, 1],
  [1, 1],
];

const VERT = /* glsl */ `
precision highp float;
in vec3 position;     // quad corner in [-1,1] (z unused)
in vec3 aCenter;      // world-space lattice position (REAL grid coords)
in vec4 aColor;       // community rgb + isDead
in vec3 aParam;       // size, activation, highlight

uniform mat4 uViewProj;
uniform vec3 uCamRight;
uniform vec3 uCamUp;
uniform float uSplatR;

out vec2 vSig;
out vec3 vColor;
out float vAct;
out float vHighlight;
out float vDead;

void main() {
  vColor = aColor.rgb;
  vDead = aColor.a;
  vAct = aParam.y;
  vHighlight = aParam.z;
  vSig = position.xy * 3.0;   // falloff domain: 3 sigma at the quad edge

  float r = uSplatR * aParam.x;
  vec3 offset = (uCamRight * position.x + uCamUp * position.y) * r * 3.0;
  gl_Position = uViewProj * vec4(aCenter + offset, 1.0);
}
`;

const FRAG = /* glsl */ `
precision highp float;
in vec2 vSig;
in vec3 vColor;
in float vAct;
in float vHighlight;
in float vDead;

uniform vec3 uHighlightColor;
uniform float uShimmer;   // global liveness breath (NOT a data channel)

out vec4 frag;

void main() {
  float d2 = dot(vSig, vSig);
  float d = sqrt(d2);
  float g = exp(-0.5 * d2);

  // body: community hue; brightness = resting glow + measured activation.
  float bright = (0.22 + vAct * 1.9) * uShimmer;
  vec3 col = vColor * g * bright;

  // activation bloom: hot neurons flare wider (g^0.5 widens the falloff).
  col += vColor * sqrt(g) * vAct * 0.55;

  // dead neurons: a faint cool ring, no body (shown, never hidden).
  float ring = exp(-pow(d - 2.0, 2.0) / (2.0 * 0.35 * 0.35));
  col = mix(col, vec3(0.30, 0.33, 0.38) * ring * 0.8, vDead);

  // hover/selection rim.
  float rim = exp(-pow(d - 2.55, 2.0) / (2.0 * 0.22 * 0.22));
  col += uHighlightColor * (rim * 1.5 + g * 0.5) * vHighlight;

  if (max(col.r, max(col.g, col.b)) < 0.004) discard;
  frag = vec4(col, 1.0);
}
`;

const EDGE_VERT = /* glsl */ `
precision highp float;
in vec3 position;
in float aEdgeW;      // similarity weight (measured)
uniform mat4 uViewProj;
out float vW;
void main() {
  vW = aEdgeW;
  gl_Position = uViewProj * vec4(position, 1.0);
}
`;

const EDGE_FRAG = /* glsl */ `
precision highp float;
in float vW;
uniform vec3 uColor;
uniform float uGain;
out vec4 frag;
void main() { frag = vec4(uColor * vW * uGain, 1.0); }
`;

export class SomLatticeRenderer {
  private renderer: THREE.WebGLRenderer;
  private scene = new THREE.Scene();
  private camera = new THREE.Camera(); // identity — shaders own the projection
  private camMath = new THREE.PerspectiveCamera(30, 1, 0.01, 100);
  private viewProj = new THREE.Matrix4();
  private camRight = new THREE.Vector3(1, 0, 0);
  private camUp = new THREE.Vector3(0, 1, 0);

  private material: THREE.RawShaderMaterial;
  private edgeMaterial: THREE.RawShaderMaterial;
  private geometry: THREE.BufferGeometry | null = null;
  private edgeGeometry: THREE.BufferGeometry | null = null;

  private centers: Float32Array | null = null; // [K*3] world positions
  private aParam: Float32Array | null = null; // per-vertex size/act/highlight
  private sizes: Float32Array | null = null; // per-neuron size
  private K = 0;

  private width = 1;
  private height = 1;
  private startTime = performance.now();
  private lastFrame = performance.now();
  private orbitAzimuth = Math.PI * 0.22;
  private tiltFrac = 0;
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
    // Same dark-slate instrument inset as the Gaussian field.
    this.renderer.setClearColor(0x0e1626, 1);

    const additive = {
      transparent: true,
      depthTest: false,
      depthWrite: false,
      blending: THREE.CustomBlending,
      blendEquation: THREE.AddEquation,
      blendSrc: THREE.OneFactor,
      blendDst: THREE.OneFactor,
    } as const;

    this.material = new THREE.RawShaderMaterial({
      glslVersion: THREE.GLSL3,
      vertexShader: VERT,
      fragmentShader: FRAG,
      ...additive,
      uniforms: {
        uViewProj: { value: this.viewProj },
        uCamRight: { value: this.camRight },
        uCamUp: { value: this.camUp },
        uSplatR: { value: SPLAT_R },
        uShimmer: { value: 1 },
        uHighlightColor: { value: HIGHLIGHT_COLOR },
      },
    });

    this.edgeMaterial = new THREE.RawShaderMaterial({
      glslVersion: THREE.GLSL3,
      vertexShader: EDGE_VERT,
      fragmentShader: EDGE_FRAG,
      ...additive,
      uniforms: {
        uViewProj: { value: this.viewProj },
        uColor: { value: EDGE_COLOR },
        uGain: { value: 0.34 },
      },
    });
  }

  /** Bind a SOM: build the K-quad splat mesh + the lattice edge lines. */
  setSom(som: SgpSom): void {
    const K = som.num_neurons;
    this.K = K;
    const grid = som.grid;

    this.centers = new Float32Array(K * 3);
    this.sizes = new Float32Array(K);
    let maxHits = 0;
    for (const n of som.nodes) if (n.hits > maxHits) maxHits = n.hits;

    const verts = K * 4;
    const pos = new Float32Array(verts * 3);
    const aCenter = new Float32Array(verts * 3);
    const aColor = new Float32Array(verts * 4);
    this.aParam = new Float32Array(verts * 3);

    for (const n of som.nodes) {
      const k = n.idx;
      const [wx, wy, wz] = latticeWorld(n.grid[0], n.grid[1], n.grid[2], grid);
      this.centers[k * 3] = wx;
      this.centers[k * 3 + 1] = wy;
      this.centers[k * 3 + 2] = wz;
      this.sizes[k] = sizeForHits(n.hits, maxHits);
      const [r, g, b] = n.dead ? DEAD_RGB : communityRgb(n.community);
      for (let c = 0; c < 4; c++) {
        const v = k * 4 + c;
        pos[v * 3] = CORNERS[c][0];
        pos[v * 3 + 1] = CORNERS[c][1];
        pos[v * 3 + 2] = 0;
        aCenter[v * 3] = wx;
        aCenter[v * 3 + 1] = wy;
        aCenter[v * 3 + 2] = wz;
        aColor[v * 4] = r;
        aColor[v * 4 + 1] = g;
        aColor[v * 4 + 2] = b;
        aColor[v * 4 + 3] = n.dead ? 1 : 0;
        this.aParam[v * 3] = this.sizes[k];
        this.aParam[v * 3 + 1] = 0; // activation (per frame)
        this.aParam[v * 3 + 2] = 0; // highlight
      }
    }

    const index = new Uint32Array(K * 6);
    for (let k = 0; k < K; k++) {
      const b = k * 4;
      const o = k * 6;
      index[o] = b;
      index[o + 1] = b + 1;
      index[o + 2] = b + 2;
      index[o + 3] = b + 2;
      index[o + 4] = b + 1;
      index[o + 5] = b + 3;
    }

    this.geometry?.dispose();
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    geo.setAttribute("aCenter", new THREE.BufferAttribute(aCenter, 3));
    geo.setAttribute("aColor", new THREE.BufferAttribute(aColor, 4));
    geo.setAttribute("aParam", new THREE.BufferAttribute(this.aParam, 3));
    geo.setIndex(new THREE.BufferAttribute(index, 1));
    this.geometry = geo;

    // Edges: two vertices per edge, weight replicated on both.
    const E = som.edges.length;
    const epos = new Float32Array(E * 2 * 3);
    const ew = new Float32Array(E * 2);
    for (let e = 0; e < E; e++) {
      const [a, b, w] = som.edges[e];
      for (const [slot, kk] of [
        [0, a],
        [1, b],
      ] as const) {
        const v = e * 2 + slot;
        epos[v * 3] = this.centers[kk * 3];
        epos[v * 3 + 1] = this.centers[kk * 3 + 1];
        epos[v * 3 + 2] = this.centers[kk * 3 + 2];
        ew[v] = w;
      }
    }
    this.edgeGeometry?.dispose();
    const egeo = new THREE.BufferGeometry();
    egeo.setAttribute("position", new THREE.BufferAttribute(epos, 3));
    egeo.setAttribute("aEdgeW", new THREE.BufferAttribute(ew, 1));
    this.edgeGeometry = egeo;

    // Rebuild the scene: edges under splats (both additive, order irrelevant,
    // but keeping edges first reads better on overdraw).
    this.scene.clear();
    const lines = new THREE.LineSegments(egeo, this.edgeMaterial);
    lines.frustumCulled = false;
    this.scene.add(lines);
    const mesh = new THREE.Mesh(geo, this.material);
    mesh.frustumCulled = false;
    this.scene.add(mesh);

    // Entry animation: tilt up from a flatter angle.
    this.tiltFrac = 0;
  }

  /** Upload per-neuron activation [K] in [0,1] (already EMA-blended by the view). */
  setActivation(act: Float32Array): void {
    if (!this.aParam || !this.geometry || act.length !== this.K) return;
    for (let k = 0; k < this.K; k++) {
      const a = act[k];
      for (let c = 0; c < 4; c++) this.aParam[(k * 4 + c) * 3 + 1] = a;
    }
    (this.geometry.getAttribute("aParam") as THREE.BufferAttribute).needsUpdate = true;
  }

  /** Highlight a set of neuron idxs (hover + selection). */
  setHighlight(lit: ReadonlySet<number>): void {
    if (!this.aParam || !this.geometry) return;
    for (let k = 0; k < this.K; k++) {
      const on = lit.has(k) ? 1 : 0;
      for (let c = 0; c < 4; c++) this.aParam[(k * 4 + c) * 3 + 2] = on;
    }
    (this.geometry.getAttribute("aParam") as THREE.BufferAttribute).needsUpdate = true;
  }

  /** Pointer-drag orbit: dx spins azimuth, dy nudges the tilt. */
  dragOrbit(dx: number, dy: number): void {
    this.orbitAzimuth += dx * 0.006;
    this.tiltFrac = Math.min(1.6, Math.max(0.15, this.tiltFrac - dy * 0.004));
  }

  setDragging(on: boolean): void {
    this.dragging = on;
  }

  /**
   * Hover/selection pick: project every live neuron center with the current
   * view-projection; nearest on screen within PICK_RADIUS_PX (same approach as
   * the Gaussian relief pick — no GPU readback).
   */
  pick(px: number, py: number, rectW: number, rectH: number): number {
    if (!this.centers) return -1;
    const v = new THREE.Vector4();
    const pts: ScreenPoint[] = [];
    for (let k = 0; k < this.K; k++) {
      v.set(this.centers[k * 3], this.centers[k * 3 + 1], this.centers[k * 3 + 2], 1).applyMatrix4(
        this.viewProj,
      );
      if (v.w <= 0) continue;
      pts.push({
        idx: k,
        x: (v.x / v.w * 0.5 + 0.5) * rectW,
        y: (1 - (v.y / v.w * 0.5 + 0.5)) * rectH,
      });
    }
    return nearestScreenIndex(pts, px, py, PICK_RADIUS_PX);
  }

  resize(width: number, height: number): void {
    this.width = Math.max(1, width);
    this.height = Math.max(1, height);
    this.renderer.setSize(this.width, this.height, false);
  }

  private updateCamera(dt: number): void {
    if (!this.dragging) this.orbitAzimuth += AUTO_ORBIT_SPEED * dt;
    // settle the entry tilt toward 1 (drag can push past it a little).
    if (this.tiltFrac < 1) this.tiltFrac += (1 - this.tiltFrac) * Math.min(1, dt * 2.2);
    const polar = TOPDOWN_POLAR + (ORBIT_POLAR - TOPDOWN_POLAR) * easeInOut(Math.min(1, this.tiltFrac));
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

    // Idle micro-motion: uniform breath (legibility, not data).
    this.material.uniforms.uShimmer.value = 1 + 0.05 * Math.sin((now - this.startTime) / 900);

    this.updateCamera(dt);
    this.renderer.render(this.scene, this.camera);
  }

  dispose(): void {
    this.geometry?.dispose();
    this.edgeGeometry?.dispose();
    this.material.dispose();
    this.edgeMaterial.dispose();
    this.renderer.dispose();
  }
}
