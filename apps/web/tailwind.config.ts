import type { Config } from "tailwindcss";

/**
 * Light "brain-first" palette (UX-VISION-2). Clean, professional (Apple /
 * Notion / Linear / Obsidian): white page, light-gray panels, dark type, and a
 * restrained semantic accent set — soft blue = activity, green = confirmed
 * evidence, orange = intermediate activation, purple = latent space only.
 *
 * The token NAMES are kept from the v1 instrument palette (void/panel/edge/
 * readout/muted/signal + the four pane accents) so every existing className
 * keeps working; only their VALUES invert to the light design language.
 */
const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./src/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Surfaces.
        void: "#ffffff",       // page canvas (white)
        panel: "#f4f5f7",      // panel / card fill (light gray)
        "panel-hi": "#eceef2", // raised panel / bar
        edge: "#e3e6ec",       // hairline borders
        grid: "#eef1f5",       // barely-there grid lines
        // Type.
        readout: "#28313f",    // primary text (dark)
        muted: "#8b94a4",      // secondary / labels
        signal: "#0f1723",     // strongest text / near-black emphasis
        // Semantic accents (UX-VISION-2 §Design language).
        image: "#3b82f6",      // soft blue — activity (Image + Brain wayfinding)
        gauss: "#0d9488",      // teal — the sensory field
        graph: "#3b82f6",      // soft blue — the Brain
        latent: "#8b5cf6",     // purple — latent embeddings only
        evidence: "#22c55e",   // green — confirmed evidence
        warm: "#f59e0b",       // orange — intermediate activation
      },
      fontFamily: {
        sans: [
          "var(--font-sans)",
          "system-ui",
          "-apple-system",
          "Segoe UI",
          "Roboto",
          "Helvetica Neue",
          "Arial",
          "sans-serif",
        ],
        mono: [
          "var(--font-mono)",
          "ui-monospace",
          "SFMono-Regular",
          "Menlo",
          "Consolas",
          "monospace",
        ],
      },
      boxShadow: {
        // Soft, diffuse card lift (no neon glow).
        soft: "0 1px 2px rgba(15,23,35,0.04), 0 4px 16px -6px rgba(15,23,35,0.10)",
        glow: "0 0 24px -6px var(--tw-shadow-color)",
      },
    },
  },
  plugins: [],
};

export default config;
