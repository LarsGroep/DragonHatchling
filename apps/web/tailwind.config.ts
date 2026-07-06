import type { Config } from "tailwindcss";

/**
 * Dark scientific-instrument palette (§1). Near-black canvas, luminous marks,
 * monospaced readouts. Colors are placeholders that later views (Gaussian
 * field, graph, embedding) will draw luminous marks against.
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
        // Instrument chrome.
        void: "#05060a",       // near-black page canvas
        panel: "#0b0e14",      // panel fill
        "panel-hi": "#111623", // raised panel / bar
        edge: "#1c2333",       // hairline borders
        grid: "#141b2b",       // subtle grid lines
        // Readout text.
        readout: "#c8d3e6",    // primary monospaced text
        muted: "#5c6b85",      // secondary / labels
        // Luminous accents (one per view, for wayfinding).
        image: "#4cc9f0",      // Image Space
        gauss: "#b5179e",      // Gaussian Feature Field
        graph: "#f0a04c",      // Interaction Graph
        latent: "#57e389",     // Latent Embeddings
        signal: "#e8eefc",     // hot highlight
      },
      fontFamily: {
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
        glow: "0 0 24px -6px var(--tw-shadow-color)",
      },
    },
  },
  plugins: [],
};

export default config;
