import { defineConfig } from "vitest/config";
import { fileURLToPath } from "node:url";

/**
 * Vitest config for the pure data-layer units (fp16 decode, attention
 * dequantization, resolver). Node environment; aliases mirror tsconfig paths so
 * the `@vitreous/schema` type-only import and `@/…` resolve.
 */
export default defineConfig({
  resolve: {
    alias: {
      "@vitreous/schema": fileURLToPath(
        new URL("../../packages/schema/src/index.ts", import.meta.url),
      ),
      "@": fileURLToPath(new URL(".", import.meta.url)),
    },
  },
  test: {
    environment: "node",
    include: ["src/**/*.test.ts"],
  },
});
