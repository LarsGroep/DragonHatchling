/**
 * HTTP Range fetch helper (§5, §15). The manifest asset index carries exact
 * byte sizes (and quant offsets), so a view can fetch exactly the bytes it
 * needs via an HTTP `Range` request against Supabase public storage
 * (`Access-Control-Allow-Origin: *`, honors Range → 206). HF `resolve/`
 * 302-redirects to a range-capable CDN, so redirects are followed.
 *
 * Robustness rule (per handoff): if the server does NOT answer with 206, fall
 * back to the full body and slice locally, so static hosts (and the mock's
 * Next public dir) that ignore Range still work.
 */

export interface RangeResult {
  buffer: ArrayBuffer;
  /** True when the server honored the Range request (206). */
  partial: boolean;
}

/**
 * Fetch `[start, end)` bytes of `url`. When `start`/`end` are omitted the whole
 * object is fetched. Always returns exactly the requested slice, whether the
 * server honored Range (206) or returned the full body (200).
 */
export async function fetchRange(
  url: string,
  start?: number,
  end?: number,
  fetchImpl: typeof fetch = fetch,
): Promise<RangeResult> {
  const wantsRange = start !== undefined && end !== undefined && end > start;
  const headers: Record<string, string> = {};
  if (wantsRange) headers["Range"] = `bytes=${start}-${end! - 1}`;

  const res = await fetchImpl(url, {
    headers,
    redirect: "follow",
  });
  if (!res.ok && res.status !== 206) {
    throw new Error(`fetch ${url} failed: ${res.status} ${res.statusText}`);
  }

  const body = await res.arrayBuffer();
  if (wantsRange && res.status === 206) {
    return { buffer: body, partial: true };
  }
  // Fallback: server ignored Range (200) — slice locally.
  if (wantsRange) {
    return { buffer: body.slice(start!, end!), partial: false };
  }
  return { buffer: body, partial: res.status === 206 };
}
