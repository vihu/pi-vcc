/**
 * Minimal client for a local System One decision server (Von or Laya via
 * scripts/von-server.py and scripts/laya-server.py, both speaking the
 * /v1/systemone shape: a state plus typed questions, typed answers back).
 * The only place a network call lives in pi-vcc, and it only ever targets
 * the URL from settings.localModel. Every caller treats `undefined` as "use
 * the algorithmic path". The payload is never logged or written anywhere.
 */
export type ModelQuestion =
  | { type: "noul"; instructions: unknown; criteria?: { true?: unknown; false?: unknown } }
  | { type: "choice"; instructions: unknown; criteria: Record<string, unknown> };

export type ModelAnswer =
  | { type: "noul"; noul: number }
  | { type: "choice"; choice: string; probabilities: Record<string, number>; confidence: number };

export interface AskModelOptions {
  /** Server base URL, e.g. http://localhost:8000 (settings.localModel.url). */
  url: string;
  signal?: AbortSignal;
  timeoutMs?: number;
}

export type AskModel = (
  state: unknown,
  questions: Record<string, ModelQuestion>,
  opts: AskModelOptions,
) => Promise<Record<string, ModelAnswer> | undefined>;

const DEFAULT_TIMEOUT_MS = 4000;

// ponytail: process-wide pause, a timeout or 5xx pauses the model for PAUSE_MS
// so a server that is down costs one timeout per minute, not one per search.
const PAUSE_MS = 60_000;
let pausedUntil = 0;

export const askModel: AskModel = async (state, questions, opts) => {
  if (Date.now() < pausedUntil) return undefined;
  const timeout = AbortSignal.timeout(opts.timeoutMs ?? DEFAULT_TIMEOUT_MS);
  const signal = opts.signal ? AbortSignal.any([opts.signal, timeout]) : timeout;
  try {
    const res = await fetch(`${opts.url.replace(/\/$/, "")}/v1/systemone`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: "local", state, questions }),
      signal,
    });
    if (!res.ok) {
      if (res.status >= 500) pausedUntil = Date.now() + PAUSE_MS;
      return undefined;
    }
    const json = (await res.json()) as { answers?: unknown };
    return json.answers && typeof json.answers === "object"
      ? (json.answers as Record<string, ModelAnswer>)
      : undefined;
  } catch {
    // Our own timeout pauses the model; a caller's abort or a connection
    // refusal does not (the next search retries at once).
    if (timeout.aborted) pausedUntil = Date.now() + PAUSE_MS;
    return undefined;
  }
};

/** Test seam. */
export const resetModelPause = (): void => {
  pausedUntil = 0;
};
