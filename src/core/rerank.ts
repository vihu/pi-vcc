import type { RenderedEntry } from "./render-entries";
import { SEARCH_RESULT_CAP, type SearchHit } from "./search-entries";
import type { AskModel, AskModelOptions, ModelAnswer, ModelQuestion } from "./decision-model";
import { clip } from "./content";

/** Per-candidate text cap, roughly 120 tokens. */
const CANDIDATE_CHARS = 480;
/**
 * Noul at or above which an entry BM25 did not find is admitted to the results.
 * Low on purpose: with rank fusion the admitted entries are ordered by the
 * model's rank, so a loose floor adds reach without adding noise at the top
 * (bench: 0.2 was best or equal for Von and Laya).
 */
const ADMIT_NOUL = 0.2;
/** Requests in flight per search; a single-process local server queues the rest. */
const MAX_IN_FLIGHT = 8;

/**
 * What one entry contributes to the model payload. Only user- and
 * assistant-authored text leaves the machine: tool results and bash output
 * are program output, which is where `cat .env` and tokens in URLs live.
 */
export const payloadText = (entry: RenderedEntry): string => {
  if (entry.role === "tool_result") return entry.summary.match(/^\[[^\]]*\]/)?.[0] ?? "[tool]";
  if (entry.role === "bash") return entry.summary.split("\n")[0];
  return clip(entry.summary, CANDIDATE_CHARS);
};

export interface RetrieveOptions extends AskModelOptions {
  /** Candidates per request: 30 for the batched Von server, 1 for Laya. */
  chunk?: number;
  /** Result cap after the union; defaults to the search cap. */
  cap?: number;
  /** Explicit true/false criteria on every noul (Laya needs them; Von does not). */
  criteria?: { true: string; false: string };
  /**
   * "noul": one relevance noul per entry. "composite": three narrower nouls
   * per entry (files, task, error) combined by max; better for both local
   * models, three times the passes.
   */
  shape?: "noul" | "composite";
  /** How the final order is built: reciprocal rank fusion with BM25 (default) or raw noul. */
  fusion?: "rrf" | "noul";
  /** Noul at or above which an entry BM25 did not find is admitted (default ADMIT_NOUL). */
  admit?: number;
}

/** Literal criteria for the relevance nouls. */
export const RELEVANCE_CRITERIA = {
  true: "The candidate is about the same subject as the query: the same files, feature, error, command, or task.",
  false: "The candidate is about a different subject, or is generic conversation with no specific subject.",
};

/** Composite shape: narrower questions per candidate (Von's composite_score pattern). */
const COMPOSITE_QUESTIONS = [
  "Does the candidate mention the same files, paths, or code identifiers as the query?",
  "Does the candidate work on the same task, feature, or bug as the query?",
  "Does the candidate report the same error, command, or test as the query?",
];

/**
 * Per-model request shape, measured on the recall bench (PLAN-001 exp. 3 to 5).
 * von: 30 candidates per request against scripts/von-server.py (which batches
 *      and resolves the `candidates[k]` references), composite questions.
 * laya: one entry per request against scripts/laya-server.py, composite
 *      questions with explicit criteria.
 */
export const PROFILES = {
  von: { chunk: 30, shape: "composite" },
  laya: { chunk: 1, shape: "composite", criteria: RELEVANCE_CRITERIA },
} as const satisfies Record<string, Omit<RetrieveOptions, "url">>;
export type ModelProfile = keyof typeof PROFILES;

/**
 * One relevance score per non-tool_result entry, `chunk` entries per request,
 * chunks in parallel (bounded). Returns undefined if any request fails or any
 * answer is missing, so callers fall back to BM25 order.
 */
export const retrieveNouls = async (
  query: string,
  entries: RenderedEntry[],
  ask: AskModel,
  opts: RetrieveOptions,
): Promise<Map<number, number> | undefined> => {
  const size = Math.max(1, Math.floor(opts.chunk ?? 30));
  const composite = opts.shape === "composite";
  const pool = entries.filter((e) => e.role !== "tool_result");
  const chunks: RenderedEntry[][] = [];
  for (let i = 0; i < pool.length; i += size) chunks.push(pool.slice(i, i + size));
  const results: Array<Record<string, ModelAnswer> | undefined> = new Array(chunks.length);
  let next = 0;
  const worker = async () => {
    while (next < chunks.length) {
      const i = next++;
      const chunk = chunks[i];
      const state = size === 1
        ? { query, candidate: payloadText(chunk[0]) }
        : { query, candidates: chunk.map(payloadText) };
      const questions: Record<string, ModelQuestion> = {};
      chunk.forEach((e, k) => {
        const ref = size === 1 ? "`candidate`" : `\`candidates[${k}]\``;
        if (composite) {
          COMPOSITE_QUESTIONS.forEach((text, j) => {
            questions[`e${e.index}_${j}`] = { type: "noul", instructions: text.replace("the candidate", ref).replace("the query", "`query`"), criteria: opts.criteria };
          });
        } else {
          questions[`e${e.index}`] = { type: "noul", instructions: `Is ${ref} about \`query\`?`, criteria: opts.criteria };
        }
      });
      results[i] = await ask(state, questions, opts);
    }
  };
  await Promise.all(Array.from({ length: Math.min(MAX_IN_FLIGHT, chunks.length) }, worker));

  const nouls = new Map<number, number>();
  for (let i = 0; i < chunks.length; i++) {
    const answers = results[i];
    if (!answers) return undefined;
    for (const e of chunks[i]) {
      const keys = composite ? COMPOSITE_QUESTIONS.map((_, j) => `e${e.index}_${j}`) : [`e${e.index}`];
      const vals: number[] = [];
      for (const key of keys) {
        const a = answers[key];
        if (a?.type !== "noul" || typeof a.noul !== "number") return undefined;
        vals.push(a.noul);
      }
      nouls.set(e.index, Math.max(...vals));
    }
  }
  return nouls;
};

/**
 * Union of the BM25 hits and the entries the model admits (noul >= admit),
 * ordered by reciprocal rank fusion of the BM25 order and the model order
 * (BM25 position breaks ties). Entries the model never saw (tool results)
 * keep their BM25 rank credit only. Returns `hits` unchanged on any failure.
 */
export const retrieveAndRank = async (
  query: string,
  entries: RenderedEntry[],
  hits: SearchHit[],
  ask: AskModel,
  opts: RetrieveOptions,
): Promise<SearchHit[]> => {
  const nouls = await retrieveNouls(query, entries, ask, opts);
  if (!nouls) return hits;
  const pos = new Map(hits.map((h, i) => [h.index, i]));
  const admit = opts.admit ?? ADMIT_NOUL;
  const extra = entries.filter((e) => !pos.has(e.index) && (nouls.get(e.index) ?? 0) >= admit);
  let score = (e: RenderedEntry) => nouls.get(e.index) ?? -1;
  if (opts.fusion !== "noul") {
    // Reciprocal rank fusion (k = 60): robust when the model's order is noisy.
    // Bench: turns 10 to 13 "worse than BM25" queries out of 37 into 1 to 3.
    const modelRank = new Map([...nouls.entries()].sort((a, b) => b[1] - a[1]).map(([i], r) => [i, r]));
    score = (e) => (pos.has(e.index) ? 1 / (60 + pos.get(e.index)!) : 0) + (modelRank.has(e.index) ? 1 / (60 + modelRank.get(e.index)!) : 0);
  }
  return [...hits, ...extra]
    .sort((a, b) => score(b) - score(a) || (pos.get(a.index) ?? Infinity) - (pos.get(b.index) ?? Infinity))
    .slice(0, opts.cap ?? SEARCH_RESULT_CAP);
};
