/**
 * Recall eval with a self-supervised gold set (PLAN-001 M1/M2).
 *
 * Query  = first 5 meaningful terms of user turn N (leading 300 chars).
 * Gold   = assistant entries between turn N and N+1 that carry a tool call
 *          with a file path. The user wrote the query before that work
 *          existed, so query wording differs from gold by construction.
 * Rankers: bm25 (shipped `searchEntriesDetailed`, floor + cap as shipped) and
 *          model (the production `retrieveAndRank` against a local Von or Laya
 *          server, every entry scored, union with BM25 by rank fusion).
 *
 * Reports, per ranker: MRR, recall@5 (page 1), gold-in-shortlist rate, how
 * many BM25 misses were rescued, retrieval p50 latency, and paired per-query
 * deltas. Also query-term coverage in the first gold entry (bias check:
 * median > 0.8 means the paraphrase fixture is still needed). The source user
 * turn is excluded from ranks.
 *
 * Usage: [MODEL_URL=http://localhost:8000] [RETRIEVE_PROFILE=von|laya]
 *        bun scripts/benchmark-recall-rerank.ts [sessionCount] [queriesPerSession]
 */
import type { Message } from "@earendil-works/pi-ai";
import { renderMessage, type RenderedEntry } from "../src/core/render-entries";
import { searchEntriesDetailed, SEARCH_RESULT_CAP, type SearchHit } from "../src/core/search-entries";
import { textOf, extractToolCallArgsText } from "../src/core/content";
import { prepareSessionSamples } from "../tests/support/real-sessions";
import { loadSessionMessages } from "../tests/support/load-session";
import { askModel as askModelRaw, resetModelPause, type AskModel } from "../src/core/decision-model";
import { retrieveAndRank, RELEVANCE_CRITERIA, PROFILES, type ModelProfile } from "../src/core/rerank";

const SESSION_COUNT = Number(process.argv[2] ?? 25);
const PER_SESSION = Number(process.argv[3] ?? 20);
const QUERY_TERMS = 5;
// RETRIEVE_PROFILE picks the production shape (von | laya); the other
// RETRIEVE_* variables override single knobs for experiments.
const MODEL_URL = process.env.MODEL_URL ?? "http://localhost:8000";
const PROFILE = (process.env.RETRIEVE_PROFILE ?? "von") as ModelProfile;
const base = PROFILES[PROFILE] as { chunk: number; shape?: "noul" | "composite"; criteria?: typeof RELEVANCE_CRITERIA };
const CHUNK = Number(process.env.RETRIEVE_CHUNK ?? base.chunk);
const CRITERIA = process.env.RETRIEVE_CRITERIA ? (process.env.RETRIEVE_CRITERIA === "1" ? RELEVANCE_CRITERIA : undefined) : base.criteria;
const SHAPE = (process.env.RETRIEVE_SHAPE ?? base.shape ?? "noul") as "noul" | "composite";
const FUSION = process.env.RETRIEVE_FUSION === "noul" ? "noul" as const : "rrf" as const;
const ADMIT = process.env.RETRIEVE_ADMIT ? Number(process.env.RETRIEVE_ADMIT) : undefined;

const STOPWORDS = new Set([
  "the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
  "do", "does", "did", "will", "would", "could", "should", "may", "might", "can", "shall",
  "of", "in", "to", "for", "with", "on", "at", "from", "by", "as", "into", "through", "and",
  "that", "this", "what", "which", "who", "you", "your", "i", "we", "it", "its", "please",
  "just", "also", "then", "here", "there", "make", "sure", "want", "need", "like", "about",
]);

const terms = (text: string): string[] =>
  (text.toLowerCase().match(/[a-z][a-z0-9_-]{3,}/g) ?? []).filter((t) => !STOPWORDS.has(t));

const hayOf = (msg: Message): string => {
  const parts = [textOf(msg.content)];
  if (msg.content && typeof msg.content !== "string") {
    for (const p of msg.content) if (p.type === "toolCall") parts.push(extractToolCallArgsText(p.arguments));
  }
  return parts.join("\n").toLowerCase();
};

interface Query { label: string; text: string; source: number; gold: Set<number>; coverage: number }

const queriesOf = (label: string, messages: Message[], rendered: RenderedEntry[]): Query[] => {
  const out: Query[] = [];
  const userIdx = messages.map((m, i) => (m.role === "user" ? i : -1)).filter((i) => i >= 0);
  for (let k = 0; k < userIdx.length && out.length < PER_SESSION; k++) {
    const n = userIdx[k];
    const end = userIdx[k + 1] ?? messages.length;
    const qTerms = [...new Set(terms(textOf(messages[n].content).slice(0, 300)))].slice(0, QUERY_TERMS);
    if (qTerms.length < 3) continue;
    const gold = new Set<number>();
    for (let i = n + 1; i < end; i++) if (rendered[i].role === "assistant" && rendered[i].files?.length) gold.add(i);
    if (gold.size === 0) continue;
    const firstGoldHay = hayOf(messages[[...gold][0]]);
    const coverage = qTerms.filter((t) => firstGoldHay.includes(t)).length / qTerms.length;
    out.push({ label, text: qTerms.join(" "), source: n, gold, coverage });
  }
  return out;
};

/**
 * Bench-only wrapper: production askModel pauses itself for a minute on the
 * first 5xx/timeout (fine for one search, fatal for a sweep). Reset and retry
 * with backoff instead.
 */
const askModel: AskModel = async (state, questions, opts) => {
  for (let attempt = 0; attempt < 4; attempt++) {
    const r = await askModelRaw(state, questions, opts);
    if (r) return r;
    resetModelPause();
    await new Promise((res) => setTimeout(res, 500 * 2 ** attempt));
  }
  return undefined;
};

type Ranker = (query: string, hits: SearchHit[], rendered: RenderedEntry[]) => SearchHit[] | Promise<SearchHit[]>;
const RANKERS: Record<string, Ranker> = {
  bm25: (_q, hits) => hits,
  model: (q, hits, rendered) => retrieveAndRank(q, rendered, hits, askModel, { url: MODEL_URL, chunk: CHUNK, timeoutMs: 60000, cap: SEARCH_RESULT_CAP, criteria: CRITERIA, shape: SHAPE, fusion: FUSION, admit: ADMIT }),
};

const median = (xs: number[]): number => {
  const s = [...xs].sort((a, b) => a - b);
  return s.length ? s[Math.floor(s.length / 2)] : 0;
};

async function main() {
  const samples = await prepareSessionSamples(SESSION_COUNT);
  const corpus: { q: Query; rendered: RenderedEntry[]; messages: Message[] }[] = [];
  for (const sample of samples) {
    let loaded;
    try { loaded = loadSessionMessages(sample.copy); } catch { continue; }
    const rendered = loaded.messages.map((m, i) => renderMessage(m, i));
    const label = sample.source.split("/").slice(-2).join("/");
    for (const q of queriesOf(label, loaded.messages, rendered)) corpus.push({ q, rendered, messages: loaded.messages });
  }
  const n = corpus.length || 1;
  const sessions = new Set(corpus.map((c) => c.q.label)).size;
  console.log(`sessions ${sessions}, queries ${corpus.length}, median query-term coverage in first gold ${median(corpus.map((c) => c.q.coverage)).toFixed(2)}`);
  const pools = corpus.map((c) => c.rendered.filter((e) => e.role !== "tool_result").length);
  console.log(`scorable entries per query: median ${median(pools)}, max ${Math.max(...pools)}; profile ${PROFILE}, chunk ${CHUNK}, shape ${SHAPE}, criteria ${Boolean(CRITERIA)}, fusion ${FUSION}, admit ${ADMIT ?? 0.2}; server ${MODEL_URL}`);

  const rrBy: Record<string, number[]> = {};
  const bm25Miss: boolean[] = [];
  for (const [name, ranker] of Object.entries(RANKERS)) {
    const rrs: number[] = [];
    const latency: number[] = [];
    let at5 = 0, inShort = 0;
    for (let i = 0; i < corpus.length; i++) {
      const { q, rendered, messages } = corpus[i];
      const { hits } = searchEntriesDetailed(rendered, messages, q.text);
      if (name === "bm25") bm25Miss.push(!hits.some((h) => q.gold.has(h.index)));
      const t0 = performance.now();
      const ranked = (await ranker(q.text, hits, rendered)).filter((h) => h.index !== q.source);
      latency.push(performance.now() - t0);
      const rank = ranked.findIndex((h) => q.gold.has(h.index));
      rrs.push(rank >= 0 ? 1 / (rank + 1) : 0);
      if (rank >= 0) { inShort++; if (rank < 5) at5++; }
    }
    rrBy[name] = rrs;
    const mrr = rrs.reduce((a, b) => a + b, 0) / n;
    const missIdx = bm25Miss.map((m, i) => (m ? i : -1)).filter((i) => i >= 0);
    const rescued = missIdx.filter((i) => rrs[i] > 0).length;
    console.log(`${name.padEnd(5)} MRR ${mrr.toFixed(3)}, recall@5 ${(at5 / n).toFixed(3)}, gold-in-shortlist ${(inShort / n).toFixed(3)}, bm25-misses rescued ${rescued}/${missIdx.length}, p50 ms ${Math.round(median(latency))}`);
  }
  // Paired per-query deltas against bm25 (the corpus is small; counts beat means).
  const d = rrBy.model.map((v, i) => v - rrBy.bm25[i]);
  const up = d.filter((x) => x > 1e-9).length, down = d.filter((x) => x < -1e-9).length;
  console.log(`model vs bm25: improved ${up}, worse ${down}, same ${n - up - down}, mean dRR ${(d.reduce((a, b) => a + b, 0) / n).toFixed(3)}`);
}

main();
