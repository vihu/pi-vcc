import { describe, expect, it } from "bun:test";
import { retrieveAndRank, retrieveNouls, payloadText, PROFILES } from "../src/core/rerank";
import type { RenderedEntry } from "../src/core/render-entries";
import type { SearchHit } from "../src/core/search-entries";
import type { AskModel } from "../src/core/decision-model";

const entry = (index: number, role = "assistant", summary = `entry ${index}`): RenderedEntry => ({ index, role, summary });
const hit = (index: number, role = "assistant"): SearchHit => entry(index, role);
const URL = { url: "http://localhost:1" };
/** Fake model: answers every question with the noul for its entry index (and sub-question), recording states. */
const fake = (noul: (index: number, sub: number) => number, seen: unknown[] = []): AskModel => async (state, questions) => {
  seen.push(state);
  return Object.fromEntries(Object.keys(questions).map((k) => {
    const m = k.match(/^e(\d+)(?:_(\d+))?$/)!;
    return [k, { type: "noul" as const, noul: noul(Number(m[1]), Number(m[2] ?? 0)) }];
  }));
};

describe("retrieveAndRank", () => {
  it("fuses BM25 and model ranks: BM25 hits first, admitted misses after, by model order", async () => {
    const entries = [entry(0), entry(1), entry(2), entry(3), entry(4)];
    const hits = [hit(0), hit(1)];
    const nouls: Record<number, number> = { 0: 0.6, 1: 0.6, 2: 0.9, 3: 0.1, 4: 0.5 };
    const out = await retrieveAndRank("q", entries, hits, fake((i) => nouls[i]), URL);
    expect(out.map((h) => h.index)).toEqual([0, 1, 2, 4]); // 3 is below the admit floor
  });

  it("raw noul order when fusion is 'noul'", async () => {
    const entries = [entry(0), entry(1), entry(2)];
    const out = await retrieveAndRank("q", entries, [hit(0), hit(1)], fake((i) => [0.3, 0.9, 0.6][i]), { ...URL, fusion: "noul" });
    expect(out.map((h) => h.index)).toEqual([1, 2, 0]);
  });

  it("never sends tool_result text; those hits keep only their BM25 rank credit", async () => {
    const entries = [entry(0), entry(1, "tool_result", "[bash] SECRET=abc"), entry(2, "tool_result", "[read] TOKEN=x"), entry(3)];
    const hits = [hit(1, "tool_result"), hit(0), hit(2, "tool_result")];
    const seen: unknown[] = [];
    const out = await retrieveAndRank("q", entries, hits, fake((i) => (i === 3 ? 0.9 : 0.5), seen), URL);
    expect(out.map((h) => h.index)).toEqual([0, 1, 3, 2]);
    expect(JSON.stringify(seen)).not.toContain("SECRET");
    expect(JSON.stringify(seen)).not.toContain("TOKEN");
  });

  it("falls back to BM25 order when any request fails or an answer is missing", async () => {
    const entries = [entry(0), entry(1)];
    const hits = [hit(1), hit(0)];
    expect((await retrieveAndRank("q", entries, hits, async () => undefined, URL)).map((h) => h.index)).toEqual([1, 0]);
    const partial: AskModel = async () => ({ e0: { type: "noul", noul: 1 } });
    expect((await retrieveAndRank("q", entries, hits, partial, URL)).map((h) => h.index)).toEqual([1, 0]);
  });

  it("caps the union", async () => {
    const entries = Array.from({ length: 10 }, (_, i) => entry(i));
    expect(await retrieveAndRank("q", entries, [], fake(() => 0.9), URL)).toHaveLength(10);
    expect(await retrieveAndRank("q", entries, [], fake(() => 0.9), { ...URL, cap: 3 })).toHaveLength(3);
  });
});

describe("retrieveNouls", () => {
  it("plain shape: chunk-sized candidate arrays with one noul each", async () => {
    const entries = Array.from({ length: 65 }, (_, i) => entry(i));
    const seen: any[] = [];
    await retrieveNouls("q", entries, fake(() => 0.5, seen), { ...URL, chunk: 30 });
    expect(seen.map((s) => s.candidates.length)).toEqual([30, 30, 5]);
  });

  it("von profile: three questions per candidate, combined by max", async () => {
    const entries = [entry(0), entry(1)];
    let keys: string[] = [];
    const ask: AskModel = async (state, questions) => { keys = Object.keys(questions); return fake((i, j) => (i === 1 && j === 2 ? 0.9 : 0.1))(state, questions); };
    const nouls = await retrieveNouls("q", entries, ask, { ...URL, ...PROFILES.von });
    expect(keys).toEqual(["e0_0", "e0_1", "e0_2", "e1_0", "e1_1", "e1_2"]);
    expect(nouls!.get(0)).toBe(0.1);
    expect(nouls!.get(1)).toBe(0.9);
  });

  it("laya profile: one entry per request with explicit criteria", async () => {
    const seen: any[] = [];
    let question: any;
    const ask: AskModel = async (state, questions) => { seen.push(state); question = Object.values(questions)[0]; return fake(() => 0.5)(state, questions); };
    await retrieveNouls("redis retry", [entry(0, "user", "fix the cache")], ask, { ...URL, ...PROFILES.laya });
    expect(seen).toEqual([{ query: "redis retry", candidate: "fix the cache" }]);
    expect(question.instructions).toContain("`candidate`");
    expect(question.criteria.true).toBeTruthy();
  });
});

describe("payloadText", () => {
  it("drops program output and keeps authored text", () => {
    expect(payloadText(entry(0, "tool_result", "[read] const token = 'x'"))).toBe("[read]");
    expect(payloadText(entry(1, "bash", "$ cat .env\nSECRET=1"))).toBe("$ cat .env");
    expect(payloadText(entry(2, "user", "fix the redis retry"))).toBe("fix the redis retry");
  });
});
