import { afterEach, beforeEach, describe, expect, it } from "bun:test";
import { askModel, resetModelPause } from "../src/core/decision-model";

const q = { ask: { type: "noul" as const, instructions: "?" } };
const realFetch = globalThis.fetch;

describe("askModel", () => {
  beforeEach(() => resetModelPause());
  afterEach(() => { globalThis.fetch = realFetch; });

  it("posts to <url>/v1/systemone without any auth header and returns the answers", async () => {
    let seen: { url: string; headers: Record<string, string> } | undefined;
    globalThis.fetch = (async (url: string, init: any) => {
      seen = { url, headers: init.headers };
      return new Response(JSON.stringify({ answers: { ask: { type: "noul", noul: 0.7 } } }), { status: 200 });
    }) as any;
    expect(await askModel("s", q, { url: "http://localhost:8000/" })).toEqual({ ask: { type: "noul", noul: 0.7 } });
    expect(seen!.url).toBe("http://localhost:8000/v1/systemone");
    expect(Object.keys(seen!.headers).map((k) => k.toLowerCase())).not.toContain("authorization");
  });

  it("returns undefined on 5xx and pauses further calls", async () => {
    let calls = 0;
    globalThis.fetch = (async () => { calls++; return new Response("boom", { status: 503 }); }) as any;
    expect(await askModel("s", q, { url: "http://localhost:8000" })).toBeUndefined();
    expect(await askModel("s", q, { url: "http://localhost:8000" })).toBeUndefined();
    expect(calls).toBe(1);
  });

  it("returns undefined when the server is unreachable, without pausing", async () => {
    let calls = 0;
    globalThis.fetch = (async () => { calls++; throw new Error("ECONNREFUSED"); }) as any;
    expect(await askModel("s", q, { url: "http://localhost:8000" })).toBeUndefined();
    expect(await askModel("s", q, { url: "http://localhost:8000" })).toBeUndefined();
    expect(calls).toBe(2);
  });
});
