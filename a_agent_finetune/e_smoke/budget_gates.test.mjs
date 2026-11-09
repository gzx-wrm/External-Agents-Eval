import assert from "node:assert";
import { mkdtempSync, readFileSync, existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { writeFileSync } from "node:fs";
import { ToolLoopDetector } from "./loop_detector.mjs";

// 1. action-only rule fires when results keep changing (the yaegi/termenv miss)
let d = new ToolLoopDetector({ repeatLimit: 8, maxCycleLength: 8, actionRepeatLimit: 16 });
let hit = null;
for (let i = 1; i <= 16 && !hit; i++) {
  const p = d.prepare(`c${i}`, "grep", { pattern: "x" });
  hit = p.hit;
  if (!hit) d.complete(`c${i}`, { content: [{ type: "text", text: `result ${i}` }] });
}
assert.equal(hit?.type, "action_repeat", "action repeat should fire");
assert.equal(hit.threshold, 16);

// 2. disabled by default, and differing calls never fire it
d = new ToolLoopDetector({ repeatLimit: 8, maxCycleLength: 8 });
for (let i = 1; i <= 40; i++) {
  assert.equal(d.prepare(`c${i}`, "grep", { pattern: "x" }).hit, null, "off by default");
  d.complete(`c${i}`, { content: [{ type: "text", text: `r${i}` }] });
}
d = new ToolLoopDetector({ repeatLimit: 8, maxCycleLength: 8, actionRepeatLimit: 16 });
for (let i = 1; i <= 40; i++) {
  assert.equal(d.prepare(`c${i}`, "read", { path: `/f${i}` }).hit, null, "distinct calls are fine");
  d.complete(`c${i}`, { content: [{ type: "text", text: "r" }] });
}

// 3. the gates, driven through a fake pi
const dir = mkdtempSync(join(tmpdir(), "gate-"));
writeFileSync(join(dir, "inference-config.json"), "{}");
process.env.E_ARTIFACT_DIR = dir;
process.env.E_MAX_TOOL_CALLS = "0";
process.env.E_CONTEXT_WINDOW = "1000";
process.env.E_BUDGET_SOFT_FRACTION = "0.5";
process.env.E_BUDGET_HARD_FRACTION = "0.75";
process.env.E_FILE_SEGMENT_LIMIT = "3";
const mod = await import("./experiment_options.ts?x=1");
const handlers = {};
mod.default({ on: (name, fn) => (handlers[name] = fn) });
const ctx = { cwd: "/work/base" };
const call = (input, toolName = "read") => handlers.tool_call({ toolName, toolCallId: String(Math.random()), input }, ctx);
const result = (chars) => handlers.tool_result({ toolCallId: "t", content: [{ type: "text", text: "x".repeat(chars) }] });

assert.equal(call({ path: "/work/base/a.py" }), undefined, "first call allowed");
assert.equal(result(100), undefined, "no notice below the soft gate");
// 2000 chars ~ 500 tokens = 50% of a 1000-token window
const notice = result(1900);
assert.ok(notice?.content?.at(-1).text.includes("Budget notice"), "soft notice appended");
assert.equal(result(100), undefined, "soft notice fires only once");
// push past 75%
const blocked = (result(1000), call({ path: "/work/base/a.py" }));
assert.ok(blocked.block && !blocked.terminate, "hard gate blocks without terminating");
assert.ok(blocked.reason.includes("return the Ledger JSON now"), "hard gate asks for the Ledger");

// 4. file-segment cap, isolated from the context gates
process.env.E_CONTEXT_WINDOW = "0";
const mod2 = await import("./experiment_options.ts?x=2");
const h2 = {};
mod2.default({ on: (name, fn) => (h2[name] = fn) });
const call2 = (p) => h2.tool_call({ toolName: "read", toolCallId: String(Math.random()), input: { path: p } }, ctx);
for (let i = 0; i < 3; i++) assert.equal(call2("/work/base/big.py"), undefined, `segment ${i} allowed`);
const capped = call2("/work/base/big.py");
assert.ok(capped.block && !capped.terminate && capped.reason.includes("separate ranges"), "segment cap blocks");
assert.equal(call2("/work/base/other.py"), undefined, "cap is per file");
assert.ok(h2.tool_call({ toolName: "grep", toolCallId: "g", input: { path: "/work/base/big.py" } }, ctx) === undefined, "cap only applies to read");
assert.ok(existsSync(join(dir, "budget-events.json")), "events recorded");

// 5. an ignored gate terminates instead of spinning, and the attempts are capped
process.env.E_CONTEXT_WINDOW = "0";
process.env.E_FILE_SEGMENT_LIMIT = "2";
process.env.E_GATE_BLOCK_LIMIT = "4";
process.env.E_LOOP_ACTION_REPEAT_LIMIT = "16";
const mod3 = await import("./experiment_options.ts?x=3");
const h3 = {};
mod3.default({ on: (name, fn) => (h3[name] = fn) });
const read3 = () => h3.tool_call({ toolName: "read", toolCallId: String(Math.random()), input: { path: "/work/base/b.py", offset: 1 } }, ctx);
assert.equal(read3(), undefined);
assert.equal(read3(), undefined);
let terminated = null;
for (let i = 0; i < 6 && !terminated?.terminate; i++) {
  const r = read3();
  assert.ok(r.block, "gate keeps blocking");
  terminated = r;
}
assert.ok(terminated.terminate, "ignored gate terminates");
assert.ok(terminated.reason.includes("no change after"), "termination explains why");
const ev3 = JSON.parse(readFileSync(join(dir, "budget-events.json")));
assert.ok(ev3.events.some((e) => e.gate === "file_segment_limit_not_followed"), "non-compliance recorded");
assert.ok(ev3.events.filter((e) => e.gate === "file_segment_limit").length <= 4, "recorded attempts are capped");
console.log("gate attempts:", ev3.gate_blocked_attempts);

// 6. turns that only call unavailable tools abort the role; one real call resets the count
process.env.E_UNKNOWN_TOOL_TURN_LIMIT = "3";
process.env.E_FILE_SEGMENT_LIMIT = "0";
const mod4 = await import("./experiment_options.ts?x=4");
const h4 = {};
mod4.default({ on: (name, fn) => (h4[name] = fn) });
let aborted = 0;
const ctx4 = { abort: () => (aborted += 1) };
const unknownTurn = () => h4.turn_end({ message: {}, toolResults: [
  { toolName: "command", isError: true, content: [{ type: "text", text: "Tool command not found" }] }] }, ctx4);
const realTurn = () => h4.turn_end({ message: {}, toolResults: [
  { toolName: "read", isError: false, content: [{ type: "text", text: "ok" }] }] }, ctx4);
unknownTurn(); unknownTurn(); realTurn(); unknownTurn(); unknownTurn();
assert.equal(aborted, 0, "a real call resets the count");
unknownTurn();
assert.equal(aborted, 1, "three unknown-only turns in a row abort");
console.log("gates:", JSON.parse(readFileSync(join(dir, "budget-events.json"))).events.map((e) => e.gate));
console.log("ALL GATE TESTS PASSED");
