import { appendFileSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { ToolLoopDetector } from "./loop_detector.mjs";

// Explicitly loaded by the E runner. It never changes model text or tool
// results, except for the optional budget gates below: those append a bracketed
// reminder to a tool result, or block further tool calls so the role must
// deliver. Every gate is off unless its environment variable is set, so earlier
// experiment configurations are unaffected. All firings are recorded.
export default function (pi) {
  const output = process.env.E_ARTIFACT_DIR;
  const limit = Number(process.env.E_MAX_TOOL_CALLS);
  const loopEnabled = process.env.E_LOOP_DETECTION !== "0";
  const loopRepeatLimit = Number(process.env.E_LOOP_REPEAT_LIMIT || 8);
  const loopMaxCycleLength = Number(process.env.E_LOOP_MAX_CYCLE_LENGTH || 8);
  const loopActionRepeatLimit = Number(process.env.E_LOOP_ACTION_REPEAT_LIMIT || 0);
  const contextWindow = Number(process.env.E_CONTEXT_WINDOW || 0);
  const softFraction = Number(process.env.E_BUDGET_SOFT_FRACTION || 0);
  const hardFraction = Number(process.env.E_BUDGET_HARD_FRACTION || 0);
  const fileSegmentLimit = Number(process.env.E_FILE_SEGMENT_LIMIT || 0);
  // A role that keeps reissuing a gate-blocked call is not recovering; stop it
  // instead of letting it spin until the wall-clock timeout.
  const gateBlockLimit = Number(process.env.E_GATE_BLOCK_LIMIT || 6);
  // Calls to a tool the role does not have never reach tool_call, so neither the
  // budget nor the loop detector sees them; stop after this many such turns in a row.
  const unknownToolTurnLimit = Number(process.env.E_UNKNOWN_TOOL_TURN_LIMIT || 0);
  const sampling = JSON.parse(readFileSync(`${output}/inference-config.json`, "utf8"));
  const budgetEvents = [];
  let toolCalls = 0;
  let loopDetected = false;
  let softFired = false;
  let hardFired = false;
  // Provider-reported prompt tokens of the last turn, plus a local estimate of
  // what has been added since, so a single large read cannot overshoot the gate.
  let reportedInputTokens = 0;
  let addedCharsSinceReport = 0;
  const fileSegments = new Map();
  const gateBlocks = new Map();
  let unknownToolTurns = 0;
  const detector = new ToolLoopDetector({
    repeatLimit: loopRepeatLimit,
    maxCycleLength: loopMaxCycleLength,
    actionRepeatLimit: loopActionRepeatLimit,
  });

  const estimatedTokens = () => reportedInputTokens + Math.ceil(addedCharsSinceReport / 4);

  // Blocking a call keeps the role alive, so count the attempts per gate and
  // hand back a terminate once the guidance has clearly not been taken. Only
  // the first few attempts of each gate are recorded individually.
  function gateBlock(gate, reason, details = {}) {
    const attempts = (gateBlocks.get(gate) ?? 0) + 1;
    gateBlocks.set(gate, attempts);
    if (attempts <= 3 || attempts === gateBlockLimit) {
      recordBudgetEvent(gate, { ...details, attempt: attempts });
    }
    if (gateBlockLimit > 0 && attempts >= gateBlockLimit) {
      recordBudgetEvent(`${gate}_not_followed`, { ...details, attempts });
      return { block: true, terminate: true, reason: `${reason} (no change after ${attempts} attempts)` };
    }
    return { block: true, reason };
  }

  function recordBudgetEvent(gate, details) {
    budgetEvents.push({
      gate,
      at: new Date().toISOString(),
      tool_calls_completed: toolCalls,
      estimated_context_tokens: estimatedTokens(),
      reported_input_tokens: reportedInputTokens,
      context_window: contextWindow,
      ...details,
    });
    writeFileSync(
      `${output}/budget-events.json`,
      JSON.stringify(
        {
          schema_version: "e-budget-control-1",
          context_window: contextWindow,
          soft_fraction: softFraction,
          hard_fraction: hardFraction,
          file_segment_limit: fileSegmentLimit,
          gate_block_limit: gateBlockLimit,
          unknown_tool_turn_limit: unknownToolTurnLimit,
          loop_action_repeat_limit: loopActionRepeatLimit,
          gate_blocked_attempts: Object.fromEntries(gateBlocks),
          events: budgetEvents,
        },
        null,
        2,
      ),
    );
  }

  pi.on("before_provider_request", (event) => {
    const payload = { ...event.payload, ...sampling };
    appendFileSync(`${output}/provider-requests.jsonl`, `${JSON.stringify(payload)}\n`);
    return payload;
  });

  pi.on("turn_end", (event, ctx) => {
    const usage = event?.message?.usage;
    const input = Number(usage?.input ?? 0) + Number(usage?.cacheRead ?? 0);
    if (input > 0) {
      reportedInputTokens = input + Number(usage?.output ?? 0);
      addedCharsSinceReport = 0;
    }
    if (!unknownToolTurnLimit) return;
    const results = event?.toolResults ?? [];
    const text = (result) => (result?.content ?? []).map((block) => block?.text ?? "").join("").trim();
    const onlyUnknown = results.length > 0 &&
      results.every((result) => result?.isError && /^Tool \S+ not found$/.test(text(result)));
    unknownToolTurns = onlyUnknown ? unknownToolTurns + 1 : 0;
    if (unknownToolTurns >= unknownToolTurnLimit) {
      recordBudgetEvent("unknown_tool_repeat", {
        turns: unknownToolTurns, tools: [...new Set(results.map((result) => result?.toolName))],
      });
      ctx?.abort?.();
    }
  });

  pi.on("tool_call", (event, ctx) => {
    const path = resolve(ctx.cwd, event.input.path || ".");
    if (path !== "/work/base" && !path.startsWith("/work/base/")) {
      return { block: true, reason: "Inspect only the frozen repository at /work/base." };
    }
    if (limit > 0 && toolCalls >= limit) {
      writeFileSync(`${output}/budget-exhausted.json`, JSON.stringify({ toolCalls, limit }));
      return { block: true, terminate: true, reason: "E exploration tool budget exhausted." };
    }

    // Gate 2: stop exploring and deliver. The call is blocked but the role is
    // not terminated, so its only remaining action is to return the Ledger.
    if (contextWindow > 0 && hardFraction > 0 && estimatedTokens() >= contextWindow * hardFraction) {
      hardFired = true;
      return gateBlock(
        "context_hard_stop",
        "[Budget exhausted] No further tool calls are available. Using the evidence you have " +
          "already read, return the Ledger JSON now, recording anything still uncertain in " +
          "open_questions.",
        { fraction: hardFraction },
      );
    }

    // Count every attempted call, including ones a gate below blocks, so a role
    // that keeps reissuing the same blocked call is still seen as repeating.
    const prepared = detector.prepare(event.toolCallId, event.toolName, event.input);

    // Gate 3: one file read in many successive windows fills the context while
    // every call differs, so the loop fingerprints never match.
    if (fileSegmentLimit > 0 && event.toolName === "read" && !(loopEnabled && prepared.hit)) {
      const segments = fileSegments.get(path) ?? 0;
      if (segments >= fileSegmentLimit) {
        return gateBlock(
          "file_segment_limit",
          `[Budget] This file has already been read in ${segments} separate ranges. Use grep to ` +
            "locate the symbols you still need, or continue from the evidence you already have.",
          { path, segments, limit: fileSegmentLimit },
        );
      }
      fileSegments.set(path, segments + 1);
    }

    if (loopEnabled && prepared.hit && !loopDetected) {
      loopDetected = true;
      writeFileSync(
        `${output}/loop-detected.json`,
        JSON.stringify({
          schema_version: "e-loop-detection-1",
          detector: "action-and-result-fingerprint",
          repeat_limit: loopRepeatLimit,
          max_cycle_length: loopMaxCycleLength,
          action_repeat_limit: loopActionRepeatLimit,
          tool_calls_completed: toolCalls,
          detected_at: new Date().toISOString(),
          ...prepared.hit,
        }, null, 2),
      );
      return {
        block: true,
        terminate: true,
        reason: `E loop detected (${prepared.hit.type}, ${prepared.hit.threshold} repetitions).`,
      };
    }
    toolCalls += 1;
    writeFileSync(`${output}/tool-budget.json`, JSON.stringify({ toolCalls, limit }));
  });

  pi.on("tool_result", (event) => {
    detector.complete(event.toolCallId, event);
    const text = (event.content ?? [])
      .map((block) => (typeof block?.text === "string" ? block.text : ""))
      .join("");
    addedCharsSinceReport += text.length;

    // Gate 1: one reminder, appended to the observation the role is about to read.
    if (contextWindow > 0 && softFraction > 0 && !softFired && !hardFired &&
        estimatedTokens() >= contextWindow * softFraction) {
      softFired = true;
      const notice =
        `\n\n[Budget notice] About ${Math.round(softFraction * 100)}% of the context is used. ` +
        "Prioritise the evidence still missing rather than broad exploration, and once the " +
        "evidence supports the obligations, record remaining uncertainty in open_questions and " +
        "return the Ledger.";
      recordBudgetEvent("context_soft_notice", { fraction: softFraction });
      addedCharsSinceReport += notice.length;
      return { content: [...(event.content ?? []), { type: "text", text: notice }] };
    }
  });
}
