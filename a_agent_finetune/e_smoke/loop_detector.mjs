import { createHash } from "node:crypto";

function stable(value) {
  if (value === null || typeof value !== "object") return value;
  if (Array.isArray(value)) return value.map(stable);
  return Object.fromEntries(
    Object.keys(value)
      .sort()
      .map((key) => [key, stable(value[key])]),
  );
}

function digest(value) {
  const encoded = JSON.stringify(stable(value)) ?? "null";
  return createHash("sha256").update(encoded).digest("hex");
}

export class ToolLoopDetector {
  // actionRepeatLimit catches a repeated identical call whose result keeps
  // changing, which the action+result fingerprint below cannot see; 0 disables it.
  constructor({ repeatLimit = 8, maxCycleLength = 8, actionRepeatLimit = 0 } = {}) {
    this.repeatLimit = Math.max(2, Number(repeatLimit) || 8);
    this.maxCycleLength = Math.max(1, Number(maxCycleLength) || 8);
    this.actionRepeatLimit = Math.max(0, Number(actionRepeatLimit) || 0);
    this.history = [];
    this.pending = new Map();
    this.actionCounts = new Map();
  }

  prepare(toolCallId, toolName, input) {
    const actionKey = digest({ toolName, input });
    const seen = (this.actionCounts.get(actionKey) ?? 0) + 1;
    this.actionCounts.set(actionKey, seen);
    let hit = this._detect(actionKey);
    if (!hit && this.actionRepeatLimit && seen >= this.actionRepeatLimit) {
      hit = {
        type: "action_repeat",
        threshold: this.actionRepeatLimit,
        cycle_length: 1,
        action_key: actionKey,
        history: [{ tool_name: toolName, action_key: actionKey, occurrences: seen }],
      };
    }
    this.pending.set(toolCallId, { toolName, input, actionKey });
    return { actionKey, hit };
  }

  complete(toolCallId, result) {
    const pending = this.pending.get(toolCallId);
    if (!pending) return;
    this.pending.delete(toolCallId);
    this.history.push({
      ...pending,
      resultKey: digest({
        isError: Boolean(result?.isError),
        content: result?.content ?? [],
        details: result?.details ?? null,
      }),
    });
    const keep = this.repeatLimit * this.maxCycleLength * 2;
    if (this.history.length > keep) this.history.splice(0, this.history.length - keep);
  }

  _detect(actionKey) {
    const threshold = this.repeatLimit;
    if (this.history.length >= threshold) {
      const suffix = this.history.slice(-threshold);
      const resultKey = suffix[0].resultKey;
      if (suffix.every((record) => record.actionKey === actionKey && record.resultKey === resultKey)) {
        return {
          type: "consecutive",
          threshold,
          cycle_length: 1,
          action_key: actionKey,
          history: suffix.map((record) => ({ tool_name: record.toolName, action_key: record.actionKey })),
        };
      }
    }

    const maxLength = Math.min(this.maxCycleLength, Math.floor(this.history.length / threshold));
    for (let length = 2; length <= maxLength; length += 1) {
      const segment = this.history.slice(-threshold * length);
      const pattern = segment.slice(0, length);
      let repeated = true;
      for (let cycle = 1; cycle < threshold && repeated; cycle += 1) {
        for (let index = 0; index < length; index += 1) {
          const expected = pattern[index];
          const actual = segment[cycle * length + index];
          if (actual.actionKey !== expected.actionKey || actual.resultKey !== expected.resultKey) {
            repeated = false;
            break;
          }
        }
      }
      if (repeated && actionKey === pattern[0].actionKey) {
        return {
          type: "cycle",
          threshold,
          cycle_length: length,
          action_key: actionKey,
          history: pattern.map((record) => ({ tool_name: record.toolName, action_key: record.actionKey })),
        };
      }
    }
    return null;
  }
}
