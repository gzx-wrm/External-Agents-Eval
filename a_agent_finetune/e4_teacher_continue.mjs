// E4: continue a real student prefix with Pi's native agent loop and read-only tools.
import { appendFileSync, readFileSync, writeFileSync, realpathSync, renameSync } from 'node:fs';
import { resolve } from 'node:path';
import { isDeepStrictEqual } from 'node:util';
import { Agent } from '/opt/pi-runtime/node_modules/@earendil-works/pi-agent-core/dist/index.js';
import { ModelRuntime } from '/opt/pi-runtime/node_modules/@earendil-works/pi-coding-agent/dist/core/model-runtime.js';
import { createReadOnlyTools } from '/opt/pi-runtime/node_modules/@earendil-works/pi-coding-agent/dist/core/tools/index.js';

const output = '/work/output';
const seed = JSON.parse(readFileSync(`${output}/seed.json`, 'utf8'));
const config = JSON.parse(readFileSync(`${output}/config.json`, 'utf8'));
const dryRun = process.argv.includes('--dry-run');
const zeroUsage = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0,
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } };
const stamp = () => new Date().toISOString();
const append = (file, data) => appendFileSync(`${output}/${file}`, `${JSON.stringify(data)}\n`);
const save = (file, data) => {
  const path = `${output}/${file}`;
  writeFileSync(`${path}.tmp`, JSON.stringify(data, null, 2) + '\n');
  renameSync(`${path}.tmp`, path);
};
const text = content => typeof content === 'string' ? content : (content || []).map(b => b.text || '').join('');
const normalizeWire = m => ({ role: m.role, content: text(m.content),
  ...(m.tool_call_id ? { tool_call_id: m.tool_call_id } : {}),
  ...(m.tool_calls?.length ? { tool_calls: m.tool_calls.map(c => ({ id: c.id, name: c.function.name,
    arguments: typeof c.function.arguments === 'string' ? JSON.parse(c.function.arguments) : c.function.arguments })) } : {}) });

const runtime = await ModelRuntime.create({ authPath: `${output}/auth.json`,
  modelsPath: `${output}/models.json`, allowModelNetwork: false, refreshOnCreate: false });
const model = runtime.getModel('deepseek', config.model);
if (!model) throw new Error('Configured teacher model was not loaded');
const nativeTools = createReadOnlyTools('/work/base');
const tools = seed.tools.map(({ function: f }) => {
  const native = nativeTools.find(t => t.name === f.name);
  if (!native) throw new Error(`Tool not available: ${f.name}`);
  return { ...native, description: f.description, parameters: f.parameters };
});
const calls = new Map();
for (const m of seed.messages) for (const c of m.tool_calls || []) calls.set(c.id, c.function.name);
const prefix = seed.messages.slice(1).map((m, index) => {
  const timestamp = index;
  if (m.role === 'user') return { role: 'user', content: m.content, timestamp };
  if (m.role === 'tool') return { role: 'toolResult', toolCallId: m.tool_call_id,
    toolName: calls.get(m.tool_call_id), content: [{ type: 'text', text: m.content }],
    isError: false, timestamp };
  if (m.role !== 'assistant') throw new Error(`Unexpected prefix role: ${m.role}`);
  const content = m.content ? [{ type: 'text', text: m.content }] : [];
  for (const c of m.tool_calls || []) content.push({ type: 'toolCall', id: c.id,
    name: c.function.name, arguments: c.function.arguments });
  // Provider tags here are transport metadata only; provenance still identifies the student.
  return { role: 'assistant', content, api: model.api, provider: model.provider, model: model.id,
    usage: structuredClone(zeroUsage), stopReason: m.tool_calls?.length ? 'toolUse' : 'stop', timestamp };
});
const status = { sample_id: seed.sample_id, experiment: 'E4', status: 'RUNNING', dry_run: dryRun,
  started_at: stamp(), requests: 0, assistant_turns: 0, tool_calls: 0, tool_errors: 0,
  prefix_verified: false, timed_out: false, usage: structuredClone(zeroUsage) };
let fixtureRequests = 0;
const fixtureFetch = async () => {
  fixtureRequests++;
  const chunk = fixtureRequests === 1
    ? { choices: [{ index: 0, delta: { role: 'assistant', tool_calls: [{ index: 0, id: 'e4-fixture-ls',
        type: 'function', function: { name: 'ls', arguments: '{"path":"/work/base","limit":1}' } }] }, finish_reason: 'tool_calls' }] }
    : { choices: [{ index: 0, delta: { role: 'assistant', content: '{"dry_run":true}' }, finish_reason: 'stop' }] };
  return new Response(`data: ${JSON.stringify(chunk)}\n\ndata: [DONE]\n\n`,
    { status: 200, headers: { 'content-type': 'text/event-stream' } });
};
const agent = new Agent({
  initialState: { systemPrompt: seed.messages[0].content, messages: prefix, model,
    thinkingLevel: 'high', tools },
  streamFn: (model, context, options) => runtime.streamSimple(model, context, {
    ...options, maxTokens: config.max_output_tokens, timeoutMs: config.timeout_sec * 1000,
    ...(dryRun ? { apiKey: 'dry-run-not-a-secret', fetch: fixtureFetch } : {}) }),
  getApiKey: () => dryRun ? 'dry-run-not-a-secret' : process.env.DEEPSEEK_API_KEY,
  sessionId: seed.sample_id,
  toolExecution: 'parallel',
  beforeToolCall: ({ args }) => {
    const path = resolve('/work/base', args.path || '.');
    let resolved = path;
    try { resolved = realpathSync(path); } catch { /* Native tool returns missing-path errors. */ }
    if (resolved !== '/work/base' && !resolved.startsWith('/work/base/'))
      return { block: true, reason: 'Inspect only the frozen repository at /work/base.' };
  },
  onPayload: payload => {
    const actualPrefix = payload.messages.slice(0, seed.messages.length).map(normalizeWire);
    if (!isDeepStrictEqual(actualPrefix, seed.messages.map(normalizeWire))) {
      save('prefix-check.json', { passed: false, requests: status.requests });
      throw new Error('Native provider payload does not preserve the student prefix');
    }
    const actualTools = payload.tools.map(t => t.function);
    if (!isDeepStrictEqual(actualTools.map(f => ({name:f.name,description:f.description,parameters:f.parameters})),
      seed.tools.map(t => ({name:t.function.name,description:t.function.description,parameters:t.function.parameters}))))
      throw new Error('Native provider tool schemas differ from the seed');
    status.prefix_verified = true;
    status.requests++;
    status.updated_at = stamp();
    save('prefix-check.json', { passed: true, requests_checked: status.requests,
      messages: seed.messages.length, tools: tools.map(t => t.name),
      ignored_transport_fields: ['empty reasoning_content', 'content string/block representation'] });
    append('provider-requests.jsonl', payload);
    append('request-times.jsonl', { request: status.requests, timestamp: status.updated_at });
    save('status.json', status);
    return payload;
  },
  onResponse: response => append('http-responses.jsonl', { timestamp: stamp(), status: response.status,
    request_id: response.headers['x-request-id'] || response.headers['request-id'] || null }),
});
agent.subscribe(event => {
  if (event.type === 'message_update') {
    const e = event.assistantMessageEvent;
    if (e.type.endsWith('_delta')) append('provider-stream.jsonl', { type:e.type,
      delta:e.delta, contentIndex:e.contentIndex });
    return;
  }
  append('pi.jsonl', event);
  if (event.type === 'message_end' && event.message.role === 'assistant') {
    status.assistant_turns++;
    status.stop_reason = event.message.stopReason;
    status.error = event.message.errorMessage || null;
    const usage = event.message.usage || zeroUsage;
    for (const k of ['input','output','cacheRead','cacheWrite','totalTokens']) status.usage[k] += usage[k] || 0;
    for (const k of Object.keys(status.usage.cost)) status.usage.cost[k] += usage.cost?.[k] || 0;
  }
  if (event.type === 'tool_execution_end') {
    status.tool_calls++;
    status.tool_errors += Number(event.isError);
  }
  status.updated_at = stamp();
  save('status.json', status);
  if (event.type === 'turn_end') {
    save('native-session.json', agent.state.messages);
    console.log(JSON.stringify({ timestamp:stamp(), sample_id:seed.sample_id,
      turns:status.assistant_turns, tool_calls:status.tool_calls, stop:status.stop_reason }));
  }
});
const timer = setTimeout(() => { status.timed_out = true; agent.abort(); }, config.timeout_sec * 1000);
for (const signal of ['SIGTERM','SIGINT']) process.on(signal, () => agent.abort());
try {
  save('status.json', status);
  await agent.continue();
  const last = agent.state.messages.findLast(m => m.role === 'assistant');
  writeFileSync(`${output}/output.txt`, text(last?.content));
  status.status = status.timed_out ? 'TIMED_OUT' : last?.stopReason === 'stop' ? 'GENERATED_REVIEW_PENDING' : 'FAILED';
} catch (error) {
  status.status = 'FAILED';
  status.error = String(error);
} finally {
  clearTimeout(timer);
  save('native-session.json', agent.state.messages);
  status.finished_at = stamp();
  save('status.json', status);
}
process.exitCode = status.status === 'GENERATED_REVIEW_PENDING' ? 0 : 1;
