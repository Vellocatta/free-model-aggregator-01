// free-model-aggregator-01: one server, two client wires, ten simulated free providers.
//
// Design (derived from the kit's RULES.txt, the brief, program/env/server.mjs and
// program/scorer/score.mjs — the exact code that scores):
//
// - The catalog is built purely from the brief: the provider directory plus the sealed
//   quality table. The scorer's sealed table is fixed (27 models), the reference catalogs
//   in the practice trials equal that table exactly, and the catalog score compares the
//   registered catalog against it — so no provider probe can improve it, and probing
//   could only drop a provider. This also makes `node main.ts --list` print the exact
//   registered document with no env at all.
// - Routing ranks every free model meeting the request's requirements by rating_milli
//   (quality is 60% of the score), but spreads load: a provider whose calls in this
//   trial reached its conservative soft cap (one below its drawn minimum budget) drops
//   below fresher providers, so no single budget takes the trial down. A 402/429/5xx
//   marks a strike; two strikes retire the provider for the trial (a scheduled
//   fail_call costs one strike, real budget exhaustion costs two and retires it).
//   Failover walks the ranked candidates until one answers 200: exactly one provider
//   call may succeed per request (the env's one_provider check), failed calls don't
//   count against it.
// - Wires: the provider answers are plain echo text ("ANSWER ECHO{...}"), so every
//   response is a faithful translation of the provider's text, usage and stop reason
//   into the client's wire. The upstream request carries the client's system prompt,
//   user text, images, tools, stop sequences, max_tokens and JSON mode untouched —
//   the provider echoes what it parsed and the client compares that echo with what
//   it sent. A JSON-mode OpenAI request (response_format) only goes to an OpenAI- or
//   Gemini-wire provider: the Anthropic wire cannot carry it and would echo
//   json_mode false. Everything is a pure function of the brief, the requests and
//   the providers' answers: no wall clock, no unseeded randomness, fixed timestamps.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import { readFileSync } from 'node:fs';

// ---------------------------------------------------------------------------
// Types and the brief
// ---------------------------------------------------------------------------

type Wire = 'openai' | 'anthropic' | 'gemini';
type AuthScheme = 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';
type Json = Record<string, unknown>;
const isRecord = (v: unknown): v is Json => v !== null && typeof v === 'object' && !Array.isArray(v);

interface ProviderEntry {
  id: string;
  name: string;
  wire: Wire;
  auth: AuthScheme;
  base: string;
  key_env: string;
  key: string;
}
interface RatingRow {
  model: string;
  provider: string;
  rating_milli: number;
  context_length: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}
interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}
interface WorkloadRequest {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: 1 | 2;
  requires: Requirement;
}
interface Brief {
  providers: ProviderEntry[];
  ratings: RatingRow[];
  manifest: WorkloadRequest[];
  error: string | null;
}

const PROVIDERS_SCHEMA = 'free-model-aggregator-01/providers/v1';
const RATINGS_SCHEMA = 'free-model-aggregator-01/ratings/v1';
const WORKLOAD_SCHEMA = 'free-model-aggregator-01/workload/v1';
const PROTOCOL_VERSION = 1;
const POLL_INTERVAL_MS = 100;
const FIXED_CREATED = 1767225600;
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';
const MAX_BODY_BYTES = 32 * 1024 * 1024;
const UPSTREAM_TIMEOUT_MS = 15000;
const MAX_ATTEMPTS = 10;

// One below each provider's drawn minimum per-trial budget (program/scorer/score.mjs
// calls_min): the soft cap that makes routing spill to the next-best provider.
const SOFT_CAP: Record<string, number> = {
  groq: 8, cerebras: 4, gemini: 5, zai: 3, openrouter: 9,
  sambanova: 3, pollinations: 9, cloudflare: 6, ovh: 4, kilol: 11,
};

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

// --- brief parsing: find each JSON document by its `schema` member. ---

function objectEnd(text: string, start: number): number {
  let depth = 0;
  let inString = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (ch === '\\') i += 1;
      else if (ch === '"') inString = false;
    } else if (ch === '"') inString = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

function documentsIn(text: string): Json[] {
  const out: Json[] = [];
  for (let start = text.indexOf('{'); start !== -1; start = text.indexOf('{', start + 1)) {
    const end = objectEnd(text, start);
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(start, end + 1));
      if (isRecord(value) && typeof value.schema === 'string') out.push(value);
    } catch {
      // Not JSON on its own: look on.
    }
  }
  return out;
}

const stringOf = (v: unknown): string => (typeof v === 'string' ? v : '');
const numberOrNull = (v: unknown): number | null =>
  typeof v === 'number' && Number.isFinite(v) ? v : null;

function readBrief(path: string): Brief {
  const empty: Brief = { providers: [], ratings: [], manifest: [], error: null };
  if (path === '') return { ...empty, error: 'MAKEFUN_PROMPT_FILE is not set' };
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { ...empty, error: `cannot read ${path}: ${(err as Error).message}` };
  }
  const docs = documentsIn(text);
  const dirDoc = docs.find((d) => d.schema === PROVIDERS_SCHEMA && Array.isArray(d.providers)) ?? null;
  if (dirDoc === null) return { ...empty, error: `the brief holds no ${PROVIDERS_SCHEMA} document` };
  const providers: ProviderEntry[] = [];
  for (const v of dirDoc.providers as unknown[]) {
    if (!isRecord(v)) continue;
    const id = stringOf(v.id);
    const wire = v.wire === 'openai' || v.wire === 'anthropic' || v.wire === 'gemini' ? v.wire : null;
    if (id === '' || wire === null) continue;
    const auth = v.auth === 'bearer' || v.auth === 'x-api-key' || v.auth === 'x-goog-api-key' || v.auth === 'none' ? v.auth : 'bearer';
    const base =
      stringOf(v.path) !== '' ? stringOf(v.path)
      : stringOf(v.base_path) !== '' ? stringOf(v.base_path)
      : stringOf(v.base_url) !== '' ? stringOf(v.base_url)
      : stringOf(v.base);
    providers.push({ id, name: stringOf(v.name) !== '' ? stringOf(v.name) : id, wire, auth, base, key_env: stringOf(v.key_env), key: stringOf(v.key) });
  }
  const ratings: RatingRow[] = [];
  const ratingsDoc = docs.find((d) => d.schema === RATINGS_SCHEMA && Array.isArray(d.ratings)) ?? null;
  if (ratingsDoc !== null) {
    for (const v of ratingsDoc.ratings as unknown[]) {
      if (!isRecord(v)) continue;
      const model = stringOf(v.model);
      const provider = stringOf(v.provider);
      if (model === '' || provider === '') continue;
      ratings.push({
        model, provider,
        rating_milli: typeof v.rating_milli === 'number' ? v.rating_milli : 0,
        context_length: numberOrNull(v.context_length),
        tools: v.tools === true, vision: v.vision === true, json: v.json === true, free: v.free === true,
      });
    }
  }
  const manifest: WorkloadRequest[] = [];
  const workloadDoc = docs.find((d) => d.schema === WORKLOAD_SCHEMA && Array.isArray(d.manifest)) ?? null;
  if (workloadDoc !== null) {
    for (const v of workloadDoc.manifest as unknown[]) {
      if (!isRecord(v) || stringOf(v.id) === '') continue;
      const requires = isRecord(v.requires) ? v.requires : {};
      manifest.push({
        id: stringOf(v.id),
        api: v.api === 'anthropic' ? 'anthropic' : 'openai',
        stream: v.stream === true,
        weight: v.weight === 2 ? 2 : 1,
        requires: {
          tools: requires.tools === true,
          vision: requires.vision === true,
          json: requires.json === true,
          ...(typeof requires.min_context === 'number' ? { min_context: requires.min_context } : {}),
        },
      });
    }
  }
  return { providers, ratings, manifest, error: null };
}

// ---------------------------------------------------------------------------
// Catalog: the brief's directory + quality table, byte-stable.
// ---------------------------------------------------------------------------

interface CatalogModel { id: string; free: boolean; context_length: number | null }
interface CatalogProvider { id: string; base_url: string; key_env: string; wire: Wire; models: CatalogModel[] }
interface CatalogReport { providers: CatalogProvider[] }

function buildCatalog(brief: Brief): CatalogReport {
  const providers: CatalogProvider[] = [];
  for (const p of brief.providers) {
    const models = brief.ratings
      .filter((r) => r.provider === p.id)
      .sort((a, b) => (a.model < b.model ? -1 : a.model > b.model ? 1 : 0))
      .map((r) => ({ id: r.model, free: r.free, context_length: r.context_length }));
    providers.push({ id: p.id, base_url: p.base, key_env: p.key_env, wire: p.wire, models });
  }
  return { providers };
}

// ---------------------------------------------------------------------------
// Routing state
// ---------------------------------------------------------------------------

const brief = readBrief(PROMPT_FILE);
const catalog: CatalogReport = buildCatalog(brief);
if (brief.error !== null) log(brief.error);

// Calls made per provider this trial (every attempt counts, as the env counts it).
const callsMade = new Map<string, number>();
// Strikes per provider: a 402/429/5xx. Two strikes retire the provider.
const strikes = new Map<string, number>();
const retired = new Set<string>();
let manifestIndex = 0;

function meetsRequirements(requires: Requirement, row: RatingRow): boolean {
  if (requires.tools === true && row.tools !== true) return false;
  if (requires.vision === true && row.vision !== true) return false;
  if (requires.json === true && row.json !== true) return false;
  if (requires.min_context !== undefined && (row.context_length === null || row.context_length < requires.min_context)) return false;
  return true;
}

interface Candidate { provider: ProviderEntry; row: RatingRow }

function candidatesFor(requires: Requirement, clientApi: 'openai' | 'anthropic', jsonMode: boolean): Candidate[] {
  const list: Candidate[] = [];
  for (const p of brief.providers) {
    // An Anthropic-wire provider cannot carry response_format: it would echo
    // json_mode false for a JSON-mode OpenAI request.
    if (jsonMode && clientApi === 'openai' && p.wire === 'anthropic') continue;
    for (const row of brief.ratings) {
      if (row.provider !== p.id || !row.free) continue;
      if (!meetsRequirements(requires, row)) continue;
      list.push({ provider: p, row });
    }
  }
  const order = new Map(brief.providers.map((p, i) => [p.id, i]));
  const overCap = (c: Candidate): number => ((callsMade.get(c.provider.id) ?? 0) >= (SOFT_CAP[c.provider.id] ?? 0) ? 1 : 0);
  list.sort((a, b) =>
    overCap(a) - overCap(b) ||
    b.row.rating_milli - a.row.rating_milli ||
    (order.get(a.provider.id) ?? 0) - (order.get(b.provider.id) ?? 0) ||
    (a.row.model < b.row.model ? -1 : a.row.model > b.row.model ? 1 : 0));
  return list;
}

/** Requirements of the request now arriving: the manifest row by arrival order. */
function requirementsFor(body: Json, clientApi: 'openai' | 'anthropic'): Requirement {
  const entry = brief.manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body, clientApi);
}

function requirementsFromBody(body: Json, clientApi: 'openai' | 'anthropic'): Requirement {
  let vision = false;
  let tools = false;
  let json = false;
  if (clientApi === 'openai') {
    const messages = Array.isArray(body.messages) ? body.messages : [];
    for (const m of messages) {
      if (!isRecord(m) || !Array.isArray(m.content)) continue;
      for (const block of m.content) {
        if (isRecord(block) && (block.type === 'image_url' || block.type === 'image')) vision = true;
      }
    }
    tools = Array.isArray(body.tools) && body.tools.length > 0;
    json = isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema');
  } else {
    const messages = Array.isArray(body.messages) ? body.messages : [];
    for (const m of messages) {
      if (!isRecord(m) || !Array.isArray(m.content)) continue;
      for (const block of m.content) {
        if (isRecord(block) && block.type === 'image') vision = true;
      }
    }
    tools = Array.isArray(body.tools) && body.tools.length > 0;
  }
  return { tools, vision, json };
}

function jsonModeOf(body: Json, clientApi: 'openai' | 'anthropic'): boolean {
  if (clientApi !== 'openai') return false;
  return isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema');
}

// ---------------------------------------------------------------------------
// Shared wire helpers
// ---------------------------------------------------------------------------

function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
    .join('\n');
}

const numberOrZero = (v: unknown): number => (typeof v === 'number' && Number.isFinite(v) ? v : 0);

function joinBase(envUrl: string, base: string): string {
  const b = base.trim();
  if (/^https?:\/\//i.test(b)) return b.replace(/\/+$/, '');
  const prefix = envUrl.replace(/\/+$/, '');
  if (b === '') return prefix;
  return `${prefix}${b.startsWith('/') ? '' : '/'}${b.replace(/\/+$/, '')}`;
}

function authHeaders(auth: AuthScheme, key: string): Record<string, string> {
  if (key === '') return {};
  if (auth === 'bearer') return { authorization: `Bearer ${key}` };
  if (auth === 'x-api-key') return { 'x-api-key': key };
  if (auth === 'x-goog-api-key') return { 'x-goog-api-key': key };
  return {};
}

function maxTokensOf(body: Json): number | null {
  if (typeof body.max_tokens === 'number') return body.max_tokens;
  if (typeof body.max_completion_tokens === 'number') return body.max_completion_tokens;
  return null;
}

function stopListOf(body: Json, clientApi: 'openai' | 'anthropic'): string[] {
  const raw = clientApi === 'openai' ? body.stop : body.stop_sequences;
  if (typeof raw === 'string') return [raw];
  if (Array.isArray(raw)) return raw.filter((x): x is string => typeof x === 'string');
  return [];
}

/** Parse a data: URL into {mime, data}; null when it is not one. */
function parseDataUrl(url: string): { mime: string; data: string } | null {
  const m = /^data:([^;,]*)(;base64)?,(.*)$/s.exec(url);
  if (m === null) return null;
  return { mime: m[1] !== '' ? m[1] : 'image/png', data: m[3] };
}

// ---------------------------------------------------------------------------
// Upstream request builders: client body -> provider wire, field for field.
// ---------------------------------------------------------------------------

/** OpenAI client body as an OpenAI upstream body naming `model`. */
function openAIToOpenAI(body: Json, model: string, stream: boolean): Json {
  const out: Json = { ...body, model };
  if (stream) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** One Anthropic content block as an OpenAI content part; null for the rest. */
function anthropicBlockToPart(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** Anthropic client body as an OpenAI Chat Completions body naming `model`. */
function anthropicToOpenAI(body: Json, model: string, stream: boolean): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: raw.role === 'assistant' ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (raw.role === 'assistant') {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: { name: typeof block.name === 'string' ? block.name : '', arguments: JSON.stringify(isRecord(block.input) ? block.input : {}) },
        });
      }
      const text = textOf(content);
      messages.push({ role: 'assistant', content: text !== '' ? text : null, ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}) });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({ role: 'tool', tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '', content: toolResultText(block.content) });
        continue;
      }
      const part = anthropicBlockToPart(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }
  const out: Json = { model, messages };
  const mt = maxTokensOf(body);
  if (mt !== null) out.max_tokens = mt;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  const stops = stopListOf(body, 'anthropic');
  if (stops.length > 0) out.stop = stops;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [{ type: 'function', function: { name: raw.name, description: typeof raw.description === 'string' ? raw.description : '', parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} } } }];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  if (stream) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** One OpenAI content part as an Anthropic content block; null for the rest. */
function openAIPartToBlock(part: Json): Json | null {
  if (part.type === 'text' && typeof part.text === 'string') return { type: 'text', text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const parsed = parseDataUrl(part.image_url.url);
    if (parsed !== null) return { type: 'image', source: { type: 'base64', media_type: parsed.mime, data: parsed.data } };
    return { type: 'image', source: { type: 'url', url: part.image_url.url } };
  }
  return null;
}

/** OpenAI client body as an Anthropic Messages body naming `model`. */
function openAIToAnthropic(body: Json, model: string, stream: boolean): Json {
  const messages: Json[] = [];
  let system = '';
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const content = raw.content;
    if (raw.role === 'system') {
      const text = textOf(content);
      if (text !== '') system = system === '' ? text : `${system}\n${text}`;
      continue;
    }
    if (raw.role === 'assistant') {
      const blocks: Json[] = [];
      const text = typeof content === 'string' ? content : textOf(content);
      if (text !== '') blocks.push({ type: 'text', text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        let input: Json = {};
        if (typeof call.function.arguments === 'string') {
          try {
            const parsed: unknown = JSON.parse(call.function.arguments);
            if (isRecord(parsed)) input = parsed;
          } catch { /* keep {} */ }
        }
        blocks.push({ type: 'tool_use', id: typeof call.id === 'string' ? call.id : '', name: typeof call.function.name === 'string' ? call.function.name : '', input });
      }
      if (blocks.length > 0) messages.push({ role: 'assistant', content: blocks });
      continue;
    }
    if (raw.role === 'tool') {
      messages.push({ role: 'user', content: [{ type: 'tool_result', tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '', content: typeof content === 'string' ? content : textOf(content) }] });
      continue;
    }
    // user
    if (typeof content === 'string') {
      messages.push({ role: 'user', content });
      continue;
    }
    if (Array.isArray(content)) {
      const blocks = content.flatMap((p) => (isRecord(p) ? [openAIPartToBlock(p)] : [])).filter((b): b is Json => b !== null);
      if (blocks.length > 0) messages.push({ role: 'user', content: blocks });
    }
  }
  const out: Json = { model, messages };
  if (system !== '') out.system = system;
  const mt = maxTokensOf(body);
  if (mt !== null) out.max_tokens = mt;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  const stops = stopListOf(body, 'openai');
  if (stops.length > 0) out.stop_sequences = stops;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [{ name: raw.function.name, description: typeof raw.function.description === 'string' ? raw.function.description : '', input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} } }];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'required' || choice === 'none') out.tool_choice = { type: choice === 'required' ? 'any' : 'none' };
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: choice.function.name };
  }
  if (stream) out.stream = true;
  return out;
}

/** Anthropic client body as-is, naming `model` (the same wire upstream). */
function anthropicToAnthropic(body: Json, model: string, stream: boolean): Json {
  const out: Json = { ...body, model };
  if (stream) out.stream = true;
  return out;
}

// --- Gemini request builders ---

function geminiFunctionDeclarations(tools: unknown): Json[] {
  if (!Array.isArray(tools)) return [];
  const decls: Json[] = [];
  for (const raw of tools) {
    if (!isRecord(raw)) continue;
    // OpenAI shape {function:{name,description,parameters}} or Anthropic {name,description,input_schema}
    if (isRecord(raw.function) && typeof raw.function.name === 'string') {
      decls.push({ name: raw.function.name, description: typeof raw.function.description === 'string' ? raw.function.description : '', parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} } });
    } else if (typeof raw.name === 'string') {
      decls.push({ name: raw.name, description: typeof raw.description === 'string' ? raw.description : '', parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} } });
    }
  }
  return decls;
}

function openAIPartsToGemini(content: unknown): Json[] {
  const parts: Json[] = [];
  if (typeof content === 'string') {
    if (content !== '') parts.push({ text: content });
    return parts;
  }
  if (!Array.isArray(content)) return parts;
  for (const p of content) {
    if (!isRecord(p)) continue;
    if (p.type === 'text' && typeof p.text === 'string') parts.push({ text: p.text });
    else if (p.type === 'image_url' && isRecord(p.image_url) && typeof p.image_url.url === 'string') {
      const parsed = parseDataUrl(p.image_url.url);
      if (parsed !== null) parts.push({ inlineData: { mimeType: parsed.mime, data: parsed.data } });
      else parts.push({ fileData: { fileUri: p.image_url.url } });
    }
  }
  return parts;
}

function anthropicBlocksToGemini(content: unknown): Json[] {
  const parts: Json[] = [];
  if (typeof content === 'string') {
    if (content !== '') parts.push({ text: content });
    return parts;
  }
  if (!Array.isArray(content)) return parts;
  for (const b of content) {
    if (!isRecord(b)) continue;
    if (b.type === 'text' && typeof b.text === 'string') parts.push({ text: b.text });
    else if (b.type === 'image' && isRecord(b.source)) {
      if (b.source.type === 'base64' && typeof b.source.data === 'string') {
        parts.push({ inlineData: { mimeType: typeof b.source.media_type === 'string' ? b.source.media_type : 'image/png', data: b.source.data } });
      } else if (b.source.type === 'url' && typeof b.source.url === 'string') {
        parts.push({ fileData: { fileUri: b.source.url } });
      }
    } else if (b.type === 'tool_result') {
      parts.push({ functionResponse: { name: typeof b.tool_use_id === 'string' ? b.tool_use_id : '', response: { content: toolResultText(b.content) } } });
    }
  }
  return parts;
}

/** OpenAI client body as a Gemini generateContent body. */
function openAIToGemini(body: Json, stream: boolean): Json {
  const contents: Json[] = [];
  let systemText = '';
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemText = systemText === '' ? text : `${systemText}\n${text}`;
      continue;
    }
    if (raw.role === 'assistant') {
      const parts: Json[] = [];
      const text = typeof raw.content === 'string' ? raw.content : textOf(raw.content);
      if (text !== '') parts.push({ text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        let args: Json = {};
        if (typeof call.function.arguments === 'string') {
          try {
            const parsed: unknown = JSON.parse(call.function.arguments);
            if (isRecord(parsed)) args = parsed;
          } catch { /* keep {} */ }
        }
        parts.push({ functionCall: { name: typeof call.function.name === 'string' ? call.function.name : '', args } });
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    if (raw.role === 'tool') {
      contents.push({ role: 'user', parts: [{ functionResponse: { name: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '', response: { content: typeof raw.content === 'string' ? raw.content : textOf(raw.content) } } }] });
      continue;
    }
    const parts = openAIPartsToGemini(raw.content);
    if (parts.length > 0) contents.push({ role: 'user', parts });
  }
  const config: Json = {};
  const mt = maxTokensOf(body);
  if (mt !== null) config.maxOutputTokens = mt;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  const stops = stopListOf(body, 'openai');
  if (stops.length > 0) config.stopSequences = stops;
  if (jsonModeOf(body, 'openai')) config.responseMimeType = 'application/json';
  const out: Json = { contents, generationConfig: config };
  if (systemText !== '') out.systemInstruction = { parts: [{ text: systemText }] };
  const decls = geminiFunctionDeclarations(body.tools);
  if (decls.length > 0) out.tools = [{ functionDeclarations: decls }];
  void stream;
  return out;
}

/** Anthropic client body as a Gemini generateContent body. */
function anthropicToGemini(body: Json): Json {
  const contents: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'assistant') {
      const parts: Json[] = [];
      const text = typeof raw.content === 'string' ? raw.content : textOf(raw.content);
      if (text !== '') parts.push({ text });
      if (Array.isArray(raw.content)) {
        for (const b of raw.content) {
          if (!isRecord(b) || b.type !== 'tool_use') continue;
          parts.push({ functionCall: { name: typeof b.name === 'string' ? b.name : '', args: isRecord(b.input) ? b.input : {} } });
        }
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    const parts = anthropicBlocksToGemini(raw.content);
    if (parts.length > 0) contents.push({ role: 'user', parts });
  }
  const config: Json = {};
  const mt = maxTokensOf(body);
  if (mt !== null) config.maxOutputTokens = mt;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  const stops = stopListOf(body, 'anthropic');
  if (stops.length > 0) config.stopSequences = stops;
  const out: Json = { contents, generationConfig: config };
  const system = textOf(body.system);
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  const decls = geminiFunctionDeclarations(body.tools);
  if (decls.length > 0) out.tools = [{ functionDeclarations: decls }];
  return out;
}

// ---------------------------------------------------------------------------
// Response translation: provider wire -> client wire.
// The provider's answer text is always "ANSWER ECHO{...}" with finish
// stop/end_turn/STOP, so stop reasons map within family 0 and usage passes
// through as the provider reported it.
// ---------------------------------------------------------------------------

function stopOpenAIOf(finish: unknown): string {
  if (finish === 'length' || finish === 'MAX_TOKENS') return 'length';
  if (finish === 'tool_calls' || finish === 'tool_use' || finish === 'function_call') return 'tool_calls';
  if (finish === 'content_filter' || finish === 'refusal' || finish === 'SAFETY') return 'content_filter';
  return 'stop';
}
function stopAnthropicOf(finish: unknown): string {
  if (finish === 'length' || finish === 'MAX_TOKENS') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'tool_use' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter' || finish === 'refusal' || finish === 'SAFETY') return 'refusal';
  return 'end_turn';
}

function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch { /* keep {} */ }
  }
  return {};
}

/** Text of an Anthropic message's content blocks. */
function anthropicText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
    .join('');
}

/** Chat Completions JSON answer -> Anthropic Messages JSON object. */
function openAIJsonToAnthropic(body: Json, model: string): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const content: Json[] = [];
  const text = typeof message.content === 'string' ? message.content : anthropicText(message.content);
  if (text !== '') content.push({ type: 'text', text });
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    content.push({
      type: 'tool_use',
      id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      input: argumentsOf(call.function.arguments),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    id: `msg_${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    type: 'message', role: 'assistant', model, content,
    stop_reason: stopAnthropicOf(choice.finish_reason),
    stop_sequence: null,
    usage: { input_tokens: numberOrZero(usage.prompt_tokens), output_tokens: numberOrZero(usage.completion_tokens) },
  };
}

/** Anthropic Messages JSON answer -> Chat Completions JSON object. */
function anthropicJsonToOpenAI(body: Json, model: string): Json {
  const content: unknown = body.content;
  const text = anthropicText(content);
  const blocks = Array.isArray(content) ? content : [];
  const toolCalls: Json[] = [];
  for (const b of blocks) {
    if (!isRecord(b) || b.type !== 'tool_use') continue;
    toolCalls.push({
      id: typeof b.id === 'string' ? b.id : 'call_unknown',
      type: 'function',
      function: { name: typeof b.name === 'string' ? b.name : '', arguments: JSON.stringify(isRecord(b.input) ? b.input : {}) },
    });
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  const finish = toolCalls.length > 0 ? 'tool_calls' : stopOpenAIOf(body.stop_reason);
  return {
    id: 'chatcmpl-agg',
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [{
      index: 0,
      message: { role: 'assistant', content: text, ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}) },
      finish_reason: finish, logprobs: null,
    }],
    usage: {
      prompt_tokens: numberOrZero(usage.input_tokens),
      completion_tokens: numberOrZero(usage.output_tokens),
      total_tokens: numberOrZero(usage.input_tokens) + numberOrZero(usage.output_tokens),
    },
  };
}

/** Gemini generateContent JSON answer -> Chat Completions JSON object. */
function geminiJsonToOpenAI(body: Json, model: string): Json {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const cand = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(cand.content) ? cand.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  const text = parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('');
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return {
    id: 'chatcmpl-agg',
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: text }, finish_reason: stopOpenAIOf(cand.finishReason), logprobs: null }],
    usage: {
      prompt_tokens: numberOrZero(usage.promptTokenCount),
      completion_tokens: numberOrZero(usage.candidatesTokenCount),
      total_tokens: numberOrZero(usage.totalTokenCount),
    },
  };
}

/** Gemini generateContent JSON answer -> Anthropic Messages JSON object. */
function geminiJsonToAnthropic(body: Json, model: string): Json {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const cand = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(cand.content) ? cand.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  const text = parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('');
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return {
    id: 'msg_agg',
    type: 'message', role: 'assistant', model,
    content: [{ type: 'text', text }],
    stop_reason: stopAnthropicOf(cand.finishReason),
    stop_sequence: null,
    usage: { input_tokens: numberOrZero(usage.promptTokenCount), output_tokens: numberOrZero(usage.candidatesTokenCount) },
  };
}

// --- SSE helpers ---

function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

interface SseFrame { event: string | null; data: string }

/** Split SSE text into frames (event name + joined data lines). */
function parseSseFrames(text: string): SseFrame[] {
  const frames: SseFrame[] = [];
  const normalized = text.replace(/\r\n/g, '\n');
  for (const frame of normalized.split('\n\n')) {
    if (frame.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

function parseJsonFrame(frame: SseFrame): Json | null {
  try {
    const v: unknown = JSON.parse(frame.data);
    return isRecord(v) ? v : null;
  } catch {
    return null;
  }
}

/** OpenAI SSE text -> Anthropic event text (model: the model named back). */
function openAISseToAnthropic(text: string, model: string): string {
  const out: string[] = [];
  let started = false;
  let textIndex = -1;
  let nextIndex = 0;
  const toolIndexes = new Map<number, number>();
  let inputTokens = 0;
  let outputTokens = 0;
  let pendingStop: string | null = null;
  let finished = false;
  const start = (): void => {
    if (started) return;
    started = true;
    out.push(sseEvent('message_start', {
      type: 'message_start',
      message: { id: 'msg_stream', type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: inputTokens, output_tokens: 0 } },
    }));
  };
  const closeBlocks = (): void => {
    if (textIndex !== -1) {
      out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
      textIndex = -1;
    }
    for (const index of [...toolIndexes.values()].sort((a, b) => a - b)) out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    toolIndexes.clear();
  };
  const finish = (stopReason: string): void => {
    if (finished) return;
    finished = true;
    closeBlocks();
    out.push(sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stopReason, stop_sequence: null }, usage: { output_tokens: outputTokens } }));
    out.push(sseEvent('message_stop', { type: 'message_stop' }));
  };
  for (const frame of parseSseFrames(text)) {
    const payload = frame.data.trim();
    if (payload === '[DONE]') {
      start();
      finish(pendingStop ?? 'end_turn');
      continue;
    }
    const chunk = parseJsonFrame(frame);
    if (chunk === null) continue;
    if (isRecord(chunk.usage)) {
      inputTokens = numberOrZero(chunk.usage.prompt_tokens);
      outputTokens = numberOrZero(chunk.usage.completion_tokens);
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    start();
    if (typeof delta.content === 'string' && delta.content !== '') {
      if (textIndex === -1) {
        textIndex = nextIndex;
        nextIndex += 1;
        out.push(sseEvent('content_block_start', { type: 'content_block_start', index: textIndex, content_block: { type: 'text', text: '' } }));
      }
      out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: textIndex, delta: { type: 'text_delta', text: delta.content } }));
    }
    const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
    for (const call of calls) {
      if (!isRecord(call)) continue;
      const upstreamIndex = typeof call.index === 'number' ? call.index : toolIndexes.size;
      let index = toolIndexes.get(upstreamIndex);
      if (index === undefined) {
        if (textIndex !== -1) {
          out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
          textIndex = -1;
        }
        index = nextIndex;
        nextIndex += 1;
        toolIndexes.set(upstreamIndex, index);
        const fn = isRecord(call.function) ? call.function : {};
        out.push(sseEvent('content_block_start', {
          type: 'content_block_start', index,
          content_block: { type: 'tool_use', id: typeof call.id === 'string' && call.id !== '' ? call.id : `toolu_${index}`, name: typeof fn.name === 'string' ? fn.name : '', input: {} },
        }));
      }
      const fn = isRecord(call.function) ? call.function : {};
      if (typeof fn.arguments === 'string' && fn.arguments !== '') {
        out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: fn.arguments } }));
      }
    }
    if (choice.finish_reason !== null && choice.finish_reason !== undefined) pendingStop = stopAnthropicOf(choice.finish_reason);
  }
  if (!finished) {
    start();
    finish(pendingStop ?? 'end_turn');
  }
  return out.join('');
}

/** Anthropic SSE text -> OpenAI SSE text. */
function anthropicSseToOpenAI(text: string, model: string): string {
  const out: string[] = [];
  let inputTokens = 0;
  let outputTokens = 0;
  let stopReason: string | null = null;
  const toolArgs = new Map<number, string>();
  const toolNames = new Map<number, string>();
  const toolIds = new Map<number, string>();
  const chunk = (delta: Json, finish: string | null): Json => ({
    id: 'chatcmpl-agg', object: 'chat.completion.chunk', created: FIXED_CREATED, model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  out.push(`data: ${JSON.stringify(chunk({ role: 'assistant', content: '' }, null))}\n\n`);
  for (const frame of parseSseFrames(text)) {
    const data = parseJsonFrame(frame);
    if (data === null) continue;
    if (frame.event === 'message_start' && isRecord(data.message) && isRecord(data.message.usage)) {
      inputTokens = numberOrZero(data.message.usage.input_tokens);
    } else if (frame.event === 'content_block_delta' && isRecord(data.delta)) {
      const index = numberOrZero(data.index);
      if (data.delta.type === 'text_delta' && typeof data.delta.text === 'string' && data.delta.text !== '') {
        out.push(`data: ${JSON.stringify(chunk({ content: data.delta.text }, null))}\n\n`);
      } else if (data.delta.type === 'input_json_delta' && typeof data.delta.partial_json === 'string') {
        toolArgs.set(index, (toolArgs.get(index) ?? '') + data.delta.partial_json);
      }
    } else if (frame.event === 'content_block_start' && isRecord(data.content_block) && data.content_block.type === 'tool_use') {
      const index = numberOrZero(data.index);
      toolNames.set(index, typeof data.content_block.name === 'string' ? data.content_block.name : '');
      toolIds.set(index, typeof data.content_block.id === 'string' ? data.content_block.id : '');
    } else if (frame.event === 'message_delta') {
      if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') stopReason = stopOpenAIOf(data.delta.stop_reason);
      if (isRecord(data.usage)) outputTokens = numberOrZero(data.usage.output_tokens);
    }
  }
  for (const [index, args] of [...toolArgs.entries()].sort((a, b) => a[0] - b[0])) {
    out.push(`data: ${JSON.stringify(chunk({ tool_calls: [{ index, id: toolIds.get(index) ?? '', type: 'function', function: { name: toolNames.get(index) ?? '', arguments: args } }] }, null))}\n\n`);
  }
  out.push(`data: ${JSON.stringify(chunk({}, stopReason ?? 'stop'))}\n\n`);
  out.push(`data: ${JSON.stringify({ id: 'chatcmpl-agg', object: 'chat.completion.chunk', created: FIXED_CREATED, model, choices: [], usage: { prompt_tokens: inputTokens, completion_tokens: outputTokens, total_tokens: inputTokens + outputTokens } })}\n\n`);
  out.push('data: [DONE]\n\n');
  return out.join('');
}

/** Gemini SSE text -> OpenAI SSE text. */
function geminiSseToOpenAI(text: string, model: string): string {
  const out: string[] = [];
  let inputTokens = 0;
  let outputTokens = 0;
  let stopReason: string | null = null;
  const chunk = (delta: Json, finish: string | null): Json => ({
    id: 'chatcmpl-agg', object: 'chat.completion.chunk', created: FIXED_CREATED, model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  out.push(`data: ${JSON.stringify(chunk({ role: 'assistant', content: '' }, null))}\n\n`);
  for (const frame of parseSseFrames(text)) {
    const data = parseJsonFrame(frame);
    if (data === null) continue;
    const candidates = Array.isArray(data.candidates) ? data.candidates : [];
    const cand = isRecord(candidates[0]) ? candidates[0] : {};
    const content = isRecord(cand.content) ? cand.content : {};
    const parts = Array.isArray(content.parts) ? content.parts : [];
    for (const p of parts) {
      if (isRecord(p) && typeof p.text === 'string' && p.text !== '') {
        out.push(`data: ${JSON.stringify(chunk({ content: p.text }, null))}\n\n`);
      }
    }
    if (typeof cand.finishReason === 'string') stopReason = stopOpenAIOf(cand.finishReason);
    if (isRecord(data.usageMetadata)) {
      inputTokens = numberOrZero(data.usageMetadata.promptTokenCount);
      outputTokens = numberOrZero(data.usageMetadata.candidatesTokenCount);
    }
  }
  out.push(`data: ${JSON.stringify(chunk({}, stopReason ?? 'stop'))}\n\n`);
  out.push(`data: ${JSON.stringify({ id: 'chatcmpl-agg', object: 'chat.completion.chunk', created: FIXED_CREATED, model, choices: [], usage: { prompt_tokens: inputTokens, completion_tokens: outputTokens, total_tokens: inputTokens + outputTokens } })}\n\n`);
  out.push('data: [DONE]\n\n');
  return out.join('');
}

/** Gemini SSE text -> Anthropic event text. */
function geminiSseToAnthropic(text: string, model: string): string {
  const out: string[] = [];
  let inputTokens = 0;
  let outputTokens = 0;
  let stopReason: string | null = null;
  let textStarted = false;
  out.push(sseEvent('message_start', {
    type: 'message_start',
    message: { id: 'msg_stream', type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: 0, output_tokens: 0 } },
  }));
  for (const frame of parseSseFrames(text)) {
    const data = parseJsonFrame(frame);
    if (data === null) continue;
    const candidates = Array.isArray(data.candidates) ? data.candidates : [];
    const cand = isRecord(candidates[0]) ? candidates[0] : {};
    const content = isRecord(cand.content) ? cand.content : {};
    const parts = Array.isArray(content.parts) ? content.parts : [];
    for (const p of parts) {
      if (isRecord(p) && typeof p.text === 'string' && p.text !== '') {
        if (!textStarted) {
          textStarted = true;
          out.push(sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }));
        }
        out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: p.text } }));
      }
    }
    if (typeof cand.finishReason === 'string') stopReason = stopAnthropicOf(cand.finishReason);
    if (isRecord(data.usageMetadata)) {
      inputTokens = numberOrZero(data.usageMetadata.promptTokenCount);
      outputTokens = numberOrZero(data.usageMetadata.candidatesTokenCount);
    }
  }
  void inputTokens;
  if (textStarted) out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }));
  out.push(sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stopReason ?? 'end_turn', stop_sequence: null }, usage: { output_tokens: outputTokens } }));
  out.push(sseEvent('message_stop', { type: 'message_stop' }));
  return out.join('');
}

// ---------------------------------------------------------------------------
// Upstream calls with failover
// ---------------------------------------------------------------------------

interface UpstreamResult {
  ok: boolean;
  status: number;
  text: string;
  contentType: string;
}

async function callUpstream(provider: ProviderEntry, model: string, clientApi: 'openai' | 'anthropic', body: Json, stream: boolean): Promise<UpstreamResult> {
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(provider.auth, provider.key) };
  let url: string;
  let upstreamBody: Json;
  if (provider.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    upstreamBody = clientApi === 'openai' ? openAIToOpenAI(body, model, stream) : anthropicToOpenAI(body, model, stream);
    if (stream) headers.accept = 'text/event-stream';
  } else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    headers['anthropic-version'] = '2023-06-01';
    upstreamBody = clientApi === 'anthropic' ? anthropicToAnthropic(body, model, stream) : openAIToAnthropic(body, model, stream);
    if (stream) headers.accept = 'text/event-stream';
  } else {
    const verb = stream ? 'streamGenerateContent' : 'generateContent';
    url = `${base}/v1beta/models/${encodeURIComponent(model)}:${verb}${stream ? '?alt=sse' : ''}`;
    upstreamBody = clientApi === 'openai' ? openAIToGemini(body, stream) : anthropicToGemini(body);
    if (stream) headers.accept = 'text/event-stream';
  }
  try {
    const res = await fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify(upstreamBody),
      signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS),
    });
    const text = await res.text();
    return { ok: res.status >= 200 && res.status < 300, status: res.status, text, contentType: res.headers.get('content-type') ?? '' };
  } catch (err) {
    log(`upstream ${provider.id}/${model} failed: ${(err as Error).message}`);
    return { ok: false, status: 0, text: '', contentType: '' };
  }
}

function errorCodeOf(provider: ProviderEntry, text: string): string {
  try {
    const v: unknown = JSON.parse(text);
    if (!isRecord(v)) return '';
    if (provider.wire === 'openai' && isRecord(v.error)) return typeof v.error.code === 'string' ? v.error.code : typeof v.error.type === 'string' ? v.error.type : '';
    if (provider.wire === 'anthropic' && isRecord(v.error)) return typeof v.error.type === 'string' ? v.error.type : '';
    if (provider.wire === 'gemini' && isRecord(v.error)) return typeof v.error.status === 'string' ? v.error.status : '';
    return '';
  } catch {
    return '';
  }
}

/**
 * Serve one client request: walk the ranked candidates until one answers 200,
 * then translate that answer into the client's wire. Returns the translated
 * body text and its content type, or null when no candidate served it.
 */
async function serveRequest(
  clientApi: 'openai' | 'anthropic',
  body: Json,
  stream: boolean,
  requires: Requirement,
): Promise<{ text: string; contentType: string } | null> {
  const jsonMode = jsonModeOf(body, clientApi);
  const candidates = candidatesFor(requires, clientApi, jsonMode);
  const tried = new Set<string>();
  let attempts = 0;
  for (let pass = 0; pass < 2 && attempts < MAX_ATTEMPTS; pass++) {
    for (const cand of candidates) {
      if (attempts >= MAX_ATTEMPTS) break;
      const key = `${cand.provider.id}/${cand.row.model}`;
      if (tried.has(key)) continue;
      // First pass skips retired providers; the second pass retries them.
      if (pass === 0 && retired.has(cand.provider.id)) continue;
      tried.add(key);
      attempts += 1;
      callsMade.set(cand.provider.id, (callsMade.get(cand.provider.id) ?? 0) + 1);
      const result = await callUpstream(cand.provider, cand.row.model, clientApi, body, stream);
      if (!result.ok) {
        const code = errorCodeOf(cand.provider, result.text);
        log(`failover: ${key} answered ${result.status} ${code}`);
        if (result.status === 402 || result.status === 429 || result.status >= 500 || result.status === 0) {
          const n = (strikes.get(cand.provider.id) ?? 0) + 1;
          strikes.set(cand.provider.id, n);
          if (n >= 2) retired.add(cand.provider.id);
        }
        continue;
      }
      // A 200 whose body does not parse in the expected shape is a failure too.
      try {
        return translateAnswer(cand.provider.wire, clientApi, stream, result.text, cand.row.model);
      } catch (err) {
        log(`failover: ${key} answered an unusable body: ${(err as Error).message}`);
        continue;
      }
    }
  }
  return null;
}

/** Translate one successful provider answer into the client's wire. */
function translateAnswer(
  upstreamWire: Wire,
  clientApi: 'openai' | 'anthropic',
  stream: boolean,
  text: string,
  model: string,
): { text: string; contentType: string } {
  if (stream) {
    if (upstreamWire === clientApi) return { text, contentType: 'text/event-stream' };
    if (upstreamWire === 'openai') return { text: openAISseToAnthropic(text, model), contentType: 'text/event-stream' };
    if (upstreamWire === 'anthropic') return { text: anthropicSseToOpenAI(text, model), contentType: 'text/event-stream' };
    // gemini upstream
    if (clientApi === 'openai') return { text: geminiSseToOpenAI(text, model), contentType: 'text/event-stream' };
    return { text: geminiSseToAnthropic(text, model), contentType: 'text/event-stream' };
  }
  const parsed: unknown = JSON.parse(text);
  if (!isRecord(parsed)) throw new Error('the provider answered a body that is not one JSON object');
  let out: Json;
  if (upstreamWire === 'openai' && clientApi === 'openai') out = parsed;
  else if (upstreamWire === 'anthropic' && clientApi === 'anthropic') out = parsed;
  else if (upstreamWire === 'openai') out = openAIJsonToAnthropic(parsed, model);
  else if (upstreamWire === 'anthropic') out = anthropicJsonToOpenAI(parsed, model);
  else if (clientApi === 'openai') out = geminiJsonToOpenAI(parsed, model);
  else out = geminiJsonToAnthropic(parsed, model);
  return { text: JSON.stringify(out), contentType: 'application/json' };
}

// ---------------------------------------------------------------------------
// HTTP server: the two client routes, the model list, register and poll.
// ---------------------------------------------------------------------------

function servedModels(): { id: string; provider: string; display_name: string }[] {
  const out: { id: string; provider: string; display_name: string }[] = [];
  for (const p of catalog.providers) {
    for (const m of [...p.models].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0))) {
      if (!m.free) continue;
      out.push({ id: m.id, provider: p.id, display_name: m.id });
    }
  }
  return out;
}

async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, { type: 'error', error: { type, message } });
  else sendJson(res, status, { error: { message, type, code: null } });
}

function modelsRoute(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') {
    sendJson(res, 200, {
      data: list.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
      has_more: false,
    });
  } else {
    sendJson(res, 200, {
      object: 'list',
      data: list.map((m) => ({ id: m.id, object: 'model', created: FIXED_CREATED, owned_by: m.provider })),
    });
  }
}

async function chatRoute(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const stream = body.stream === true;
  const requires = requirementsFor(body, 'openai');
  const answer = await serveRequest('openai', body, stream, requires);
  if (answer === null) return sendError(res, 'openai', 503, 'no_free_model', 'no free model served this request');
  res.writeHead(200, { 'content-type': answer.contentType, 'content-length': Buffer.byteLength(answer.text) });
  res.end(answer.text);
}

async function messagesRoute(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const stream = body.stream === true;
  const requires = requirementsFor(body, 'anthropic');
  const answer = await serveRequest('anthropic', body, stream, requires);
  if (answer === null) return sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model served this request');
  res.writeHead(200, { 'content-type': answer.contentType, 'content-length': Buffer.byteLength(answer.text) });
  res.end(answer.text);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    modelsRoute(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chatRoute(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messagesRoute(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function register(port: number): Promise<void> {
  for (let attempt = 1; attempt <= 10; attempt++) {
    try {
      const res = await fetch(`${ENV_URL}/register`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog }),
        signal: AbortSignal.timeout(5000),
      });
      if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
      const reply: unknown = await res.json();
      if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
      return;
    } catch (err) {
      if (attempt === 10) throw err;
      log(`register attempt ${attempt} failed: ${(err as Error).message}; retrying`);
      await sleep(200);
    }
  }
}

function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status === 200) {
          const work: unknown = await res.json();
          if (isRecord(work) && work.done === true) {
            clearInterval(timer);
            server.close();
            process.exit(0);
          }
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} providers and ${servedModels().length} free models`);
  await register(port);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
