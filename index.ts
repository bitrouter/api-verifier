#!/usr/bin/env bun
import { generateText, streamText, tool, stepCountIs } from "ai";
import { createOpenAI } from "@ai-sdk/openai";
import { createAnthropic } from "@ai-sdk/anthropic";
import { z } from "zod";

// ─── Types ────────────────────────────────────────────────────────────────────

type ApiType = "openai.chat" | "openai.responses" | "anthropic";
type TestName = "text" | "stream" | "text-tools" | "stream-tools";

interface Config {
  type: ApiType;
  apiBase: string;
  apiKey: string;
  model: string;
  tests: TestName[];
}

interface TestResult {
  name: TestName;
  passed: boolean;
  output?: string;
  error?: string;
  durationMs: number;
}

// ─── CLI arg parsing ──────────────────────────────────────────────────────────

function parseArgs(): Config {
  const argv = process.argv.slice(2);
  const get = (flag: string) => {
    const i = argv.indexOf(flag);
    return i !== -1 ? argv[i + 1] : undefined;
  };
  const getAll = (flag: string): string[] => {
    const results: string[] = [];
    for (let i = 0; i < argv.length; i++) {
      if (argv[i] === flag && argv[i + 1]) results.push(argv[++i]!);
    }
    return results;
  };

  if (argv.includes("--help") || argv.includes("-h")) {
    console.log(`
Usage: api-verifier --type <type> --api-base <url> --api-key <key> --model <model> [--test <test>]...

Options:
  --type      API type: openai.chat | openai.responses | anthropic
  --api-base  Base URL of the API (e.g. http://localhost:8000/v1)
  --api-key   API key
  --model     Model name
  --test      Test(s) to run: text | stream | text-tools | stream-tools
              (default: all four tests)
  --help      Show this help message
`);
    process.exit(0);
  }

  const type = get("--type") as ApiType | undefined;
  const apiBase = get("--api-base");
  const apiKey = get("--api-key");
  const model = get("--model");
  const rawTests = getAll("--test") as TestName[];

  const errors: string[] = [];
  if (!type) errors.push("--type is required");
  else if (!["openai.chat", "openai.responses", "anthropic"].includes(type))
    errors.push(`--type must be one of: openai.chat, openai.responses, anthropic`);
  if (!apiBase) errors.push("--api-base is required");
  if (!apiKey) errors.push("--api-key is required");
  if (!model) errors.push("--model is required");
  for (const t of rawTests) {
    if (!["text", "stream", "text-tools", "stream-tools"].includes(t))
      errors.push(`Unknown test "${t}". Must be one of: text, stream, text-tools, stream-tools`);
  }

  if (errors.length) {
    for (const e of errors) console.error(`Error: ${e}`);
    process.exit(1);
  }

  return {
    type: type!,
    apiBase: apiBase!,
    apiKey: apiKey!,
    model: model!,
    tests: rawTests.length ? rawTests : ["text", "stream", "text-tools", "stream-tools"],
  };
}

// ─── Model factory ────────────────────────────────────────────────────────────

function createModel(config: Config) {
  if (config.type === "openai.chat") {
    return createOpenAI({ baseURL: config.apiBase, apiKey: config.apiKey }).chat(config.model);
  }
  if (config.type === "openai.responses") {
    return createOpenAI({ baseURL: config.apiBase, apiKey: config.apiKey }).responses(config.model);
  }
  // anthropic
  return createAnthropic({ baseURL: config.apiBase, apiKey: config.apiKey })(config.model);
}

// ─── Shared tool definition ───────────────────────────────────────────────────

const weatherTool = tool({
  description: "Get the current weather for a location",
  inputSchema: z.object({
    location: z.string().describe("City and country, e.g. 'London, UK'"),
  }),
  execute: async ({ location }) => ({
    location,
    temperature: 22,
    unit: "celsius",
    condition: "sunny",
  }),
});

// ─── Individual tests ─────────────────────────────────────────────────────────

async function runText(model: ReturnType<typeof createModel>): Promise<string> {
  const result = await generateText({
    model,
    prompt: "Reply with exactly: Hello, world!",
    maxOutputTokens: 32,
  });
  return result.text.trim();
}

async function runStream(model: ReturnType<typeof createModel>): Promise<string> {
  const result = streamText({
    model,
    prompt: "Reply with exactly: Hello, world!",
    maxOutputTokens: 32,
  });
  let text = "";
  for await (const chunk of result.textStream) {
    text += chunk;
  }
  return text.trim();
}

async function runTextTools(model: ReturnType<typeof createModel>): Promise<string> {
  const result = await generateText({
    model,
    prompt: "What is the weather in Tokyo, Japan? Use the get_weather tool.",
    maxOutputTokens: 256,
    tools: { get_weather: weatherTool },
    stopWhen: stepCountIs(3),
  });
  const toolCalls = result.steps.flatMap((s) => s.toolCalls);
  const toolResults = result.steps.flatMap((s) => s.toolResults);
  if (toolCalls.length === 0) throw new Error("Model did not call any tools");
  return `tool_calls=${toolCalls.length} tool_results=${toolResults.length} text="${result.text.trim().slice(0, 80)}"`;
}

async function runStreamTools(model: ReturnType<typeof createModel>): Promise<string> {
  const result = streamText({
    model,
    prompt: "What is the weather in Paris, France? Use the get_weather tool.",
    maxOutputTokens: 256,
    tools: { get_weather: weatherTool },
    stopWhen: stepCountIs(3),
  });
  // consume the full stream
  for await (const _ of result.fullStream) { /* drain */ }
  const toolCalls = (await result.toolCalls);
  if (toolCalls.length === 0) throw new Error("Model did not call any tools");
  const text = (await result.text).trim().slice(0, 80);
  return `tool_calls=${toolCalls.length} text="${text}"`;
}

// ─── Runner ───────────────────────────────────────────────────────────────────

async function runTest(name: TestName, model: ReturnType<typeof createModel>): Promise<TestResult> {
  const start = Date.now();
  try {
    let output: string;
    switch (name) {
      case "text":        output = await runText(model); break;
      case "stream":      output = await runStream(model); break;
      case "text-tools":  output = await runTextTools(model); break;
      case "stream-tools":output = await runStreamTools(model); break;
    }
    return { name, passed: true, output, durationMs: Date.now() - start };
  } catch (err) {
    const error = err instanceof Error ? err.message : String(err);
    return { name, passed: false, error, durationMs: Date.now() - start };
  }
}

// ─── Main ─────────────────────────────────────────────────────────────────────

const config = parseArgs();
const model = createModel(config);

const GREEN = "\x1b[32m";
const RED = "\x1b[31m";
const YELLOW = "\x1b[33m";
const BOLD = "\x1b[1m";
const DIM = "\x1b[2m";
const RESET = "\x1b[0m";

console.log(`\n${BOLD}api-verifier${RESET}`);
console.log(`${DIM}type:     ${config.type}${RESET}`);
console.log(`${DIM}api-base: ${config.apiBase}${RESET}`);
console.log(`${DIM}model:    ${config.model}${RESET}`);
console.log(`${DIM}tests:    ${config.tests.join(", ")}${RESET}\n`);

const results: TestResult[] = [];

for (const testName of config.tests) {
  process.stdout.write(`  ${YELLOW}●${RESET} ${testName.padEnd(14)} `);
  const result = await runTest(testName, model);
  results.push(result);

  if (result.passed) {
    console.log(`${GREEN}✓ passed${RESET} ${DIM}(${result.durationMs}ms)${RESET}`);
    if (result.output) console.log(`    ${DIM}→ ${result.output}${RESET}`);
  } else {
    console.log(`${RED}✗ failed${RESET} ${DIM}(${result.durationMs}ms)${RESET}`);
    console.log(`    ${RED}${result.error}${RESET}`);
  }
}

const passed = results.filter((r) => r.passed).length;
const failed = results.filter((r) => !r.passed).length;

console.log(`\n${BOLD}Results: ${passed === results.length ? GREEN : RED}${passed}/${results.length} passed${RESET}\n`);

process.exit(failed > 0 ? 1 : 0);
