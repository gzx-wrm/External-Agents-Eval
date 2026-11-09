// Execute Pi's pinned file tools; no agent loop, model client, or shell fallback.
import { pathToFileURL } from "node:url";
import path from "node:path";

const [runtime, requestText] = process.argv.slice(2);
process.env.PI_CODING_AGENT_DIR = path.join(runtime, "agent");
process.env.PATH = `${path.join(runtime, "bin")}:${process.env.PATH ?? ""}`;
const root = path.join(runtime, "node_modules/@earendil-works/pi-coding-agent/dist");
const load = (file) => import(pathToFileURL(path.join(root, file)).href);
const factories = {
  read: (await load("core/tools/read.js")).createReadTool,
  grep: (await load("core/tools/grep.js")).createGrepTool,
  find: (await load("core/tools/find.js")).createFindTool,
  ls: (await load("core/tools/ls.js")).createLsTool,
};

if (requestText === "--prepare") {
  const { ensureTool } = await load("utils/tools-manager.js");
  const tools = {};
  for (const name of ["rg", "fd"]) tools[name] = await ensureTool(name);
  console.log(JSON.stringify(tools));
} else if (requestText === "--schemas") {
  console.log(JSON.stringify(Object.values(factories).map((factory) => {
    const tool = factory("/testbed");
    return { type: "function", function: {
      name: tool.name, description: tool.description, parameters: tool.parameters,
    } };
  })));
} else {
  const request = JSON.parse(requestText);
  try {
    if (!Object.hasOwn(factories, request.name)) throw new Error(`Unknown tool '${request.name}'`);
    const { resolveToCwd, resolveReadPathAsync } = await load("core/tools/path-utils.js");
    const resolvedPath = request.name === "read"
      ? await resolveReadPathAsync(request.arguments.path, request.cwd)
      : resolveToCwd(request.arguments.path || ".", request.cwd);
    const result = await factories[request.name](request.cwd).execute(
      request.tool_call_id, request.arguments, undefined, undefined, { model: { input: ["text"] } },
    );
    console.log(JSON.stringify({
      output: result.content.filter((item) => item.type === "text").map((item) => item.text).join("\n"),
      returncode: 0, exception_info: "", resolved_path: resolvedPath, details: result.details,
      non_text_content: result.content.some((item) => item.type !== "text"),
      source_text_returned: !result.content.some((item) => item.type !== "text")
        && !result.details?.truncation?.firstLineExceedsLimit,
    }));
  } catch (error) {
    console.log(JSON.stringify({ output: error.message, returncode: 1, exception_info: error.message }));
  }
}
