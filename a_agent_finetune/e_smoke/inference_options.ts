import { appendFileSync } from "node:fs";

// Match the F native baseline and keep the actual provider payload auditable.
export default function (pi) {
  pi.on("before_provider_request", (event) => {
    const payload = { ...event.payload, temperature: 0, seed: 42 };
    appendFileSync(
      "/work/sessions/provider-requests.jsonl",
      `${JSON.stringify(payload)}\n`,
      "utf8",
    );
    return payload;
  });
}
