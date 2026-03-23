/**
 * Transforms vague backend error messages into user-friendly messages
 * with actionable guidance.
 */

export interface HumanError {
  title: string;
  message: string;
  hint?: string;
  action?: { label: string; to: string };
}

const ERROR_PATTERNS: Array<{
  pattern: RegExp;
  resolve: (match: RegExpMatchArray) => HumanError;
}> = [
  // LLM connectivity
  {
    pattern: /ollama unavailable and no cloud api configured/i,
    resolve: () => ({
      title: "No LLM available",
      message:
        "Cannot reach Ollama and no cloud API is configured. Chat requires at least one LLM provider.",
      hint: "Start Ollama with `ollama serve`, or configure a cloud API in your .env file.",
      action: { label: "View Settings", to: "/settings" },
    }),
  },
  {
    pattern: /ollama unavailable.*falling back/i,
    resolve: () => ({
      title: "Ollama offline — using cloud fallback",
      message: "Ollama is not responding. Attempting to use the cloud API instead.",
      hint: "Start Ollama with `ollama serve` for local inference.",
    }),
  },
  {
    pattern: /cloud api not configured/i,
    resolve: () => ({
      title: "Cloud API not configured",
      message:
        "The cloud API fallback is not set up. Set CLOUD_API_KEY and CLOUD_API_BASE_URL in your .env file.",
      action: { label: "View Settings", to: "/settings" },
    }),
  },
  {
    pattern: /connect(?:ion)?(?:\s+(?:error|timeout|refused|reset)|\s+attempts?\s+failed)/i,
    resolve: () => ({
      title: "Connection failed",
      message:
        "Cannot connect to the LLM service. The Ollama server may not be running.",
      hint: "Run `ollama serve` in a terminal, then try again.",
      action: { label: "Check Services", to: "/settings" },
    }),
  },
  {
    pattern: /all connection attempts failed/i,
    resolve: () => ({
      title: "Cannot reach LLM",
      message:
        "All connection attempts to the language model failed. Ollama is not running and no cloud API is configured.",
      hint: "Start Ollama (`ollama serve`) or add a cloud API key in your .env file.",
      action: { label: "View Settings", to: "/settings" },
    }),
  },

  // Model errors
  {
    pattern: /model ['"]?([^'"]+)['"]? not found/i,
    resolve: (m) => ({
      title: "Model not found",
      message: `The model "${m[1]}" is not available in Ollama.`,
      hint: `Pull it with: ollama pull ${m[1]}`,
    }),
  },

  // Project state
  {
    pattern: /project must be indexed.*current status:\s*(\w+)/i,
    resolve: (m) => ({
      title: "Project not indexed",
      message: `This project needs to be indexed before you can chat with it. Current status: ${m[1]}.`,
      action: { label: "Index Now", to: "/clone?index=_" },
    }),
  },
  {
    pattern: /project not found/i,
    resolve: () => ({
      title: "Project not found",
      message: "This project doesn't exist or has been deleted.",
      action: { label: "Go to Projects", to: "/projects" },
    }),
  },

  // Network / backend
  {
    pattern: /failed to fetch|network\s*error|err_connection_refused/i,
    resolve: () => ({
      title: "Backend unreachable",
      message:
        "Cannot connect to the GitTalk backend server. Make sure it is running on port 8000.",
      hint: "Start it with: cd app/backend && python main.py",
      action: { label: "Check Services", to: "/settings" },
    }),
  },
  {
    pattern: /http 5\d{2}/i,
    resolve: () => ({
      title: "Server error",
      message: "The backend encountered an internal error. Check the server logs for details.",
    }),
  },
  {
    pattern: /timeout|timed?\s*out/i,
    resolve: () => ({
      title: "Request timed out",
      message:
        "The LLM took too long to respond. This can happen with large contexts or slow hardware.",
      hint: "Try a shorter question, or increase OLLAMA_TIMEOUT in your .env.",
    }),
  },
];

/**
 * Transform a raw error string into a user-friendly HumanError.
 * Falls back to a generic message if no pattern matches.
 */
export function humanizeError(raw: string): HumanError {
  for (const { pattern, resolve } of ERROR_PATTERNS) {
    const match = raw.match(pattern);
    if (match) return resolve(match);
  }

  // Generic fallback — still better than raw error
  return {
    title: "Something went wrong",
    message: raw,
  };
}
