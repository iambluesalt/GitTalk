import type {
  Project,
  ProjectListResponse,
  Conversation,
  Message,
  SSEMessage,
  HealthResponse,
  AppConfig,
  ConfigUpdate,
  ChatModel,
} from "./types";

const BASE = "/api";

async function fetchJSON<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${url}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || body.error || `HTTP ${res.status}`);
  }
  return res.json();
}

// --- SSE Stream Parser ---

export async function* parseSSEStream(
  response: Response
): AsyncGenerator<SSEMessage> {
  if (!response.body) {
    throw new Error("Response body is empty — server may be unreachable");
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    while (true) {
      const end = buffer.indexOf("\n\n");
      if (end === -1) break;

      const block = buffer.slice(0, end);
      buffer = buffer.slice(end + 2);

      let event = "message";
      let data = "";

      for (const line of block.split("\n")) {
        if (line.startsWith("event: ")) event = line.slice(7).trim();
        else if (line.startsWith("data: ")) data = line.slice(6);
      }

      if (data) {
        try {
          yield { event, data: JSON.parse(data) };
        } catch {
          yield { event, data: { raw: data } as Record<string, unknown> };
        }
      }
    }
  }
}

async function postSSE(
  url: string,
  body: Record<string, unknown>,
  signal?: AbortSignal,
): Promise<Response> {
  const res = await fetch(`${BASE}${url}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || err.error || `HTTP ${res.status}`);
  }
  return res;
}

// --- Projects ---

export async function getProjects(): Promise<ProjectListResponse> {
  return fetchJSON<ProjectListResponse>("/projects");
}

export async function getProject(id: string): Promise<Project> {
  return fetchJSON<Project>(`/projects/${id}`);
}

export async function deleteProject(id: string): Promise<void> {
  await fetchJSON(`/projects/${id}`, { method: "DELETE" });
}

// --- Clone (SSE) ---

export async function cloneRepo(githubUrl: string, force = false, signal?: AbortSignal) {
  return postSSE("/clone", { github_url: githubUrl, force }, signal);
}

// --- Index (SSE) ---

export async function indexProject(projectId: string, forceReindex = false, signal?: AbortSignal) {
  return postSSE("/index", { project_id: projectId, force_reindex: forceReindex }, signal);
}

// --- Models ---

export async function getModels(): Promise<{ models: ChatModel[] }> {
  return fetchJSON<{ models: ChatModel[] }>("/models");
}

// --- Chat (SSE) ---

export async function sendMessage(
  projectId: string,
  message: string,
  conversationId?: string,
  signal?: AbortSignal,
  model?: string,
) {
  return postSSE("/chat", {
    project_id: projectId,
    message,
    conversation_id: conversationId,
    stream: true,
    model: model || undefined,
  }, signal);
}

// --- Conversations ---

export async function getConversations(projectId: string): Promise<{
  conversations: Conversation[];
  total: number;
}> {
  return fetchJSON(`/conversations/${projectId}`);
}

export async function getConversationMessages(
  projectId: string,
  conversationId: string
): Promise<{ conversation: Conversation; messages: Message[] }> {
  return fetchJSON(`/conversations/${projectId}/${conversationId}`);
}

export async function deleteConversation(
  projectId: string,
  conversationId: string
): Promise<void> {
  await fetchJSON(`/conversations/${projectId}/${conversationId}`, {
    method: "DELETE",
  });
}

// --- Health ---

export async function getHealth(): Promise<HealthResponse> {
  return fetchJSON<HealthResponse>("/health");
}

// --- Config ---

export async function getConfig(): Promise<AppConfig> {
  return fetchJSON<AppConfig>("/config");
}

export async function updateConfig(update: ConfigUpdate): Promise<AppConfig> {
  return fetchJSON<AppConfig>("/config", {
    method: "PUT",
    body: JSON.stringify(update),
  });
}
