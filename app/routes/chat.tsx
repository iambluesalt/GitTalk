import { useState, useEffect, useRef, useCallback } from "react";
import { useParams, useNavigate, Link } from "react-router";
import {
  Send,
  Plus,
  MessageSquare,
  Trash2,
  ChevronLeft,
  GitBranch,
  Database,
  Loader2,
  PanelRightClose,
  PanelRightOpen,
  Sparkles,
  ArrowDown,
  Brain,
  AlertTriangle,
  Settings,
  ChevronDown,
  Square,
} from "lucide-react";
import type { Route } from "./+types/chat";
import type {
  Project,
  Conversation,
  Message,
  CodeReference,
  HealthResponse,
  ChatModel,
} from "~/lib/types";
import {
  getProject,
  getConversations,
  getConversationMessages,
  deleteConversation,
  sendMessage,
  parseSSEStream,
  getHealth,
  getModels,
} from "~/lib/api";
import ChatMessage from "~/components/ChatMessage";
import ErrorCard from "~/components/ErrorCard";
import { humanizeError } from "~/lib/errors";
import { cn, timeAgo } from "~/lib/utils";

export function meta({}: Route.MetaArgs) {
  return [{ title: "Chat — GitTalk" }];
}

interface ChatEntry {
  role: "user" | "assistant";
  content: string;
  sources?: CodeReference[];
  isStreaming?: boolean;
  isSearching?: boolean;
  error?: ReturnType<typeof humanizeError>;
}

export default function Chat() {
  const { projectId } = useParams();
  const navigate = useNavigate();

  // State
  const [project, setProject] = useState<Project | null>(null);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeConvId, setActiveConvId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatEntry[]>([]);
  const [input, setInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(() =>
    typeof window !== "undefined" ? window.innerWidth >= 1024 : true
  );
  const [loading, setLoading] = useState(true);
  const [showScrollBtn, setShowScrollBtn] = useState(false);
  const [llmStatus, setLlmStatus] = useState<"ok" | "warn" | "checking">("checking");
  const [models, setModels] = useState<ChatModel[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>(() =>
    typeof window !== "undefined" ? localStorage.getItem("gittalk_model") || "" : ""
  );
  const [modelMenuOpen, setModelMenuOpen] = useState(false);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const messagesContainerRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  // Abort SSE stream on unmount
  useEffect(() => () => { abortRef.current?.abort(); }, []);

  // Load project + conversations + health + models
  useEffect(() => {
    if (!projectId) return;
    setLoading(true);

    Promise.all([
      getProject(projectId),
      getConversations(projectId),
      getHealth().catch(() => null),
      getModels().catch(() => ({ models: [] })),
    ])
      .then(([proj, convData, health, modelsData]) => {
        setProject(proj);
        setConversations(convData.conversations || []);
        if (health) {
          const llmOk = health.services.ollama === true || health.services.cloud_api === true;
          setLlmStatus(llmOk ? "ok" : "warn");
        } else {
          setLlmStatus("warn");
        }
        setModels(modelsData.models || []);
        // Auto-select first model if none saved or saved one is gone
        if (modelsData.models?.length) {
          const saved = localStorage.getItem("gittalk_model") || "";
          const still_exists = modelsData.models.some((m: ChatModel) => m.id === saved);
          if (!still_exists) {
            setSelectedModel(modelsData.models[0].id);
            localStorage.setItem("gittalk_model", modelsData.models[0].id);
          }
        }
      })
      .catch(() => navigate("/projects"))
      .finally(() => setLoading(false));
  }, [projectId, navigate]);

  // Load conversation messages
  const loadConversation = useCallback(
    async (convId: string) => {
      if (!projectId) return;
      setActiveConvId(convId);
      try {
        const data = await getConversationMessages(projectId, convId);
        setMessages(
          (data.messages || []).map((m: Message) => ({
            role: m.role,
            content: m.content,
            sources: m.sources,
          }))
        );
      } catch (err) {
        const raw = err instanceof Error ? err.message : "Failed to load messages";
        setMessages([{ role: "assistant", content: "", error: humanizeError(raw) }]);
      }
    },
    [projectId]
  );

  // Auto-scroll
  const scrollToBottom = useCallback((smooth = true) => {
    messagesEndRef.current?.scrollIntoView({
      behavior: smooth ? "smooth" : "instant",
    });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, scrollToBottom]);

  // Scroll detection
  useEffect(() => {
    const container = messagesContainerRef.current;
    if (!container) return;
    const handler = () => {
      const { scrollTop, scrollHeight, clientHeight } = container;
      setShowScrollBtn(scrollHeight - scrollTop - clientHeight > 100);
    };
    container.addEventListener("scroll", handler);
    return () => container.removeEventListener("scroll", handler);
  }, []);

  // Textarea auto-resize
  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 200) + "px";
  }, [input]);

  // Send message
  const handleSend = async () => {
    if (!input.trim() || !projectId || isStreaming) return;

    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    const userMsg = input.trim();
    setInput("");
    setMessages((prev) => [...prev, { role: "user", content: userMsg }]);
    setIsStreaming(true);

    // Add placeholder assistant message (searching state)
    setMessages((prev) => [
      ...prev,
      { role: "assistant", content: "", sources: [], isStreaming: true, isSearching: true },
    ]);

    let currentSources: CodeReference[] = [];
    let fullContent = "";
    let newConvId = activeConvId;

    try {
      const response = await sendMessage(
        projectId, userMsg, activeConvId || undefined,
        controller.signal, selectedModel || undefined,
      );

      for await (const msg of parseSSEStream(response)) {
        switch (msg.event) {
          case "sources":
            currentSources = (msg.data.sources || []) as CodeReference[];
            setMessages((prev) => {
              const next = [...prev];
              const last = next[next.length - 1];
              if (last.role === "assistant") {
                next[next.length - 1] = { ...last, sources: currentSources, isSearching: false };
              }
              return next;
            });
            break;

          case "token":
            fullContent += msg.data.token as string;
            setMessages((prev) => {
              const next = [...prev];
              const last = next[next.length - 1];
              if (last.role === "assistant") {
                next[next.length - 1] = {
                  ...last,
                  content: fullContent,
                  isStreaming: true,
                };
              }
              return next;
            });
            break;

          case "done":
            newConvId = msg.data.conversation_id as string;
            setMessages((prev) => {
              const next = [...prev];
              const last = next[next.length - 1];
              if (last.role === "assistant") {
                next[next.length - 1] = {
                  ...last,
                  content: fullContent,
                  isStreaming: false,
                };
              }
              return next;
            });
            break;

          case "error": {
            const humanized = humanizeError(msg.data.message as string);
            setMessages((prev) => {
              const next = [...prev];
              const last = next[next.length - 1];
              if (last.role === "assistant") {
                next[next.length - 1] = {
                  ...last,
                  content: "",
                  isStreaming: false,
                  error: humanized,
                };
              }
              return next;
            });
            break;
          }
        }
      }
    } catch (err: unknown) {
      if (err instanceof DOMException && err.name === "AbortError") return;
      const raw = err instanceof Error ? err.message : "Failed to send message";
      const humanized = humanizeError(raw);
      setMessages((prev) => {
        const next = [...prev];
        const last = next[next.length - 1];
        if (last.role === "assistant") {
          next[next.length - 1] = {
            ...last,
            content: "",
            isStreaming: false,
            error: humanized,
          };
        }
        return next;
      });
    } finally {
      setIsStreaming(false);
      if (newConvId && newConvId !== activeConvId) {
        setActiveConvId(newConvId);
        // Refresh conversation list
        if (projectId) {
          getConversations(projectId)
            .then((data) => setConversations(data.conversations || []))
            .catch(() => {});
        }
      }
    }
  };

  const handleStop = () => {
    abortRef.current?.abort();
    setIsStreaming(false);
    setMessages((prev) => {
      const next = [...prev];
      const last = next[next.length - 1];
      if (last?.role === "assistant" && last.isStreaming) {
        next[next.length - 1] = { ...last, isStreaming: false };
      }
      return next;
    });
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleNewConversation = () => {
    setActiveConvId(null);
    setMessages([]);
    textareaRef.current?.focus();
  };

  const handleDeleteConversation = async (convId: string) => {
    if (!projectId) return;
    try {
      await deleteConversation(projectId, convId);
      setConversations((prev) => prev.filter((c) => c.id !== convId));
      if (activeConvId === convId) {
        setActiveConvId(null);
        setMessages([]);
      }
    } catch (err) {
      console.error("Failed to delete conversation:", err);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <Loader2 className="w-6 h-6 text-accent animate-spin" />
      </div>
    );
  }

  if (!project) return null;

  const isNotIndexed = project.status !== "indexed";
  const selectedModelObj = models.find((m) => m.id === selectedModel);

  return (
    <div className="flex h-full">
      {/* Chat Area */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Chat header */}
        <div className="flex items-center gap-3 pl-12 pr-4 h-13 border-b border-border-subtle shrink-0">
          {/* Back to projects */}
          <Link
            to="/projects"
            className="p-1.5 rounded-md text-text-muted hover:text-text-secondary hover:bg-hover transition-colors"
            title="Back to projects"
          >
            <ChevronLeft className="w-4 h-4" />
          </Link>

          {/* Divider */}
          <div className="w-px h-5 bg-border-subtle" />

          {/* Project info */}
          <div className="flex items-center gap-2 min-w-0">
            <GitBranch className="w-3.5 h-3.5 text-text-muted shrink-0" />
            <span className="text-sm text-text-primary font-medium truncate">
              {project.name}
            </span>
            {project.analysis?.primary_language && (
              <span className="text-[10px] px-1.5 py-0.5 rounded-md bg-elevated border border-border-subtle text-text-muted font-mono">
                {project.analysis.primary_language}
              </span>
            )}
          </div>

          {/* Spacer */}
          <div className="flex-1" />

          {/* Model selector — now in header */}
          {models.length > 0 && (
            <div className="relative">
              <button
                onClick={() => setModelMenuOpen(!modelMenuOpen)}
                disabled={isStreaming}
                className={cn(
                  "flex items-center gap-2 pl-2.5 pr-2 py-1.5 rounded-lg border text-xs transition-all",
                  modelMenuOpen
                    ? "border-accent/30 bg-accent/5 text-accent"
                    : "border-border-subtle bg-surface text-text-secondary hover:text-text-primary hover:border-border-default",
                  "disabled:opacity-50"
                )}
              >
                <Brain className="w-3.5 h-3.5" />
                <span className="max-w-[160px] truncate font-mono text-[11px]">
                  {selectedModelObj?.name || "Select model"}
                </span>
                {selectedModelObj && (
                  <span className={cn(
                    "w-1.5 h-1.5 rounded-full shrink-0",
                    selectedModelObj.provider === "Ollama" ? "bg-green" : "bg-purple"
                  )} />
                )}
                <ChevronDown className={cn(
                  "w-3 h-3 text-text-ghost transition-transform",
                  modelMenuOpen && "rotate-180"
                )} />
              </button>

              {modelMenuOpen && (
                <>
                  <div className="fixed inset-0 z-40" onClick={() => setModelMenuOpen(false)} />
                  <div className="model-selector-menu absolute top-full right-0 mt-1.5 w-72 rounded-xl border border-border-subtle bg-surface shadow-2xl shadow-black/30 z-50 py-1.5 max-h-72 overflow-y-auto">
                    <div className="px-3 py-1.5 mb-1">
                      <span className="text-[10px] font-medium uppercase tracking-wider text-text-ghost">
                        Available Models
                      </span>
                    </div>
                    {models.map((m) => (
                      <button
                        key={m.id}
                        onClick={() => {
                          setSelectedModel(m.id);
                          localStorage.setItem("gittalk_model", m.id);
                          setModelMenuOpen(false);
                        }}
                        className={cn(
                          "w-full flex items-center gap-2.5 px-3 py-2 text-left text-xs transition-colors",
                          selectedModel === m.id
                            ? "bg-accent/8 text-accent"
                            : "text-text-secondary hover:bg-hover hover:text-text-primary"
                        )}
                      >
                        <span className={cn(
                          "shrink-0 w-2 h-2 rounded-full ring-2",
                          m.provider === "Ollama"
                            ? "bg-green ring-green/20"
                            : "bg-purple ring-purple/20"
                        )} />
                        <div className="flex-1 min-w-0">
                          <span className="block truncate font-mono text-[11px]">{m.name}</span>
                          <span className="block text-[10px] text-text-ghost mt-0.5">{m.provider}</span>
                        </div>
                        {selectedModel === m.id && (
                          <span className="text-[10px] text-accent font-medium">Active</span>
                        )}
                      </button>
                    ))}
                  </div>
                </>
              )}
            </div>
          )}

          {/* Divider */}
          <div className="w-px h-5 bg-border-subtle" />

          {/* Sidebar toggle */}
          <button
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="p-1.5 rounded-md text-text-muted hover:text-text-secondary hover:bg-hover transition-colors"
            title={sidebarOpen ? "Hide conversations" : "Show conversations"}
          >
            {sidebarOpen ? (
              <PanelRightClose className="w-4 h-4" />
            ) : (
              <PanelRightOpen className="w-4 h-4" />
            )}
          </button>
        </div>

        {/* Not indexed warning */}
        {isNotIndexed && (
          <div className="mx-4 mt-3 p-3.5 rounded-xl border border-amber/20 bg-amber/5">
            <div className="flex items-center gap-3">
              <Database className="w-5 h-5 text-amber shrink-0" />
              <div>
                <p className="text-sm text-text-primary font-medium">
                  Project not indexed
                </p>
                <p className="text-xs text-text-secondary mt-0.5">
                  Index this project before chatting.
                </p>
              </div>
              <Link
                to={`/clone?index=${projectId}`}
                className="ml-auto px-3 py-1.5 rounded-lg bg-amber/10 text-amber text-xs font-medium hover:bg-amber/20 transition-colors shrink-0"
              >
                Index Now
              </Link>
            </div>
          </div>
        )}

        {/* LLM offline warning */}
        {!isNotIndexed && llmStatus === "warn" && (
          <div className="mx-4 mt-3 p-3.5 rounded-xl border border-rose/20 bg-rose/5 animate-fade-in">
            <div className="flex items-center gap-3">
              <Brain className="w-5 h-5 text-rose shrink-0" />
              <div>
                <p className="text-sm text-text-primary font-medium">
                  No LLM available
                </p>
                <p className="text-xs text-text-secondary mt-0.5">
                  Ollama is offline and no cloud API is configured. Chat will fail.
                </p>
              </div>
              <Link
                to="/settings"
                className="ml-auto flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-rose/10 text-rose text-xs font-medium hover:bg-rose/20 transition-colors shrink-0"
              >
                <Settings className="w-3 h-3" />
                Settings
              </Link>
            </div>
          </div>
        )}

        {/* Messages wrapper — relative for scroll button */}
        <div className="flex-1 relative min-h-0">
          <div
            ref={messagesContainerRef}
            className="absolute inset-0 overflow-y-auto px-4 py-6"
          >
            <div className="max-w-3xl mx-auto space-y-6">
              {messages.length === 0 ? (
                <EmptyChat projectName={project.name} disabled={isNotIndexed} llmWarn={llmStatus === "warn"} onSuggestion={setInput} />
              ) : (
                messages.map((msg, i) =>
                  msg.error ? (
                    <div key={i} className="flex gap-3">
                      <div className="w-8 shrink-0" />
                      <ErrorCard error={msg.error} className="flex-1" />
                    </div>
                  ) : (
                    <ChatMessage
                      key={i}
                      role={msg.role}
                      content={msg.content}
                      sources={msg.sources}
                      isStreaming={msg.isStreaming}
                      isSearching={msg.isSearching}
                    />
                  )
                )
              )}
              <div ref={messagesEndRef} />
            </div>
          </div>

          {/* Scroll to bottom — absolute positioned outside scroll container */}
          {showScrollBtn && (
            <button
              onClick={() => scrollToBottom()}
              className="absolute bottom-4 left-1/2 -translate-x-1/2 p-2.5 rounded-full bg-surface border border-border-default shadow-xl shadow-black/20 text-text-muted hover:text-text-primary hover:border-border-bright transition-all z-10 animate-slide-up-in"
            >
              <ArrowDown className="w-4 h-4" />
            </button>
          )}
        </div>

        {/* Input */}
        <div className="shrink-0 border-t border-border-subtle px-4 py-3 bg-base/50 backdrop-blur-sm">
          <div className="flex items-end gap-2 max-w-3xl mx-auto">
            <div className="chat-input-wrap relative flex-1 rounded-xl border border-border-default bg-surface transition-all">
              <textarea
                ref={textareaRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder={
                  isNotIndexed
                    ? "Index the project first…"
                    : "Ask about the codebase…"
                }
                disabled={isNotIndexed || isStreaming}
                rows={1}
                className="w-full resize-none px-4 py-3 bg-transparent text-sm text-text-primary placeholder:text-text-ghost focus:outline-none disabled:opacity-50"
              />
            </div>
            <button
              onClick={isStreaming ? handleStop : handleSend}
              disabled={!isStreaming && (!input.trim() || isNotIndexed)}
              className={cn(
                "flex items-center justify-center w-11 h-11 rounded-xl transition-all shrink-0",
                isStreaming
                  ? "bg-rose/10 text-rose border border-rose/20 hover:bg-rose/20"
                  : input.trim() && !isNotIndexed
                    ? "bg-accent text-void hover:bg-accent/90 hover:shadow-lg hover:shadow-accent/20"
                    : "bg-surface text-text-ghost border border-border-subtle cursor-not-allowed"
              )}
              title={isStreaming ? "Stop generating" : "Send message"}
            >
              {isStreaming ? (
                <Square className="w-3.5 h-3.5 fill-current" />
              ) : (
                <Send className="w-4.5 h-4.5" />
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Conversation Sidebar — Right side */}
      <div
        className={cn(
          "flex flex-col border-l border-border-subtle bg-base transition-all duration-300 shrink-0",
          sidebarOpen ? "w-72" : "w-0 overflow-hidden border-l-0"
        )}
      >
        {/* Sidebar header */}
        <div className="flex items-center gap-2 px-3 h-13 border-b border-border-subtle shrink-0">
          <span className="text-xs font-medium uppercase tracking-wider text-text-muted flex-1">
            Conversations
          </span>
          <button
            onClick={handleNewConversation}
            className="p-1.5 rounded-md text-text-muted hover:text-accent hover:bg-accent/10 transition-colors"
            title="New conversation"
          >
            <Plus className="w-4 h-4" />
          </button>
        </div>

        {/* Conversation list */}
        <div className="flex-1 overflow-y-auto py-2 px-2">
          {conversations.length === 0 ? (
            <p className="text-xs text-text-ghost text-center py-6 px-2">
              No conversations yet. Send a message to start.
            </p>
          ) : (
            <div className="space-y-0.5">
              {conversations.map((conv) => (
                <div
                  key={conv.id}
                  className={cn(
                    "group flex items-start gap-2 px-3 py-2.5 rounded-lg cursor-pointer transition-all",
                    activeConvId === conv.id
                      ? "bg-accent/10 text-accent shadow-[inset_0_0_0_1px_rgba(0,212,255,0.1)]"
                      : "text-text-secondary hover:bg-hover hover:text-text-primary"
                  )}
                  onClick={() => loadConversation(conv.id)}
                >
                  <MessageSquare className="w-3.5 h-3.5 shrink-0 mt-0.5" />
                  <div className="flex-1 min-w-0">
                    <span className="text-xs leading-snug block truncate">
                      {conv.title || "Untitled"}
                    </span>
                    <span className="text-[10px] text-text-ghost block mt-0.5">
                      {timeAgo(conv.updated_at || conv.created_at)}
                    </span>
                  </div>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      handleDeleteConversation(conv.id);
                    }}
                    className="p-1 rounded text-text-ghost hover:text-rose opacity-0 group-hover:opacity-100 transition-all shrink-0 mt-0.5"
                  >
                    <Trash2 className="w-3 h-3" />
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function EmptyChat({
  projectName,
  disabled,
  llmWarn,
  onSuggestion,
}: {
  projectName: string;
  disabled: boolean;
  llmWarn: boolean;
  onSuggestion: (text: string) => void;
}) {
  const suggestions = [
    { text: "What does this project do?", icon: "?" },
    { text: "Show me the main entry point", icon: ">" },
    { text: "What are the key functions?", icon: "f" },
    { text: "How is error handling done?", icon: "!" },
  ];

  return (
    <div className="flex flex-col items-center justify-center min-h-[60vh] text-center animate-fade-in">
      <div className="relative mb-6">
        <div className="absolute inset-0 bg-purple/20 rounded-full blur-xl animate-breathe" />
        <div className="relative inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-purple/10 border border-purple/20">
          <Sparkles className="w-8 h-8 text-purple" />
        </div>
      </div>
      <h2 className="font-display font-700 text-2xl text-text-primary mb-2">
        Chat with {projectName}
      </h2>
      <p className="text-sm text-text-secondary max-w-md mb-8 leading-relaxed">
        Ask questions about the codebase. GitTalk uses hybrid search and RAG to
        find relevant code and generate answers.
      </p>

      {!disabled && !llmWarn && (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5 max-w-lg w-full">
          {suggestions.map((s, i) => (
            <button
              key={s.text}
              className={cn(
                "group flex items-center gap-3 px-4 py-3 rounded-xl border border-border-subtle bg-surface text-xs text-text-secondary hover:text-text-primary hover:border-accent/20 hover:bg-accent/5 transition-all text-left animate-fade-in-up",
                `stagger-${i + 1}`
              )}
              onClick={() => onSuggestion(s.text)}
            >
              <span className="flex items-center justify-center w-7 h-7 rounded-lg bg-elevated text-text-ghost font-mono text-[10px] shrink-0 group-hover:bg-accent/10 group-hover:text-accent transition-colors">
                {s.icon}
              </span>
              {s.text}
            </button>
          ))}
        </div>
      )}

      {llmWarn && !disabled && (
        <div className="max-w-sm p-4 rounded-xl border border-amber/20 bg-amber/5 text-left">
          <div className="flex items-start gap-2.5">
            <AlertTriangle className="w-4 h-4 text-amber shrink-0 mt-0.5" />
            <div>
              <p className="text-xs font-medium text-text-primary">
                No language model detected
              </p>
              <p className="text-[11px] text-text-secondary mt-1 leading-relaxed">
                Start Ollama (<code className="font-mono text-accent/70">ollama serve</code>) or configure a cloud API to start chatting.
              </p>
              <Link
                to="/settings"
                className="inline-flex items-center gap-1 mt-2 text-[11px] text-accent hover:underline"
              >
                Open Settings
              </Link>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
