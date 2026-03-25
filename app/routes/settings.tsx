import { useState, useEffect, useCallback, useMemo, useRef } from "react";
import { createPortal } from "react-dom";
import {
  RefreshCw,
  Brain,
  Cloud,
  Database,
  Server,
  CircleCheck,
  CircleX,
  CircleDot,
  GitBranch,
  Cpu,
  HardDrive,
  Layers,
  Search,
  Key,
  Timer,
  AlertTriangle,
  Save,
  RotateCcw,
  Eye,
  EyeOff,
  Trash2,
  CheckCircle2,
  XCircle,
  ChevronDown,
  Info,
  X,
} from "lucide-react";
import type { Route } from "./+types/settings";
import type { HealthResponse, AppConfig, ConfigUpdate } from "~/lib/types";
import { getHealth, getConfig, updateConfig } from "~/lib/api";
import { cn } from "~/lib/utils";

export function meta({}: Route.MetaArgs) {
  return [{ title: "Settings — GitTalk" }];
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface FormData {
  llm_provider: string;
  ollama_base_url: string;
  ollama_model: string;
  ollama_embed_model: string;
  ollama_timeout: number;
  cloud_api_provider: string;
  cloud_api_key: string;
  cloud_api_base_url: string;
  cloud_model: string;
  github_token: string;
  max_repo_size_mb: number;
  clone_timeout_seconds: number;
  max_context_tokens: number;
  max_search_results: number;
  chunk_max_tokens: number;
  retrieval_candidates: number;
  min_relevance_score: number;
  chunk_overlap_lines: number;
  embedding_dimensions: number;
  embedding_batch_size: number;
  indexing_workers: number;
}

const SECRET_KEYS = ["cloud_api_key", "github_token"] as const;

function configToForm(c: AppConfig): FormData {
  return {
    llm_provider: c.llm_provider,
    ollama_base_url: c.ollama_base_url,
    ollama_model: c.ollama_model,
    ollama_embed_model: c.ollama_embed_model,
    ollama_timeout: c.ollama_timeout,
    cloud_api_provider: c.cloud_api_provider || "",
    cloud_api_key: "",
    cloud_api_base_url: c.cloud_api_base_url || "",
    cloud_model: c.cloud_model || "",
    github_token: "",
    max_repo_size_mb: c.max_repo_size_mb,
    clone_timeout_seconds: c.clone_timeout_seconds,
    max_context_tokens: c.max_context_tokens,
    max_search_results: c.max_search_results,
    chunk_max_tokens: c.chunk_max_tokens,
    retrieval_candidates: c.retrieval_candidates,
    min_relevance_score: c.min_relevance_score,
    chunk_overlap_lines: c.chunk_overlap_lines,
    embedding_dimensions: c.embedding_dimensions,
    embedding_batch_size: c.embedding_batch_size,
    indexing_workers: c.indexing_workers,
  };
}

// ---------------------------------------------------------------------------
// Service health metadata
// ---------------------------------------------------------------------------

const SERVICE_INFO: Record<
  string,
  {
    label: string;
    icon: typeof Server;
    critical: boolean;
    description: string;
    offlineHelp: string;
  }
> = {
  metadata_db: {
    label: "SQLite Database",
    icon: Database,
    critical: true,
    description: "Stores project metadata, conversations, and messages.",
    offlineHelp:
      "The database file may be corrupted or missing. Restart the backend.",
  },
  vector_db: {
    label: "LanceDB Vector Store",
    icon: HardDrive,
    critical: true,
    description: "Stores code embeddings for hybrid search (BM25 + vector).",
    offlineHelp: "Check that the data/lancedb/ directory exists and is writable.",
  },
  ollama: {
    label: "Ollama (LLM)",
    icon: Brain,
    critical: false,
    description: "Local language model for code Q&A.",
    offlineHelp:
      "Run `ollama serve` in a terminal. Then pull a model: `ollama pull qwen2.5-coder:7b`.",
  },
  cloud_api: {
    label: "Cloud API (Fallback)",
    icon: Cloud,
    critical: false,
    description: "Cloud LLM fallback when Ollama is unavailable.",
    offlineHelp:
      "Set CLOUD_API_KEY, CLOUD_API_BASE_URL, and CLOUD_MODEL in your .env file.",
  },
  ollama_embed: {
    label: "Embedding Model",
    icon: Layers,
    critical: false,
    description: "Converts code chunks into vector embeddings for search.",
    offlineHelp:
      "Run `ollama pull nomic-embed-text` to download the embedding model.",
  },
};

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export default function SettingsPage() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [config, setConfig] = useState<AppConfig | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [backendDown, setBackendDown] = useState(false);

  // Form state
  const [form, setForm] = useState<FormData | null>(null);
  const [original, setOriginal] = useState<FormData | null>(null);
  const [secretsCleared, setSecretsCleared] = useState<Set<string>>(new Set());

  // Save state
  const [saving, setSaving] = useState(false);
  const [toast, setToast] = useState<{
    type: "success" | "error";
    message: string;
  } | null>(null);

  // ------ data loading ------
  const load = useCallback(async () => {
    try {
      const h = await getHealth();
      setHealth(h);
      setBackendDown(false);
      try {
        const c = await getConfig();
        setConfig(c);
        const f = configToForm(c);
        setForm(f);
        setOriginal(f);
        setSecretsCleared(new Set());
      } catch {
        setConfig(null);
      }
    } catch {
      setBackendDown(true);
      setHealth(null);
      setConfig(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const handleRefresh = async () => {
    setRefreshing(true);
    await load();
    setRefreshing(false);
  };

  // ------ form helpers ------
  const set = <K extends keyof FormData>(key: K, value: FormData[K]) => {
    setForm((prev) => (prev ? { ...prev, [key]: value } : null));
    // If user types in a secret field, remove the "clear" flag
    if ((SECRET_KEYS as readonly string[]).includes(key)) {
      setSecretsCleared((prev) => {
        const next = new Set(prev);
        next.delete(key);
        return next;
      });
    }
  };

  const clearSecret = (key: string) => {
    setForm((prev) => (prev ? { ...prev, [key]: "" } : null));
    setSecretsCleared((prev) => new Set(prev).add(key));
  };

  // ------ change detection ------
  const hasChanges = useMemo(() => {
    if (!form || !original) return false;
    if (secretsCleared.size > 0) return true;
    return JSON.stringify(form) !== JSON.stringify(original);
  }, [form, original, secretsCleared]);

  // ------ save / discard ------
  const handleDiscard = () => {
    if (original) setForm({ ...original });
    setSecretsCleared(new Set());
  };

  const handleSave = async () => {
    if (!form || !original || !config) return;
    setSaving(true);
    setToast(null);

    // Build partial update: only changed fields
    const payload: Record<string, unknown> = {};
    const orig = original as unknown as Record<string, unknown>;
    for (const [key, value] of Object.entries(form)) {
      if ((SECRET_KEYS as readonly string[]).includes(key)) {
        if (value !== "") payload[key] = value;
      } else if (value !== orig[key]) {
        payload[key] = value;
      }
    }

    // Cleared secrets → send null to remove from .env
    for (const key of secretsCleared) {
      payload[key] = null;
    }

    if (Object.keys(payload).length === 0) {
      setSaving(false);
      return;
    }

    try {
      const updated = await updateConfig(payload as ConfigUpdate);
      setConfig(updated);
      const f = configToForm(updated);
      setForm(f);
      setOriginal(f);
      setSecretsCleared(new Set());
      setToast({ type: "success", message: "Settings saved and applied" });
      // Refresh health since config changes might affect service status
      try {
        const h = await getHealth();
        setHealth(h);
      } catch { /* ignore */ }
    } catch (e: unknown) {
      setToast({
        type: "error",
        message: e instanceof Error ? e.message : "Failed to save settings",
      });
    } finally {
      setSaving(false);
    }
  };

  // Auto-dismiss toast
  useEffect(() => {
    if (!toast) return;
    const t = setTimeout(() => setToast(null), 4000);
    return () => clearTimeout(t);
  }, [toast]);

  // ------ derived ------
  const services = health?.services || {};
  const llmAvailable = services.ollama === true || services.cloud_api === true;
  const canChat =
    llmAvailable &&
    services.metadata_db === true &&
    services.vector_db === true;
  const canIndex =
    services.metadata_db === true &&
    services.vector_db === true &&
    (services.ollama_embed === true || services.ollama_embed === null);

  // ------ render ------
  return (
    <div className="h-full overflow-y-auto">
      <div
        className={cn("max-w-3xl mx-auto px-6 py-8", hasChanges && "pb-28")}
      >
        {/* Header */}
        <div className="flex items-center justify-between mb-8 animate-fade-in">
          <div>
            <h1 className="font-display font-800 text-2xl text-text-primary">
              Settings
            </h1>
            <p className="text-sm text-text-secondary mt-1">
              Service health, configuration, and diagnostics
            </p>
          </div>
          <button
            onClick={handleRefresh}
            disabled={refreshing}
            className="flex items-center gap-2 px-3 py-2 rounded-lg bg-surface border border-border-default text-text-secondary text-sm hover:bg-hover transition-colors disabled:opacity-50"
          >
            <RefreshCw
              className={cn("w-4 h-4", refreshing && "animate-spin")}
            />
            Refresh
          </button>
        </div>

        {/* Toast */}
        {toast && (
          <div
            className={cn(
              "mb-6 px-4 py-3 rounded-xl border flex items-center gap-3 animate-fade-in",
              toast.type === "success" &&
                "border-green/20 bg-green/5 text-green",
              toast.type === "error" && "border-rose/20 bg-rose/5 text-rose"
            )}
          >
            {toast.type === "success" ? (
              <CheckCircle2 className="w-4 h-4 shrink-0" />
            ) : (
              <XCircle className="w-4 h-4 shrink-0" />
            )}
            <span className="text-sm">{toast.message}</span>
          </div>
        )}

        {/* Backend down banner */}
        {backendDown && (
          <div className="mb-6 p-4 rounded-xl border border-rose/20 bg-rose/5 animate-fade-in">
            <div className="flex items-start gap-3">
              <AlertTriangle className="w-5 h-5 text-rose shrink-0 mt-0.5" />
              <div>
                <p className="text-sm font-medium text-text-primary">
                  Backend server is not running
                </p>
                <p className="text-xs text-text-secondary mt-1 leading-relaxed">
                  Cannot connect to the GitTalk backend at{" "}
                  <code className="font-mono text-rose/80 bg-rose/5 px-1 rounded">
                    localhost:8000
                  </code>
                  . Start it with:
                </p>
                <pre className="mt-2 text-xs font-mono bg-base border border-border-subtle rounded-lg px-3 py-2 text-text-secondary">
                  cd app/backend && python main.py
                </pre>
              </div>
            </div>
          </div>
        )}

        {/* ============================================================== */}
        {/*  Service Health  (read-only)                                    */}
        {/* ============================================================== */}
        <section className="mb-8 animate-fade-in stagger-1">
          <h2 className="font-display font-700 text-base text-text-primary mb-4 flex items-center gap-2">
            <Server className="w-4 h-4 text-text-muted" />
            Service Health
          </h2>

          <div className="space-y-2">
            {Object.entries(SERVICE_INFO).map(
              ([
                key,
                { label, icon: Icon, critical, description, offlineHelp },
              ]) => {
                const value = services[key];
                const isOnline = value === true;
                const isOffline = value === false;
                const isNA = value === null || value === undefined;

                return (
                  <div
                    key={key}
                    className={cn(
                      "flex items-start gap-3 p-3.5 rounded-xl border transition-colors",
                      isOnline && "border-border-subtle bg-surface",
                      isOffline &&
                        critical &&
                        "border-rose/20 bg-rose/5",
                      isOffline &&
                        !critical &&
                        "border-amber/20 bg-amber/5",
                      isNA && "border-border-subtle bg-surface/50"
                    )}
                  >
                    <div
                      className={cn(
                        "flex items-center justify-center w-8 h-8 rounded-lg shrink-0 mt-0.5",
                        isOnline && "bg-green/10",
                        isOffline && "bg-rose/10",
                        isNA && "bg-elevated"
                      )}
                    >
                      <Icon
                        className={cn(
                          "w-4 h-4",
                          isOnline && "text-green",
                          isOffline && "text-rose",
                          isNA && "text-text-ghost"
                        )}
                      />
                    </div>

                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2">
                        <span className="text-sm font-medium text-text-primary">
                          {label}
                        </span>
                        {critical && (
                          <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-rose/10 text-rose/70 font-medium">
                            required
                          </span>
                        )}
                        <span className="ml-auto flex items-center gap-1.5">
                          {isOnline && (
                            <>
                              <CircleCheck className="w-3.5 h-3.5 text-green" />
                              <span className="text-xs font-mono text-green">
                                online
                              </span>
                            </>
                          )}
                          {isOffline && (
                            <>
                              <CircleX className="w-3.5 h-3.5 text-rose" />
                              <span className="text-xs font-mono text-rose">
                                offline
                              </span>
                            </>
                          )}
                          {isNA && (
                            <>
                              <CircleDot className="w-3.5 h-3.5 text-text-ghost" />
                              <span className="text-xs font-mono text-text-ghost">
                                n/a
                              </span>
                            </>
                          )}
                        </span>
                      </div>
                      <p className="text-xs text-text-muted mt-0.5">
                        {description}
                      </p>
                      {isOffline && (
                        <div className="mt-2 p-2 rounded-lg bg-base border border-border-subtle">
                          <p className="text-xs text-text-secondary leading-relaxed">
                            <span className="font-medium text-amber">
                              Fix:{" "}
                            </span>
                            {offlineHelp}
                          </p>
                        </div>
                      )}
                    </div>
                  </div>
                );
              }
            )}
          </div>

          {/* Capabilities summary */}
          <div className="mt-4 grid grid-cols-3 gap-3">
            <CapabilityCard
              label="Chat"
              ready={canChat}
              requirement="LLM + DB"
            />
            <CapabilityCard
              label="Clone"
              ready={services.metadata_db === true}
              requirement="Database"
            />
            <CapabilityCard
              label="Index"
              ready={canIndex}
              requirement="DB + Embeddings"
            />
          </div>
        </section>

        {/* ============================================================== */}
        {/*  LLM Configuration  (editable)                                 */}
        {/* ============================================================== */}
        {config && form && (
          <section className="mb-8 animate-fade-in stagger-2">
            <h2 className="font-display font-700 text-base text-text-primary mb-4 flex items-center gap-2">
              <Brain className="w-4 h-4 text-text-muted" />
              LLM Configuration
            </h2>
            <div className="rounded-xl border border-border-subtle bg-surface overflow-hidden">
              {/* Provider */}
              <FormRow label="Provider Mode" icon={Cpu} hint="Hybrid uses cloud (fast) with local Ollama fallback. Cloud-only for minimum latency.">
                <div className="relative">
                  <select
                    value={form.llm_provider}
                    onChange={(e) =>
                      set(
                        "llm_provider",
                        e.target.value as FormData["llm_provider"]
                      )
                    }
                    className={cn(selectCls)}
                  >
                    <option value="hybrid">Hybrid (Ollama + Cloud fallback)</option>
                    <option value="ollama">Ollama only</option>
                    <option value="cloud">Cloud API only</option>
                  </select>
                  <ChevronDown className="absolute right-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-text-ghost pointer-events-none" />
                </div>
              </FormRow>

              {/* Ollama URL */}
              <FormRow label="Ollama URL" icon={Server}>
                <input
                  type="text"
                  value={form.ollama_base_url}
                  onChange={(e) => set("ollama_base_url", e.target.value)}
                  placeholder="http://localhost:11434"
                  className={monoCls}
                />
              </FormRow>

              {/* Chat Model */}
              <FormRow label="Chat Model" icon={Brain} hint="Local Ollama model for code Q&A. Used as fallback in hybrid mode.">
                <input
                  type="text"
                  value={form.ollama_model}
                  onChange={(e) => set("ollama_model", e.target.value)}
                  placeholder="qwen2.5-coder:7b"
                  className={monoCls}
                />
              </FormRow>

              {/* Embed Model */}
              <FormRow label="Embed Model" icon={Layers} hint="Must match the model used during indexing. Changing requires re-index.">
                <input
                  type="text"
                  value={form.ollama_embed_model}
                  onChange={(e) => set("ollama_embed_model", e.target.value)}
                  placeholder="nomic-embed-text"
                  className={monoCls}
                />
              </FormRow>

              {/* Ollama Timeout */}
              <FormRow label="Ollama Timeout" icon={Timer} hint="Low: 30s (fast models) · Med: 120s · High: 300s (large models)">
                <div className="flex items-center gap-2">
                  <input
                    type="number"
                    value={form.ollama_timeout}
                    onChange={(e) =>
                      set("ollama_timeout", parseInt(e.target.value) || 0)
                    }
                    min={10}
                    max={600}
                    className={cn(inputCls, "w-24 font-mono text-xs")}
                  />
                  <span className="text-xs text-text-ghost">seconds</span>
                </div>
              </FormRow>

              {/* Cloud API Provider */}
              <FormRow label="Cloud Provider" icon={Cloud}>
                <input
                  type="text"
                  value={form.cloud_api_provider}
                  onChange={(e) => set("cloud_api_provider", e.target.value)}
                  placeholder="e.g. gemini, openrouter"
                  className={monoCls}
                />
              </FormRow>

              {/* Cloud API Key */}
              <FormRow label="Cloud API Key" icon={Key}>
                <SecretInput
                  value={form.cloud_api_key}
                  onChange={(v) => set("cloud_api_key", v)}
                  configured={config.cloud_api_configured}
                  cleared={secretsCleared.has("cloud_api_key")}
                  onClear={() => clearSecret("cloud_api_key")}
                  placeholder="Enter API key"
                />
              </FormRow>

              {/* Cloud API URL */}
              <FormRow label="Cloud API URL" icon={Server}>
                <input
                  type="text"
                  value={form.cloud_api_base_url}
                  onChange={(e) => set("cloud_api_base_url", e.target.value)}
                  placeholder="https://..."
                  className={monoCls}
                />
              </FormRow>

              {/* Cloud Model */}
              <FormRow label="Cloud Model" icon={Brain} hint="e.g. gemini-2.5-flash (fastest), gemini-2.0-flash, gpt-4o-mini">
                <input
                  type="text"
                  value={form.cloud_model}
                  onChange={(e) => set("cloud_model", e.target.value)}
                  placeholder="e.g. gemini-2.0-flash"
                  className={monoCls}
                />
              </FormRow>

              {/* GitHub Token */}
              <FormRow label="GitHub Token" icon={Key} last>
                <SecretInput
                  value={form.github_token}
                  onChange={(v) => set("github_token", v)}
                  configured={config.github_token_configured}
                  cleared={secretsCleared.has("github_token")}
                  onClear={() => clearSecret("github_token")}
                  placeholder="ghp_... (enables private repos)"
                />
              </FormRow>
            </div>
          </section>
        )}

        {/* ============================================================== */}
        {/*  Search & RAG  (editable)                                      */}
        {/* ============================================================== */}
        {config && form && (
          <section className="mb-8 animate-fade-in stagger-3">
            <h2 className="font-display font-700 text-base text-text-primary mb-4 flex items-center gap-2">
              <Search className="w-4 h-4 text-text-muted" />
              Search & RAG
            </h2>
            <div className="rounded-xl border border-border-subtle bg-surface overflow-hidden">
              {/* Context Window */}
              <FormRow label="Context Window" icon={Layers} hint="Low: 4096 · Med: 16384 · High: 32768+ — larger = more code context per query">
                <div className="flex items-center gap-2">
                  <input
                    type="number"
                    value={form.max_context_tokens}
                    onChange={(e) =>
                      set(
                        "max_context_tokens",
                        parseInt(e.target.value) || 0
                      )
                    }
                    min={1024}
                    max={65536}
                    step={1024}
                    className={cn(inputCls, "w-28 font-mono text-xs")}
                  />
                  <span className="text-xs text-text-ghost">tokens</span>
                </div>
              </FormRow>

              {/* Search Results */}
              <FormRow label="Search Results" icon={Search} hint="Top: Low 3 (focused) · Med 8 · High 15 (broad). Candidates: Low 10 · Med 30 · High 100 (re-ranking pool)">
                <div className="flex items-center gap-2">
                  <span className="text-xs text-text-ghost">Top</span>
                  <input
                    type="number"
                    value={form.max_search_results}
                    onChange={(e) =>
                      set(
                        "max_search_results",
                        parseInt(e.target.value) || 0
                      )
                    }
                    min={1}
                    max={50}
                    className={cn(inputCls, "w-20 font-mono text-xs")}
                  />
                  <span className="text-xs text-text-ghost">from</span>
                  <input
                    type="number"
                    value={form.retrieval_candidates}
                    onChange={(e) =>
                      set(
                        "retrieval_candidates",
                        parseInt(e.target.value) || 0
                      )
                    }
                    min={5}
                    max={200}
                    className={cn(inputCls, "w-20 font-mono text-xs")}
                  />
                  <span className="text-xs text-text-ghost">candidates</span>
                </div>
              </FormRow>

              {/* Min Relevance Score */}
              <FormRow label="Min Relevance" icon={Search} hint="Low: 0.05 (permissive) · Med: 0.15 (balanced) · High: 0.35 (strict) — filters weak matches from results">
                <div className="flex items-center gap-2">
                  <input
                    type="number"
                    value={form.min_relevance_score}
                    onChange={(e) =>
                      set(
                        "min_relevance_score",
                        parseFloat(e.target.value) || 0
                      )
                    }
                    min={0}
                    max={1}
                    step={0.05}
                    className={cn(inputCls, "w-24 font-mono text-xs")}
                  />
                  <span className="text-xs text-text-ghost">threshold (0–1)</span>
                </div>
              </FormRow>

              {/* Chunk Size */}
              <FormRow label="Chunk Size" icon={HardDrive} hint="Low: 300 (precise) · Med: 800 (balanced) · High: 2000 (full functions) — tokens per code chunk">
                <div className="flex items-center gap-2">
                  <input
                    type="number"
                    value={form.chunk_max_tokens}
                    onChange={(e) =>
                      set("chunk_max_tokens", parseInt(e.target.value) || 0)
                    }
                    min={100}
                    max={5000}
                    step={100}
                    className={cn(inputCls, "w-24 font-mono text-xs")}
                  />
                  <span className="text-xs text-text-ghost">tokens max</span>
                </div>
              </FormRow>

              {/* Chunk Overlap */}
              <FormRow label="Chunk Overlap" icon={Layers} hint="Low: 0 · Med: 3 · High: 8 — lines shared between split chunks to preserve context at boundaries">
                <div className="flex items-center gap-2">
                  <input
                    type="number"
                    value={form.chunk_overlap_lines}
                    onChange={(e) =>
                      set(
                        "chunk_overlap_lines",
                        parseInt(e.target.value) || 0
                      )
                    }
                    min={0}
                    max={20}
                    className={cn(inputCls, "w-20 font-mono text-xs")}
                  />
                  <span className="text-xs text-text-ghost">lines</span>
                </div>
              </FormRow>

              {/* Embedding Dimensions */}
              <FormRow label="Embed Dims" icon={Cpu} hint="768 for nomic-embed-text, 1536 for text-embedding-3-large. Must match model output. Batch: Low 16 · Med 64 · High 256">
                <div className="flex items-center gap-2">
                  <input
                    type="number"
                    value={form.embedding_dimensions}
                    onChange={(e) =>
                      set(
                        "embedding_dimensions",
                        parseInt(e.target.value) || 0
                      )
                    }
                    min={128}
                    max={4096}
                    className={cn(inputCls, "w-24 font-mono text-xs")}
                  />
                  <span className="text-xs text-text-ghost">dims, batch</span>
                  <input
                    type="number"
                    value={form.embedding_batch_size}
                    onChange={(e) =>
                      set(
                        "embedding_batch_size",
                        parseInt(e.target.value) || 0
                      )
                    }
                    min={1}
                    max={512}
                    className={cn(inputCls, "w-20 font-mono text-xs")}
                  />
                </div>
              </FormRow>

              {/* Max Repo Size */}
              <FormRow label="Max Repo Size" icon={GitBranch} hint="Low: 50MB · Med: 200MB · High: 1000MB — maximum allowed clone size">
                <div className="flex items-center gap-2">
                  <input
                    type="number"
                    value={form.max_repo_size_mb}
                    onChange={(e) =>
                      set("max_repo_size_mb", parseInt(e.target.value) || 0)
                    }
                    min={10}
                    max={5000}
                    className={cn(inputCls, "w-24 font-mono text-xs")}
                  />
                  <span className="text-xs text-text-ghost">MB</span>
                </div>
              </FormRow>

              {/* Indexing Workers */}
              <FormRow label="Indexing Workers" icon={Server} last hint="Low: 1 (less CPU) · Med: 4 (balanced) · High: 8+ (faster indexing, CPU intensive)">
                <input
                  type="number"
                  value={form.indexing_workers}
                  onChange={(e) =>
                    set("indexing_workers", parseInt(e.target.value) || 0)
                  }
                  min={1}
                  max={32}
                  className={cn(inputCls, "w-20 font-mono text-xs")}
                />
              </FormRow>
            </div>
          </section>
        )}

        {/* ============================================================== */}
        {/*  Help                                                           */}
        {/* ============================================================== */}
        <section className="mb-8 animate-fade-in stagger-4">
          <h2 className="font-display font-700 text-base text-text-primary mb-4">
            Quick Start
          </h2>
          <div className="glass-card rounded-xl p-5 space-y-4">
            <p className="text-sm text-text-secondary leading-relaxed">
              Changes are saved to{" "}
              <code className="font-mono text-accent/80 bg-accent/5 px-1.5 py-0.5 rounded">
                .env
              </code>{" "}
              and applied immediately. No restart needed for most settings.
            </p>

            <div className="space-y-3">
              <HelpStep
                step="1"
                title="Start Ollama"
                code="ollama serve"
                description="Download from ollama.com if not installed."
              />
              <HelpStep
                step="2"
                title="Pull models"
                code="ollama pull qwen2.5-coder:7b && ollama pull nomic-embed-text"
                description="Chat model + embedding model for code search."
              />
              <HelpStep
                step="3"
                title="(Optional) Add cloud fallback"
                code='CLOUD_API_KEY=your-key\nCLOUD_API_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai\nCLOUD_MODEL=gemini-2.0-flash'
                description="Or configure above — the form saves directly to .env."
              />
            </div>
          </div>
        </section>
      </div>

      {/* ================================================================ */}
      {/*  Floating save bar                                                */}
      {/* ================================================================ */}
      {hasChanges && (
        <div className="fixed bottom-0 inset-x-0 z-50 border-t border-border-default bg-surface/95 backdrop-blur-sm animate-fade-in">
          <div className="max-w-3xl mx-auto px-6 py-3 flex items-center justify-between">
            <span className="text-sm text-text-secondary flex items-center gap-2">
              <CircleDot className="w-3.5 h-3.5 text-amber" />
              Unsaved changes
            </span>
            <div className="flex items-center gap-2">
              <button
                onClick={handleDiscard}
                disabled={saving}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-border-default text-text-secondary text-sm hover:bg-hover transition-colors disabled:opacity-50"
              >
                <RotateCcw className="w-3.5 h-3.5" />
                Discard
              </button>
              <button
                onClick={handleSave}
                disabled={saving}
                className="flex items-center gap-1.5 px-4 py-1.5 rounded-lg bg-accent text-white text-sm font-medium hover:bg-accent/90 transition-colors disabled:opacity-50"
              >
                {saving ? (
                  <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                ) : (
                  <Save className="w-3.5 h-3.5" />
                )}
                {saving ? "Saving..." : "Save Changes"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Shared styles
// ---------------------------------------------------------------------------

const inputCls =
  "bg-base border border-border-subtle rounded-lg px-3 py-1.5 text-sm text-text-primary focus:border-accent focus:outline-none focus:ring-1 focus:ring-accent/30 transition-colors";

const monoCls = cn(inputCls, "w-full font-mono text-xs");

const selectCls = cn(
  inputCls,
  "w-full appearance-none cursor-pointer pr-8 font-mono text-xs"
);

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function FormRow({
  label,
  icon: Icon,
  children,
  hint,
  last,
}: {
  label: string;
  icon: typeof Server;
  children: React.ReactNode;
  hint?: string;
  last?: boolean;
}) {
  return (
    <div
      className={cn(
        "flex items-center gap-3 px-4 py-3",
        !last && "border-b border-border-subtle"
      )}
    >
      <Icon className="w-3.5 h-3.5 text-text-ghost shrink-0" />
      <span className="text-xs text-text-muted w-32 shrink-0 flex items-center gap-1.5">
        {label}
        {hint && <InfoTip text={hint} />}
      </span>
      <div className="flex-1 min-w-0">{children}</div>
    </div>
  );
}

function InfoTip({ text }: { text: string }) {
  const [open, setOpen] = useState(false);
  const [pos, setPos] = useState({ top: 0, left: 0, above: false });
  const btnRef = useRef<HTMLButtonElement>(null);
  const popRef = useRef<HTMLDivElement>(null);

  // Close on click outside
  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      if (
        btnRef.current?.contains(e.target as Node) ||
        popRef.current?.contains(e.target as Node)
      )
        return;
      setOpen(false);
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [open]);

  // Reposition on scroll/resize so popover tracks the icon
  useEffect(() => {
    if (!open || !btnRef.current) return;
    const reposition = () => {
      if (!btnRef.current) return;
      const rect = btnRef.current.getBoundingClientRect();
      const above = window.innerHeight - rect.bottom < 120;
      setPos({
        left: rect.left,
        top: above ? rect.top - 10 : rect.bottom + 10,
        above,
      });
    };
    const scroller = btnRef.current.closest(".overflow-y-auto") || window;
    scroller.addEventListener("scroll", reposition, { passive: true });
    window.addEventListener("resize", reposition, { passive: true });
    return () => {
      scroller.removeEventListener("scroll", reposition);
      window.removeEventListener("resize", reposition);
    };
  }, [open]);

  const handleToggle = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (!open && btnRef.current) {
      const rect = btnRef.current.getBoundingClientRect();
      const above = window.innerHeight - rect.bottom < 120;
      setPos({
        left: rect.left,
        top: above ? rect.top - 10 : rect.bottom + 10,
        above,
      });
    }
    setOpen((v) => !v);
  };

  // Split on " — " to separate range values from description
  const sep = text.indexOf(" — ");
  const ranges = sep !== -1 ? text.slice(0, sep) : text;
  const description = sep !== -1 ? text.slice(sep + 3) : null;

  return (
    <>
      <button
        ref={btnRef}
        type="button"
        onClick={handleToggle}
        className={cn(
          "rounded-full transition-colors inline-flex",
          open
            ? "text-accent"
            : "text-text-ghost/40 hover:text-accent/70"
        )}
      >
        <Info className="w-3 h-3" />
      </button>
      {open &&
        createPortal(
          <div
            ref={popRef}
            style={{
              position: "fixed",
              top: pos.above ? undefined : pos.top,
              bottom: pos.above
                ? window.innerHeight - pos.top
                : undefined,
              left: pos.left,
            }}
            className="z-[9999] w-64 max-w-[calc(100vw-2rem)] rounded-xl border border-border-default bg-elevated shadow-xl animate-fade-in"
          >
            {/* Arrow */}
            <div
              className={cn(
                "absolute left-3 w-3 h-3 rotate-45 border-border-default bg-elevated",
                pos.above
                  ? "-bottom-1.5 border-r border-b"
                  : "-top-1.5 border-l border-t"
              )}
            />
            <div className="relative p-3 space-y-2">
              <div className="flex items-start justify-between gap-2">
                <p className="text-[11px] font-medium text-text-primary leading-snug">
                  {ranges}
                </p>
                <button
                  type="button"
                  onClick={() => setOpen(false)}
                  className="text-text-ghost hover:text-text-secondary transition-colors shrink-0 mt-px"
                >
                  <X className="w-3 h-3" />
                </button>
              </div>
              {description && (
                <p className="text-[10px] text-text-muted leading-relaxed">
                  {description}
                </p>
              )}
            </div>
          </div>,
          document.body
        )}
    </>
  );
}

function SecretInput({
  value,
  onChange,
  configured,
  cleared,
  onClear,
  placeholder,
}: {
  value: string;
  onChange: (v: string) => void;
  configured: boolean;
  cleared: boolean;
  onClear: () => void;
  placeholder: string;
}) {
  const [visible, setVisible] = useState(false);

  return (
    <div className="flex items-center gap-2">
      <div className="relative flex-1">
        <input
          type={visible ? "text" : "password"}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder={
            cleared
              ? "Will be cleared on save"
              : configured
                ? "Configured — enter new value to replace"
                : placeholder
          }
          className={cn(
            monoCls,
            "pr-8",
            cleared && "border-rose/30 bg-rose/5 placeholder:text-rose/50"
          )}
        />
        {value && (
          <button
            type="button"
            onClick={() => setVisible(!visible)}
            className="absolute right-2 top-1/2 -translate-y-1/2 text-text-ghost hover:text-text-secondary transition-colors"
          >
            {visible ? (
              <EyeOff className="w-3.5 h-3.5" />
            ) : (
              <Eye className="w-3.5 h-3.5" />
            )}
          </button>
        )}
      </div>
      {configured && !cleared && (
        <button
          type="button"
          onClick={onClear}
          className="flex items-center gap-1 px-2 py-1 rounded-md text-xs text-rose/70 hover:text-rose hover:bg-rose/5 transition-colors whitespace-nowrap"
        >
          <Trash2 className="w-3 h-3" />
          Clear
        </button>
      )}
      {configured && !cleared && !value && (
        <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-green/10 text-green/70 font-medium whitespace-nowrap">
          set
        </span>
      )}
    </div>
  );
}

function CapabilityCard({
  label,
  ready,
  requirement,
}: {
  label: string;
  ready: boolean;
  requirement: string;
}) {
  return (
    <div
      className={cn(
        "flex flex-col items-center gap-1.5 p-3 rounded-xl border",
        ready ? "border-green/15 bg-green/5" : "border-rose/15 bg-rose/5"
      )}
    >
      {ready ? (
        <CircleCheck className="w-5 h-5 text-green" />
      ) : (
        <CircleX className="w-5 h-5 text-rose/60" />
      )}
      <span className="text-sm font-medium text-text-primary">{label}</span>
      <span className="text-[10px] text-text-muted">
        {ready ? "Ready" : `Needs: ${requirement}`}
      </span>
    </div>
  );
}

function HelpStep({
  step,
  title,
  code,
  description,
}: {
  step: string;
  title: string;
  code: string;
  description: string;
}) {
  return (
    <div className="flex gap-3">
      <span className="flex items-center justify-center w-6 h-6 rounded-full bg-accent/10 text-accent text-xs font-mono font-700 shrink-0 mt-0.5">
        {step}
      </span>
      <div className="flex-1 min-w-0">
        <p className="text-sm font-medium text-text-primary">{title}</p>
        <p className="text-xs text-text-muted mt-0.5">{description}</p>
        <pre className="mt-1.5 text-xs font-mono bg-base border border-border-subtle rounded-lg px-3 py-2 text-accent/80 overflow-x-auto whitespace-pre-wrap">
          {code}
        </pre>
      </div>
    </div>
  );
}
