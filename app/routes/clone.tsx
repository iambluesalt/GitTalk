import { useState, useEffect, useRef, useCallback } from "react";
import { useNavigate, useSearchParams, Link } from "react-router";
import {
  GitBranch,
  ArrowRight,
  CheckCircle2,
  AlertCircle,
  Database,
  MessageSquare,
  RotateCcw,
  ExternalLink,
  Clock,
  FileCode2,
  Layers,
  Eye,
  ChevronDown,
  ChevronUp,
} from "lucide-react";
import type { Route } from "./+types/clone";
import type { Project, RepositoryAnalysis } from "~/lib/types";
import { cloneRepo, indexProject, getProject, parseSSEStream } from "~/lib/api";
import TerminalProgress from "~/components/TerminalProgress";
import ErrorCard from "~/components/ErrorCard";
import { humanizeError } from "~/lib/errors";
import { cn, formatNumber, getLanguageColor } from "~/lib/utils";

export function meta({}: Route.MetaArgs) {
  return [{ title: "Clone Repository — GitTalk" }];
}

interface LogLine {
  type: "info" | "success" | "warning" | "error" | "dim";
  text: string;
}

type Phase = "input" | "cloning" | "cloned" | "indexing" | "indexed" | "error";

interface IndexingStats {
  filesIndexed: number;
  chunksCreated: number;
  durationSeconds: number;
}

export default function Clone() {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();

  const [url, setUrl] = useState(searchParams.get("url") || "");
  const [phase, setPhase] = useState<Phase>("input");
  const [logs, setLogs] = useState<LogLine[]>([]);
  const [percent, setPercent] = useState(0);
  const [project, setProject] = useState<Project | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [analysisOpen, setAnalysisOpen] = useState(true);
  const [indexStats, setIndexStats] = useState<IndexingStats | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  // Elapsed time tracking
  const phaseStartRef = useRef<number>(0);
  const [elapsed, setElapsed] = useState(0);
  const [eta, setEta] = useState<number | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const startTimer = useCallback(() => {
    phaseStartRef.current = Date.now();
    setElapsed(0);
    setEta(null);
    timerRef.current = setInterval(() => {
      setElapsed(Math.round((Date.now() - phaseStartRef.current) / 1000));
    }, 1000);
  }, []);

  const stopTimer = useCallback(() => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  // Cleanup timer and abort streams on unmount
  useEffect(() => () => {
    stopTimer();
    abortRef.current?.abort();
  }, [stopTimer]);

  // Auto-start if URL provided or index param
  const indexId = searchParams.get("index");
  useEffect(() => {
    if (indexId) {
      startIndexing(indexId);
    } else if (searchParams.get("url") && url) {
      startClone();
    }
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const addLog = useCallback(
    (type: LogLine["type"], text: string) => {
      setLogs((prev) => [...prev, { type, text }]);
    },
    []
  );

  // Calculate ETA from progress percentage
  const updateEta = useCallback((pct: number) => {
    if (pct > 0 && pct < 100) {
      const elapsedMs = Date.now() - phaseStartRef.current;
      const totalEstMs = (elapsedMs / pct) * 100;
      const remainingMs = totalEstMs - elapsedMs;
      setEta(Math.max(1, Math.round(remainingMs / 1000)));
    } else {
      setEta(null);
    }
  }, []);

  const startClone = async () => {
    if (!url.trim()) return;
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setPhase("cloning");
    setLogs([]);
    setPercent(0);
    setError(null);
    setIndexStats(null);
    startTimer();
    addLog("info", `Cloning ${url}…`);

    try {
      const response = await cloneRepo(url.trim(), false, controller.signal);

      for await (const msg of parseSSEStream(response)) {
        switch (msg.event) {
          case "status":
            addLog("info", msg.data.message as string);
            break;
          case "progress": {
            const pct = msg.data.percent as number;
            setPercent(pct);
            updateEta(pct);
            if (msg.data.current_file) {
              addLog("dim", `  ${msg.data.current_file}`);
            }
            break;
          }
          case "duplicate":
            stopTimer();
            addLog("warning", msg.data.message as string);
            if (msg.data.project) {
              setProject(msg.data.project as Project);
              setPhase("cloned");
            }
            return;
          case "complete":
            stopTimer();
            addLog("success", msg.data.message as string);
            setPercent(100);
            if (msg.data.project) {
              setProject(msg.data.project as Project);
            }
            setPhase("cloned");
            return;
          case "error":
            stopTimer();
            addLog("error", msg.data.message as string);
            setError(msg.data.message as string);
            setPhase("error");
            return;
        }
      }
    } catch (err: unknown) {
      if (err instanceof DOMException && err.name === "AbortError") return;
      stopTimer();
      const message = err instanceof Error ? err.message : "Clone failed";
      addLog("error", message);
      setError(message);
      setPhase("error");
    }
  };

  const startIndexing = async (projectId?: string) => {
    const pid = projectId || project?.id;
    if (!pid) return;

    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setPhase("indexing");
    setPercent(0);
    setIndexStats(null);
    startTimer();
    if (!projectId) {
      addLog("info", "Starting indexing…");
    } else {
      setLogs([{ type: "info", text: "Starting indexing…" }]);
    }

    try {
      const response = await indexProject(pid, false, controller.signal);

      for await (const msg of parseSSEStream(response)) {
        switch (msg.event) {
          case "indexing_start":
            addLog("info", msg.data.message as string);
            break;
          case "status":
            addLog("info", msg.data.message as string);
            break;
          case "indexing_progress": {
            const pct = msg.data.percent as number;
            setPercent(pct);
            updateEta(pct);
            const current = msg.data.current_file as string;
            if (current) {
              addLog(
                "dim",
                `  [${msg.data.files_processed}/${msg.data.total_files}] ${current}`
              );
            }
            break;
          }
          case "indexing_complete": {
            stopTimer();
            const duration = msg.data.duration_seconds as number;
            const filesIndexed = msg.data.files_indexed as number;
            const chunksCreated = msg.data.chunks_created as number;
            setIndexStats({
              filesIndexed,
              chunksCreated,
              durationSeconds: duration,
            });
            addLog(
              "success",
              `Indexed ${filesIndexed} files (${chunksCreated} chunks) in ${formatDuration(duration)}`
            );
            setPercent(100);
            // Fetch updated project data to get latest status
            try {
              const updatedProject = await getProject(pid);
              setProject(updatedProject);
            } catch {
              // If fetch fails, keep existing project
            }
            setPhase("indexed");
            return;
          }
          case "error":
            stopTimer();
            addLog("error", msg.data.message as string);
            setError(msg.data.message as string);
            setPhase("error");
            return;
        }
      }
    } catch (err: unknown) {
      if (err instanceof DOMException && err.name === "AbortError") return;
      stopTimer();
      const message = err instanceof Error ? err.message : "Indexing failed";
      addLog("error", message);
      setError(message);
      setPhase("error");
    }
  };

  const reset = () => {
    stopTimer();
    setPhase("input");
    setLogs([]);
    setPercent(0);
    setProject(null);
    setError(null);
    setUrl("");
    setElapsed(0);
    setEta(null);
    setIndexStats(null);
  };

  const isWorking = phase === "cloning" || phase === "indexing";
  const isValidUrl = url.trim().includes("github.com/");

  return (
    <div className="h-full overflow-y-auto">
      <div className="max-w-2xl mx-auto px-6 py-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="font-display font-800 text-2xl text-text-primary">
            {indexId ? "Index Project" : "Clone Repository"}
          </h1>
          <p className="text-sm text-text-secondary mt-1">
            {indexId
              ? "Parse and embed code for AI-powered search"
              : "Enter a GitHub URL to clone and analyze a repository"}
          </p>
        </div>

        {/* URL Input */}
        {!indexId && (
          <div className="mb-6">
            <form
              onSubmit={(e) => {
                e.preventDefault();
                startClone();
              }}
            >
              <label className="block text-xs font-medium text-text-secondary mb-2">
                GitHub Repository URL
              </label>
              <div className="flex gap-2">
                <div className="relative flex-1">
                  <GitBranch className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-text-muted" />
                  <input
                    type="text"
                    value={url}
                    onChange={(e) => setUrl(e.target.value)}
                    placeholder="https://github.com/owner/repo"
                    disabled={isWorking}
                    className="w-full pl-10 pr-4 py-3 rounded-xl bg-surface border border-border-default text-text-primary placeholder:text-text-ghost text-sm font-mono focus:outline-none focus:border-accent/40 focus:shadow-[0_0_0_3px_rgba(0,212,255,0.1)] disabled:opacity-50 transition-all"
                  />
                </div>
                <button
                  type="submit"
                  disabled={!isValidUrl || isWorking}
                  className="flex items-center gap-2 px-5 py-3 rounded-xl bg-accent text-void font-display font-600 text-sm hover:bg-accent/90 disabled:opacity-40 disabled:cursor-not-allowed transition-all hover:shadow-lg hover:shadow-accent/20"
                >
                  {isWorking ? (
                    <>
                      <span className="w-4 h-4 border-2 border-void/30 border-t-void rounded-full animate-spin" />
                      Working…
                    </>
                  ) : (
                    <>
                      Clone
                      <ArrowRight className="w-4 h-4" />
                    </>
                  )}
                </button>
              </div>
            </form>
          </div>
        )}

        {/* Terminal */}
        {logs.length > 0 && (
          <div className="mb-6 animate-fade-in-up">
            <TerminalProgress
              lines={logs}
              title={
                phase === "cloning"
                  ? "git clone"
                  : phase === "indexing"
                  ? "indexing"
                  : phase === "indexed"
                  ? "complete"
                  : phase === "error"
                  ? "error"
                  : "clone"
              }
              percent={percent}
              isRunning={isWorking}
              elapsedSeconds={elapsed}
              etaSeconds={isWorking ? eta : null}
            />
          </div>
        )}

        {/* Success: Cloned */}
        {phase === "cloned" && project && (
          <div className="space-y-4 animate-fade-in-up">
            {/* Success banner */}
            <div className="glass-card rounded-xl p-5 border-green/20">
              <div className="flex items-center gap-3 mb-1">
                <div className="flex items-center justify-center w-10 h-10 rounded-lg bg-green/10">
                  <CheckCircle2 className="w-5 h-5 text-green" />
                </div>
                <div className="flex-1">
                  <h3 className="font-display font-600 text-text-primary">
                    Repository Cloned Successfully
                  </h3>
                  <p className="text-xs text-text-secondary">
                    {project.name} is ready — index it to enable AI chat
                  </p>
                </div>
                {elapsed > 0 && (
                  <span className="flex items-center gap-1.5 text-xs text-text-ghost font-mono">
                    <Clock className="w-3 h-3" />
                    {formatDuration(elapsed)}
                  </span>
                )}
              </div>
            </div>

            {/* Analysis details */}
            {project.analysis && (
              <div className="glass-card rounded-xl overflow-hidden">
                <button
                  onClick={() => setAnalysisOpen(!analysisOpen)}
                  className="w-full flex items-center justify-between px-5 py-3 hover:bg-hover/30 transition-colors"
                >
                  <span className="flex items-center gap-2 text-sm font-display font-600 text-text-primary">
                    <Eye className="w-4 h-4 text-accent" />
                    Repository Analysis
                  </span>
                  {analysisOpen ? (
                    <ChevronUp className="w-4 h-4 text-text-muted" />
                  ) : (
                    <ChevronDown className="w-4 h-4 text-text-muted" />
                  )}
                </button>

                {analysisOpen && (
                  <div className="px-5 pb-5 space-y-4 border-t border-border-subtle pt-4">
                    {/* Stats row */}
                    <div className="grid grid-cols-3 gap-3">
                      <div className="flex flex-col items-center py-3 rounded-lg bg-base border border-border-subtle">
                        <span className="text-lg font-mono font-700 text-text-primary">
                          {formatNumber(project.analysis.total_files)}
                        </span>
                        <span className="text-xs text-text-muted">Files</span>
                      </div>
                      <div className="flex flex-col items-center py-3 rounded-lg bg-base border border-border-subtle">
                        <span className="text-lg font-mono font-700 text-text-primary">
                          {formatNumber(project.analysis.total_lines)}
                        </span>
                        <span className="text-xs text-text-muted">Lines</span>
                      </div>
                      <div className="flex flex-col items-center py-3 rounded-lg bg-base border border-border-subtle">
                        <span className="text-lg font-mono font-700 text-text-primary">
                          {project.analysis.repository_size_mb
                            ? `${project.analysis.repository_size_mb.toFixed(1)}MB`
                            : "—"}
                        </span>
                        <span className="text-xs text-text-muted">Size</span>
                      </div>
                    </div>

                    {/* Language breakdown */}
                    <LanguageBreakdown analysis={project.analysis} />
                  </div>
                )}
              </div>
            )}

            {/* Action buttons */}
            <div className="flex gap-2">
              <button
                onClick={() => startIndexing()}
                className="flex items-center gap-2 px-5 py-2.5 rounded-lg bg-accent text-void font-display font-600 text-sm hover:bg-accent/90 transition-all flex-1 justify-center hover:shadow-lg hover:shadow-accent/20"
              >
                <Database className="w-4 h-4" />
                Index Now
              </button>
              <Link
                to="/projects"
                className="flex items-center gap-2 px-4 py-2.5 rounded-lg bg-elevated text-text-secondary font-medium text-sm hover:bg-hover transition-colors"
              >
                Later
              </Link>
            </div>
          </div>
        )}

        {/* Success: Indexed */}
        {phase === "indexed" && (
          <div className="space-y-4 animate-fade-in-up">
            {/* Success banner */}
            <div className="glass-card rounded-xl p-5 border-green/20">
              <div className="flex items-center gap-3">
                <div className="flex items-center justify-center w-10 h-10 rounded-lg bg-green/10">
                  <CheckCircle2 className="w-5 h-5 text-green" />
                </div>
                <div className="flex-1">
                  <h3 className="font-display font-600 text-text-primary">
                    Indexing Complete
                  </h3>
                  <p className="text-xs text-text-secondary">
                    {project?.name
                      ? `${project.name} is ready for AI-powered chat`
                      : "Ready to chat with your codebase"}
                  </p>
                </div>
              </div>

              {/* Indexing stats */}
              {indexStats && (
                <div className="grid grid-cols-3 gap-3 mt-4">
                  <div className="flex flex-col items-center py-2.5 rounded-lg bg-base border border-border-subtle">
                    <span className="flex items-center gap-1.5 text-sm font-mono font-700 text-text-primary">
                      <FileCode2 className="w-3.5 h-3.5 text-accent" />
                      {indexStats.filesIndexed}
                    </span>
                    <span className="text-[11px] text-text-muted">Files Indexed</span>
                  </div>
                  <div className="flex flex-col items-center py-2.5 rounded-lg bg-base border border-border-subtle">
                    <span className="flex items-center gap-1.5 text-sm font-mono font-700 text-text-primary">
                      <Layers className="w-3.5 h-3.5 text-purple" />
                      {indexStats.chunksCreated}
                    </span>
                    <span className="text-[11px] text-text-muted">Chunks Created</span>
                  </div>
                  <div className="flex flex-col items-center py-2.5 rounded-lg bg-base border border-border-subtle">
                    <span className="flex items-center gap-1.5 text-sm font-mono font-700 text-text-primary">
                      <Clock className="w-3.5 h-3.5 text-green" />
                      {formatDuration(indexStats.durationSeconds)}
                    </span>
                    <span className="text-[11px] text-text-muted">Duration</span>
                  </div>
                </div>
              )}
            </div>

            {/* Action buttons */}
            <div className="flex gap-2">
              {project && (
                <Link
                  to={`/chat/${project.id}`}
                  className="flex items-center gap-2 px-5 py-2.5 rounded-lg bg-accent text-void font-display font-600 text-sm hover:bg-accent/90 transition-all flex-1 justify-center hover:shadow-lg hover:shadow-accent/20"
                >
                  <MessageSquare className="w-4 h-4" />
                  Start Chatting
                </Link>
              )}
              <button
                onClick={reset}
                className="flex items-center gap-2 px-4 py-2.5 rounded-lg bg-elevated text-text-secondary font-medium text-sm hover:bg-hover transition-colors"
              >
                <RotateCcw className="w-4 h-4" />
                Clone Another
              </button>
            </div>
          </div>
        )}

        {/* Error state */}
        {phase === "error" && error && (
          <div className="space-y-3 animate-fade-in-up">
            <ErrorCard error={humanizeError(error)} />
            <button
              onClick={reset}
              className="flex items-center gap-2 px-4 py-2.5 rounded-lg bg-elevated text-text-secondary font-medium text-sm hover:bg-hover transition-colors"
            >
              <RotateCcw className="w-4 h-4" />
              Try Again
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

/** Language breakdown bar + labels */
function LanguageBreakdown({ analysis }: { analysis: RepositoryAnalysis }) {
  const langs = Object.entries(analysis.languages)
    .map(([name, stats]) => ({
      name,
      lines: stats.lines_of_code,
      files: stats.file_count,
    }))
    .sort((a, b) => b.lines - a.lines);

  if (langs.length === 0) return null;

  const totalLines = langs.reduce((s, l) => s + l.lines, 0);

  return (
    <div>
      <span className="text-xs font-medium text-text-secondary mb-2 block">
        Languages
      </span>
      {/* Stacked bar */}
      <div className="flex h-2 rounded-full overflow-hidden mb-2.5">
        {langs.slice(0, 8).map((lang) => (
          <div
            key={lang.name}
            style={{
              width: `${Math.max((lang.lines / totalLines) * 100, 1)}%`,
              backgroundColor: getLanguageColor(lang.name),
            }}
            title={`${lang.name}: ${((lang.lines / totalLines) * 100).toFixed(1)}%`}
          />
        ))}
      </div>
      {/* Labels */}
      <div className="flex flex-wrap gap-x-4 gap-y-1">
        {langs.slice(0, 6).map((lang) => (
          <div key={lang.name} className="flex items-center gap-1.5">
            <span
              className="w-2 h-2 rounded-full shrink-0"
              style={{ backgroundColor: getLanguageColor(lang.name) }}
            />
            <span className="text-[11px] text-text-muted">
              {lang.name}
            </span>
            <span className="text-[11px] text-text-ghost font-mono">
              {((lang.lines / totalLines) * 100).toFixed(0)}%
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

function formatDuration(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return s > 0 ? `${m}m ${s}s` : `${m}m`;
}
