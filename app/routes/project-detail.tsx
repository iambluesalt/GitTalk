import { useState, useEffect, useMemo, useCallback, useRef } from "react";
import { useParams, useNavigate, Link } from "react-router";
import {
  ArrowLeft,
  GitBranch,
  ExternalLink,
  MessageSquare,
  Database,
  FileCode2,
  Clock,
  Layers,
  Braces,
  Box,
  Import,
  ChevronDown,
  ChevronRight,
  Trash2,
  Info,
  FolderOpen,
  Hash,
  Code2,
  AlertCircle,
  CheckCircle2,
} from "lucide-react";
import type { Route } from "./+types/project-detail";
import type { Project, RepositoryAnalysis, IndexStats } from "~/lib/types";
import {
  getProject,
  getProjectFiles,
  getIndexStats,
  deleteProject,
  indexProject,
  parseSSEStream,
  type ProjectFile,
} from "~/lib/api";
import { cn, formatNumber, getLanguageColor, statusLabel, timeAgo } from "~/lib/utils";
import TerminalProgress from "~/components/TerminalProgress";

export function meta({}: Route.MetaArgs) {
  return [{ title: "Project Details — GitTalk" }];
}

interface LogLine {
  type: "info" | "success" | "warning" | "error" | "dim";
  text: string;
}

export default function ProjectDetail() {
  const { projectId } = useParams();
  const navigate = useNavigate();
  const [project, setProject] = useState<Project | null>(null);
  const [files, setFiles] = useState<ProjectFile[]>([]);
  const [indexStats, setIndexStats] = useState<IndexStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [filesLoading, setFilesLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<"overview" | "files">("overview");
  const [fileSearch, setFileSearch] = useState("");
  const [expandedDirs, setExpandedDirs] = useState<Set<string>>(new Set());
  const [deleting, setDeleting] = useState(false);

  // Inline indexing state
  const [indexing, setIndexing] = useState(false);
  const [indexLogs, setIndexLogs] = useState<LogLine[]>([]);
  const [indexPercent, setIndexPercent] = useState(0);
  const [indexElapsed, setIndexElapsed] = useState(0);
  const [indexEta, setIndexEta] = useState<number | null>(null);
  const [indexComplete, setIndexComplete] = useState(false);
  const [indexError, setIndexError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const phaseStartRef = useRef<number>(0);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const startTimer = useCallback(() => {
    phaseStartRef.current = Date.now();
    setIndexElapsed(0);
    setIndexEta(null);
    timerRef.current = setInterval(() => {
      setIndexElapsed(Math.round((Date.now() - phaseStartRef.current) / 1000));
    }, 1000);
  }, []);

  const stopTimer = useCallback(() => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  useEffect(() => () => {
    stopTimer();
    abortRef.current?.abort();
  }, [stopTimer]);

  const updateEta = useCallback((pct: number) => {
    if (pct > 0 && pct < 100) {
      const elapsedMs = Date.now() - phaseStartRef.current;
      const totalEstMs = (elapsedMs / pct) * 100;
      const remainingMs = totalEstMs - elapsedMs;
      setIndexEta(Math.max(1, Math.round(remainingMs / 1000)));
    } else {
      setIndexEta(null);
    }
  }, []);

  // Load project
  useEffect(() => {
    if (!projectId) return;
    setLoading(true);
    getProject(projectId)
      .then((p) => {
        setProject(p);
        document.title = `${p.name} — GitTalk`;
      })
      .catch((err) => setError(err instanceof Error ? err.message : "Failed to load project"))
      .finally(() => setLoading(false));
  }, [projectId]);

  // Load index stats when project is indexed
  useEffect(() => {
    if (projectId && project?.status === "indexed") {
      getIndexStats(projectId)
        .then(setIndexStats)
        .catch(() => {});
    }
  }, [projectId, project?.status]);

  // Load files when switching to files tab (and project is indexed)
  useEffect(() => {
    if (activeTab === "files" && projectId && files.length === 0 && project?.status === "indexed") {
      setFilesLoading(true);
      getProjectFiles(projectId)
        .then((data) => setFiles(data.files))
        .catch(() => {})
        .finally(() => setFilesLoading(false));
    }
  }, [activeTab, projectId, project?.status]); // eslint-disable-line react-hooks/exhaustive-deps

  const handleDelete = async () => {
    if (!projectId || !project) return;
    setDeleting(true);
    try {
      await deleteProject(projectId);
      navigate("/projects");
    } catch {
      setDeleting(false);
    }
  };

  const handleIndex = async () => {
    if (!projectId) return;
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setIndexing(true);
    setIndexLogs([]);
    setIndexPercent(0);
    setIndexComplete(false);
    setIndexError(null);
    startTimer();

    const addLog = (type: LogLine["type"], text: string) => {
      setIndexLogs((prev) => [...prev, { type, text }]);
    };

    addLog("info", "Starting indexing…");

    try {
      const response = await indexProject(projectId, false, controller.signal);

      for await (const msg of parseSSEStream(response)) {
        switch (msg.event) {
          case "indexing_start":
          case "status":
            addLog("info", msg.data.message as string);
            break;
          case "indexing_progress": {
            const pct = msg.data.percent as number;
            setIndexPercent(pct);
            updateEta(pct);
            const current = msg.data.current_file as string;
            if (current) {
              addLog("dim", `  [${msg.data.files_processed}/${msg.data.total_files}] ${current}`);
            }
            break;
          }
          case "indexing_complete": {
            stopTimer();
            const duration = msg.data.duration_seconds as number;
            const filesIndexed = msg.data.files_indexed as number;
            const chunksCreated = msg.data.chunks_created as number;
            addLog(
              "success",
              `Indexed ${filesIndexed} files (${chunksCreated} chunks) in ${formatDuration(duration)}`
            );
            setIndexPercent(100);
            setIndexComplete(true);
            setIndexing(false);
            // Refresh project data + stats
            try {
              const updatedProject = await getProject(projectId);
              setProject(updatedProject);
              const stats = await getIndexStats(projectId);
              setIndexStats(stats);
              // Reset files so they reload on tab switch
              setFiles([]);
            } catch {}
            return;
          }
          case "error":
            stopTimer();
            addLog("error", msg.data.message as string);
            setIndexError(msg.data.message as string);
            setIndexing(false);
            return;
        }
      }
    } catch (err: unknown) {
      if (err instanceof DOMException && err.name === "AbortError") return;
      stopTimer();
      const message = err instanceof Error ? err.message : "Indexing failed";
      setIndexLogs((prev) => [...prev, { type: "error", text: message }]);
      setIndexError(message);
      setIndexing(false);
    }
  };

  if (loading) {
    return (
      <div className="h-full overflow-y-auto">
        <div className="max-w-5xl mx-auto px-6 py-8">
          <div className="h-8 w-48 rounded-lg bg-surface animate-shimmer bg-gradient-to-r from-surface via-elevated to-surface mb-6" />
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            {[1, 2, 3].map((i) => (
              <div key={i} className="h-32 rounded-xl bg-surface animate-shimmer bg-gradient-to-r from-surface via-elevated to-surface" />
            ))}
          </div>
        </div>
      </div>
    );
  }

  if (error || !project) {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="text-center">
          <AlertCircle className="w-12 h-12 text-rose mx-auto mb-4" />
          <p className="text-text-secondary mb-4">{error || "Project not found"}</p>
          <Link to="/projects" className="text-accent hover:underline text-sm">
            Back to Projects
          </Link>
        </div>
      </div>
    );
  }

  const analysis = project.analysis;
  const isIndexed = project.status === "indexed";
  const isError = project.status === "error";

  return (
    <div className="h-full overflow-y-auto">
      <div className="max-w-5xl mx-auto px-6 py-8">
        {/* Back + Header */}
        <div className="mb-6 animate-fade-in">
          <Link
            to="/projects"
            className="inline-flex items-center gap-1.5 text-xs text-text-muted hover:text-text-secondary transition-colors mb-4"
          >
            <ArrowLeft className="w-3.5 h-3.5" />
            All Projects
          </Link>

          <div className="flex items-start gap-4">
            {/* Icon */}
            <div
              className={cn(
                "flex items-center justify-center w-12 h-12 rounded-xl shrink-0",
                isIndexed && "bg-green/10",
                isError && "bg-rose/10",
                project.status === "cloned" && "bg-accent/10",
                (project.status === "cloning" || project.status === "indexing") && "bg-amber/10"
              )}
            >
              <GitBranch
                className={cn(
                  "w-6 h-6",
                  isIndexed && "text-green",
                  isError && "text-rose",
                  project.status === "cloned" && "text-accent",
                  (project.status === "cloning" || project.status === "indexing") && "text-amber"
                )}
              />
            </div>

            {/* Title area */}
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-3 flex-wrap">
                <h1 className="font-display font-800 text-2xl text-text-primary">
                  {project.name}
                </h1>
                <span
                  className={cn(
                    "inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-medium",
                    isIndexed && "bg-green/10 text-green",
                    isError && "bg-rose/10 text-rose",
                    project.status === "cloned" && "bg-accent/10 text-accent",
                    (project.status === "cloning" || project.status === "indexing") && "bg-amber/10 text-amber"
                  )}
                >
                  {(project.status === "cloning" || project.status === "indexing") && (
                    <span className="w-1.5 h-1.5 rounded-full bg-current animate-pulse" />
                  )}
                  {statusLabel(project.status)}
                </span>
              </div>
              <div className="flex items-center gap-4 mt-1">
                <a
                  href={project.github_url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-1.5 text-sm text-text-muted hover:text-accent transition-colors"
                >
                  {project.github_url.replace("https://github.com/", "")}
                  <ExternalLink className="w-3.5 h-3.5" />
                </a>
                {analysis?.primary_language && (
                  <span className="inline-flex items-center gap-1.5 text-sm text-text-muted">
                    <span
                      className="w-2.5 h-2.5 rounded-full"
                      style={{ backgroundColor: getLanguageColor(analysis.primary_language) }}
                    />
                    {analysis.primary_language}
                  </span>
                )}
              </div>
            </div>

            {/* Action buttons */}
            <div className="flex items-center gap-2 shrink-0">
              {isIndexed && (
                <Link
                  to={`/chat/${project.id}`}
                  className="flex items-center gap-2 px-4 py-2 rounded-lg bg-accent text-void font-display font-600 text-sm hover:bg-accent/90 transition-all hover:shadow-lg hover:shadow-accent/20"
                >
                  <MessageSquare className="w-4 h-4" />
                  Chat
                </Link>
              )}
              {(project.status === "cloned" || isIndexed) && !indexing && (
                <button
                  onClick={handleIndex}
                  className="flex items-center gap-2 px-4 py-2 rounded-lg bg-elevated text-text-secondary font-medium text-sm hover:bg-hover hover:text-text-primary transition-colors"
                >
                  <Database className="w-4 h-4" />
                  {isIndexed ? "Re-index" : "Index"}
                </button>
              )}
              <button
                onClick={handleDelete}
                disabled={deleting || indexing}
                className="flex items-center gap-2 px-3 py-2 rounded-lg text-text-ghost hover:text-rose hover:bg-rose/10 transition-colors disabled:opacity-50"
                title="Delete project"
              >
                <Trash2 className="w-4 h-4" />
              </button>
            </div>
          </div>
        </div>

        {/* Shallow clone indicator */}
        <div className="mb-6 flex items-center gap-2 px-3 py-2 rounded-lg bg-surface border border-border-subtle animate-fade-in">
          <Info className="w-3.5 h-3.5 text-accent shrink-0" />
          <span className="text-[11px] text-text-muted">
            <span className="text-accent font-medium">Shallow clone</span>
            {" "}&mdash; latest snapshot only, no git history
          </span>
          <span className="flex-1" />
          <span className="text-[11px] text-text-ghost font-mono">
            Cloned {timeAgo(project.cloned_at)}
          </span>
          {project.last_indexed && (
            <span className="text-[11px] text-text-ghost font-mono">
              Indexed {timeAgo(project.last_indexed)}
            </span>
          )}
        </div>

        {/* Error banner */}
        {isError && project.error_message && (
          <div className="mb-6 p-4 rounded-xl border border-rose/20 bg-rose/5 animate-fade-in">
            <div className="flex items-center gap-2 mb-1">
              <AlertCircle className="w-4 h-4 text-rose" />
              <span className="text-sm font-medium text-rose">Error</span>
            </div>
            <p className="text-xs text-text-secondary">{project.error_message}</p>
          </div>
        )}

        {/* Inline Indexing Progress */}
        {(indexing || indexLogs.length > 0) && (
          <div className="mb-6 animate-fade-in-up space-y-4">
            <TerminalProgress
              lines={indexLogs}
              title="indexing"
              percent={indexPercent}
              isRunning={indexing}
              elapsedSeconds={indexElapsed}
              etaSeconds={indexing ? indexEta : null}
            />
            {indexComplete && (
              <div className="glass-card rounded-xl p-4 border-green/20">
                <div className="flex items-center gap-3">
                  <CheckCircle2 className="w-5 h-5 text-green" />
                  <div className="flex-1">
                    <span className="text-sm font-display font-600 text-text-primary">
                      Indexing Complete
                    </span>
                    <p className="text-xs text-text-secondary">
                      Project is now ready for AI-powered chat
                    </p>
                  </div>
                  <Link
                    to={`/chat/${project.id}`}
                    className="flex items-center gap-2 px-4 py-2 rounded-lg bg-accent text-void font-display font-600 text-sm hover:bg-accent/90 transition-all"
                  >
                    <MessageSquare className="w-4 h-4" />
                    Chat
                  </Link>
                </div>
              </div>
            )}
            {indexError && (
              <div className="p-3 rounded-lg border border-rose/20 bg-rose/5 text-sm text-rose">
                {indexError}
              </div>
            )}
          </div>
        )}

        {analysis && (
          <>
            {/* Stats grid */}
            <div className={cn(
              "grid gap-3 mb-6 animate-fade-in-up",
              indexStats ? "grid-cols-2 md:grid-cols-4 lg:grid-cols-6" : "grid-cols-2 md:grid-cols-4"
            )}>
              <StatCard
                label="Files"
                value={formatNumber(analysis.total_files)}
                icon={FileCode2}
                color="accent"
                delay={0}
              />
              <StatCard
                label="Lines of Code"
                value={formatNumber(analysis.total_lines)}
                icon={Hash}
                color="green"
                delay={1}
              />
              <StatCard
                label="Size"
                value={`${analysis.repository_size_mb.toFixed(1)}MB`}
                icon={Database}
                color="purple"
                delay={2}
              />
              <StatCard
                label="Languages"
                value={Object.keys(analysis.languages).length.toString()}
                icon={Code2}
                color="amber"
                delay={3}
              />
              {indexStats && indexStats.files_indexed > 0 && (
                <>
                  <StatCard
                    label="Indexed Files"
                    value={formatNumber(indexStats.files_indexed)}
                    icon={Layers}
                    color="accent"
                    delay={4}
                  />
                  <StatCard
                    label="Chunks"
                    value={formatNumber(indexStats.chunks_created)}
                    icon={Braces}
                    color="purple"
                    delay={5}
                  />
                </>
              )}
            </div>

            {/* Tabs */}
            <div className="flex gap-1 p-1 rounded-lg bg-surface border border-border-subtle mb-6 w-fit animate-fade-in">
              {(["overview", "files"] as const).map((tab) => (
                <button
                  key={tab}
                  onClick={() => setActiveTab(tab)}
                  className={cn(
                    "px-4 py-1.5 rounded-md text-xs font-medium transition-colors",
                    activeTab === tab
                      ? "bg-elevated text-text-primary shadow-sm"
                      : "text-text-muted hover:text-text-secondary"
                  )}
                >
                  {tab === "overview" ? "Overview" : "Files"}
                  {tab === "files" && files.length > 0 && (
                    <span className="ml-1.5 text-[10px] px-1.5 py-0.5 rounded-full bg-accent/15 text-accent">
                      {files.length}
                    </span>
                  )}
                </button>
              ))}
            </div>

            {/* Tab Content */}
            {activeTab === "overview" ? (
              <OverviewTab analysis={analysis} indexStats={indexStats} />
            ) : (
              <FilesTab
                files={files}
                loading={filesLoading}
                isIndexed={isIndexed}
                projectId={project.id}
                search={fileSearch}
                onSearchChange={setFileSearch}
                expandedDirs={expandedDirs}
                onToggleDir={(dir) => {
                  setExpandedDirs((prev) => {
                    const next = new Set(prev);
                    next.has(dir) ? next.delete(dir) : next.add(dir);
                    return next;
                  });
                }}
                onIndex={handleIndex}
                indexing={indexing}
              />
            )}
          </>
        )}

        {/* No analysis yet */}
        {!analysis && !isError && (
          <div className="text-center py-16 animate-fade-in">
            <Database className="w-12 h-12 text-text-ghost mx-auto mb-4" />
            <p className="text-text-secondary mb-2">No analysis data yet</p>
            <p className="text-xs text-text-muted mb-4">
              This project needs to be cloned and analyzed first
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

/* ─── Stat Card ─── */

function StatCard({
  label,
  value,
  icon: Icon,
  color,
  delay,
}: {
  label: string;
  value: string;
  icon: typeof FileCode2;
  color: "accent" | "green" | "purple" | "amber";
  delay: number;
}) {
  const colors = {
    accent: { bg: "bg-accent/10", icon: "text-accent" },
    green: { bg: "bg-green/10", icon: "text-green" },
    purple: { bg: "bg-purple/10", icon: "text-purple" },
    amber: { bg: "bg-amber/10", icon: "text-amber" },
  };
  const c = colors[color];

  return (
    <div className={cn("stat-card glass-card rounded-xl p-4", `stagger-${Math.min(delay + 1, 6)}`)}>
      <div className="flex items-center gap-2 mb-2">
        <div className={cn("flex items-center justify-center w-7 h-7 rounded-lg", c.bg)}>
          <Icon className={cn("w-3.5 h-3.5", c.icon)} />
        </div>
        <span className="text-xs text-text-muted">{label}</span>
      </div>
      <span className="text-xl font-mono font-700 text-text-primary">{value}</span>
    </div>
  );
}

/* ─── Overview Tab ─── */

function OverviewTab({ analysis, indexStats }: { analysis: RepositoryAnalysis; indexStats: IndexStats | null }) {
  const languages = Object.entries(analysis.languages)
    .map(([name, stats]) => ({ name, ...stats }))
    .sort((a, b) => b.lines_of_code - a.lines_of_code);

  const totalLOC = languages.reduce((s, l) => s + l.lines_of_code, 0) || 1;
  const totalFunctions = languages.reduce((s, l) => s + l.functions, 0);
  const totalClasses = languages.reduce((s, l) => s + l.classes, 0);
  const totalImports = languages.reduce((s, l) => s + l.imports, 0);

  const fileTypes = Object.entries(analysis.file_types)
    .sort(([, a], [, b]) => b - a);

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 animate-fade-in-up">
      {/* Language breakdown */}
      <div className="glass-card rounded-xl p-5">
        <h3 className="font-display font-600 text-sm text-text-primary mb-4">
          Language Breakdown
        </h3>

        {/* Stacked bar */}
        <div className="flex h-3 rounded-full overflow-hidden mb-4">
          {languages.slice(0, 8).map((lang) => (
            <div
              key={lang.name}
              className="h-full transition-all duration-500"
              style={{
                width: `${Math.max((lang.lines_of_code / totalLOC) * 100, 1.5)}%`,
                backgroundColor: getLanguageColor(lang.name),
              }}
              title={`${lang.name}: ${((lang.lines_of_code / totalLOC) * 100).toFixed(1)}%`}
            />
          ))}
        </div>

        {/* Language list */}
        <div className="space-y-2.5">
          {languages.slice(0, 10).map((lang) => {
            const pct = (lang.lines_of_code / totalLOC) * 100;
            return (
              <div key={lang.name} className="flex items-center gap-3">
                <span
                  className="w-3 h-3 rounded-full shrink-0"
                  style={{ backgroundColor: getLanguageColor(lang.name) }}
                />
                <span className="text-sm text-text-primary flex-1 min-w-0 truncate">
                  {lang.name}
                </span>
                <span className="text-xs text-text-muted font-mono tabular-nums">
                  {formatNumber(lang.lines_of_code)} lines
                </span>
                <span className="text-xs text-text-ghost font-mono w-12 text-right tabular-nums">
                  {pct.toFixed(1)}%
                </span>
              </div>
            );
          })}
          {languages.length > 10 && (
            <span className="text-xs text-text-ghost">
              +{languages.length - 10} more
            </span>
          )}
        </div>
      </div>

      {/* Code structure summary */}
      <div className="glass-card rounded-xl p-5">
        <h3 className="font-display font-600 text-sm text-text-primary mb-4">
          Code Structure
        </h3>

        <div className="grid grid-cols-3 gap-3 mb-5">
          <div className="flex flex-col items-center py-3 rounded-lg bg-base border border-border-subtle">
            <Braces className="w-4 h-4 text-accent mb-1" />
            <span className="text-lg font-mono font-700 text-text-primary">
              {formatNumber(totalFunctions)}
            </span>
            <span className="text-[11px] text-text-muted">Functions</span>
          </div>
          <div className="flex flex-col items-center py-3 rounded-lg bg-base border border-border-subtle">
            <Box className="w-4 h-4 text-purple mb-1" />
            <span className="text-lg font-mono font-700 text-text-primary">
              {formatNumber(totalClasses)}
            </span>
            <span className="text-[11px] text-text-muted">Classes</span>
          </div>
          <div className="flex flex-col items-center py-3 rounded-lg bg-base border border-border-subtle">
            <Import className="w-4 h-4 text-green mb-1" />
            <span className="text-lg font-mono font-700 text-text-primary">
              {formatNumber(totalImports)}
            </span>
            <span className="text-[11px] text-text-muted">Imports</span>
          </div>
        </div>

        {/* Per-language structure */}
        <div className="space-y-2">
          <span className="text-xs text-text-muted font-medium block mb-1">Per Language</span>
          {languages.slice(0, 6).map((lang) => (
            <div
              key={lang.name}
              className="flex items-center gap-3 py-1.5 px-2 rounded-md hover:bg-hover/30 transition-colors"
            >
              <span
                className="w-2 h-2 rounded-full shrink-0"
                style={{ backgroundColor: getLanguageColor(lang.name) }}
              />
              <span className="text-xs text-text-secondary flex-1 truncate">{lang.name}</span>
              <span className="text-[11px] text-text-ghost font-mono tabular-nums">
                {lang.functions}f
              </span>
              <span className="text-[11px] text-text-ghost font-mono tabular-nums">
                {lang.classes}c
              </span>
              <span className="text-[11px] text-text-ghost font-mono tabular-nums">
                {lang.file_count} files
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Index summary card */}
      {indexStats && indexStats.files_indexed > 0 && (
        <div className="glass-card rounded-xl p-5">
          <h3 className="font-display font-600 text-sm text-text-primary mb-4">
            Index Summary
          </h3>
          <div className="grid grid-cols-2 gap-3">
            <div className="flex items-center gap-3 py-2 px-3 rounded-lg bg-base border border-border-subtle">
              <Layers className="w-4 h-4 text-accent shrink-0" />
              <div>
                <span className="text-sm font-mono font-700 text-text-primary block">
                  {formatNumber(indexStats.files_indexed)}
                </span>
                <span className="text-[11px] text-text-muted">Files Indexed</span>
              </div>
            </div>
            <div className="flex items-center gap-3 py-2 px-3 rounded-lg bg-base border border-border-subtle">
              <Braces className="w-4 h-4 text-purple shrink-0" />
              <div>
                <span className="text-sm font-mono font-700 text-text-primary block">
                  {formatNumber(indexStats.chunks_created)}
                </span>
                <span className="text-[11px] text-text-muted">Vector Chunks</span>
              </div>
            </div>
            <div className="flex items-center gap-3 py-2 px-3 rounded-lg bg-base border border-border-subtle">
              <Code2 className="w-4 h-4 text-green shrink-0" />
              <div>
                <span className="text-sm font-mono font-700 text-text-primary block">
                  {formatNumber(indexStats.total_functions)}
                </span>
                <span className="text-[11px] text-text-muted">Functions Parsed</span>
              </div>
            </div>
            <div className="flex items-center gap-3 py-2 px-3 rounded-lg bg-base border border-border-subtle">
              <Box className="w-4 h-4 text-amber shrink-0" />
              <div>
                <span className="text-sm font-mono font-700 text-text-primary block">
                  {formatNumber(indexStats.total_classes)}
                </span>
                <span className="text-[11px] text-text-muted">Classes Parsed</span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* File type distribution */}
      {fileTypes.length > 0 && (
        <div className={cn("glass-card rounded-xl p-5", !(indexStats && indexStats.files_indexed > 0) && "lg:col-span-2")}>
          <h3 className="font-display font-600 text-sm text-text-primary mb-4">
            File Types
          </h3>
          <div className="flex flex-wrap gap-2">
            {fileTypes.slice(0, 24).map(([ext, count]) => (
              <span
                key={ext}
                className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-base border border-border-subtle text-xs"
              >
                <span className="text-text-muted font-mono">{ext}</span>
                <span className="text-text-ghost font-mono">{count}</span>
              </span>
            ))}
            {fileTypes.length > 24 && (
              <span className="inline-flex items-center px-2.5 py-1 text-xs text-text-ghost">
                +{fileTypes.length - 24} more
              </span>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

/* ─── Files Tab ─── */

interface FilesTabProps {
  files: ProjectFile[];
  loading: boolean;
  isIndexed: boolean;
  projectId: string;
  search: string;
  onSearchChange: (v: string) => void;
  expandedDirs: Set<string>;
  onToggleDir: (dir: string) => void;
  onIndex: () => void;
  indexing: boolean;
}

function FilesTab({
  files,
  loading,
  isIndexed,
  projectId,
  search,
  onSearchChange,
  expandedDirs,
  onToggleDir,
  onIndex,
  indexing,
}: FilesTabProps) {
  // Build directory tree from flat file list
  const tree = useMemo(() => {
    let filtered = files;
    if (search.trim()) {
      const q = search.toLowerCase();
      filtered = files.filter(
        (f) =>
          f.file_path.toLowerCase().includes(q) ||
          f.language.toLowerCase().includes(q)
      );
    }
    return buildTree(filtered);
  }, [files, search]);

  if (!isIndexed) {
    return (
      <div className="text-center py-12 animate-fade-in">
        <Layers className="w-10 h-10 text-text-ghost mx-auto mb-3" />
        <p className="text-text-secondary text-sm mb-1">Project not indexed yet</p>
        <p className="text-xs text-text-muted mb-4">
          Index this project to see detailed file analysis
        </p>
        <button
          onClick={onIndex}
          disabled={indexing}
          className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-accent text-void font-display font-600 text-sm hover:bg-accent/90 transition-all disabled:opacity-50"
        >
          <Database className="w-4 h-4" />
          {indexing ? "Indexing…" : "Index Now"}
        </button>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="space-y-2 animate-fade-in">
        {[1, 2, 3, 4, 5].map((i) => (
          <div key={i} className="h-10 rounded-lg bg-surface animate-shimmer bg-gradient-to-r from-surface via-elevated to-surface" />
        ))}
      </div>
    );
  }

  return (
    <div className="animate-fade-in-up">
      {/* Search */}
      <div className="relative mb-4">
        <FileCode2 className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-text-muted" />
        <input
          type="text"
          value={search}
          onChange={(e) => onSearchChange(e.target.value)}
          placeholder="Search files…"
          className="w-full pl-9 pr-4 py-2.5 rounded-lg bg-surface border border-border-default text-sm text-text-primary placeholder:text-text-ghost focus:outline-none focus:border-accent/40 focus:shadow-[0_0_0_3px_rgba(0,212,255,0.1)] transition-all"
        />
      </div>

      {/* File tree */}
      <div className="glass-card rounded-xl overflow-hidden divide-y divide-border-subtle">
        {/* Header */}
        <div className="grid grid-cols-[1fr_80px_80px_60px_60px] gap-2 px-4 py-2 text-[11px] text-text-ghost font-medium uppercase tracking-wider">
          <span>File</span>
          <span className="text-right">Lines</span>
          <span className="text-right">Functions</span>
          <span className="text-right">Classes</span>
          <span className="text-right">Imports</span>
        </div>

        {tree.length === 0 ? (
          <div className="px-4 py-8 text-center text-sm text-text-muted">
            {search ? "No files match your search" : "No indexed files"}
          </div>
        ) : (
          <div className="max-h-[500px] overflow-y-auto">
            {tree.map((node) => (
              <TreeNode
                key={node.path}
                node={node}
                depth={0}
                expandedDirs={expandedDirs}
                onToggleDir={onToggleDir}
              />
            ))}
          </div>
        )}
      </div>

      {/* Summary */}
      {files.length > 0 && (
        <div className="mt-3 flex items-center gap-4 text-xs text-text-ghost">
          <span>{files.length} indexed files</span>
          <span>{formatNumber(files.reduce((s, f) => s + f.lines_count, 0))} total lines</span>
          <span>{formatNumber(files.reduce((s, f) => s + f.functions.length, 0))} functions</span>
          <span>{formatNumber(files.reduce((s, f) => s + f.classes.length, 0))} classes</span>
        </div>
      )}
    </div>
  );
}

/* ─── Tree Types + Builder ─── */

interface TreeNodeData {
  name: string;
  path: string;
  isDir: boolean;
  file?: ProjectFile;
  children: TreeNodeData[];
}

function buildTree(files: ProjectFile[]): TreeNodeData[] {
  const root: TreeNodeData = { name: "", path: "", isDir: true, children: [] };

  for (const file of files) {
    const parts = file.file_path.replace(/\\/g, "/").split("/");
    let current = root;

    for (let i = 0; i < parts.length; i++) {
      const part = parts[i];
      const isLast = i === parts.length - 1;
      const path = parts.slice(0, i + 1).join("/");

      if (isLast) {
        current.children.push({
          name: part,
          path,
          isDir: false,
          file,
          children: [],
        });
      } else {
        let dir = current.children.find((c) => c.isDir && c.name === part);
        if (!dir) {
          dir = { name: part, path, isDir: true, children: [] };
          current.children.push(dir);
        }
        current = dir;
      }
    }
  }

  // Sort: dirs first, then alphabetical
  const sortChildren = (node: TreeNodeData) => {
    node.children.sort((a, b) => {
      if (a.isDir !== b.isDir) return a.isDir ? -1 : 1;
      return a.name.localeCompare(b.name);
    });
    node.children.forEach(sortChildren);
  };
  sortChildren(root);

  // Collapse single-child directories
  const collapse = (nodes: TreeNodeData[]): TreeNodeData[] => {
    return nodes.map((node) => {
      if (node.isDir && node.children.length === 1 && node.children[0].isDir) {
        const child = node.children[0];
        return {
          ...child,
          name: `${node.name}/${child.name}`,
          path: child.path,
          children: collapse(child.children),
        };
      }
      return { ...node, children: node.isDir ? collapse(node.children) : node.children };
    });
  };

  return collapse(root.children);
}

/* ─── Tree Node Renderer ─── */

function TreeNode({
  node,
  depth,
  expandedDirs,
  onToggleDir,
}: {
  node: TreeNodeData;
  depth: number;
  expandedDirs: Set<string>;
  onToggleDir: (dir: string) => void;
}) {
  const isExpanded = expandedDirs.has(node.path);

  if (node.isDir) {
    return (
      <>
        <button
          onClick={() => onToggleDir(node.path)}
          className="w-full grid grid-cols-[1fr_80px_80px_60px_60px] gap-2 px-4 py-1.5 text-left hover:bg-hover/30 transition-colors"
          style={{ paddingLeft: `${16 + depth * 16}px` }}
        >
          <span className="flex items-center gap-1.5 text-sm text-text-secondary">
            {isExpanded ? (
              <ChevronDown className="w-3.5 h-3.5 text-text-ghost shrink-0" />
            ) : (
              <ChevronRight className="w-3.5 h-3.5 text-text-ghost shrink-0" />
            )}
            <FolderOpen className="w-3.5 h-3.5 text-amber shrink-0" />
            <span className="truncate">{node.name}</span>
          </span>
          <span />
          <span />
          <span />
          <span />
        </button>
        {isExpanded &&
          node.children.map((child) => (
            <TreeNode
              key={child.path}
              node={child}
              depth={depth + 1}
              expandedDirs={expandedDirs}
              onToggleDir={onToggleDir}
            />
          ))}
      </>
    );
  }

  const file = node.file!;
  return (
    <div
      className="grid grid-cols-[1fr_80px_80px_60px_60px] gap-2 px-4 py-1.5 hover:bg-hover/30 transition-colors"
      style={{ paddingLeft: `${16 + depth * 16 + 20}px` }}
    >
      <span className="flex items-center gap-1.5 text-sm text-text-primary min-w-0">
        <FileCode2 className="w-3.5 h-3.5 text-text-muted shrink-0" />
        <span className="truncate">{node.name}</span>
        {file.language && (
          <span
            className="w-2 h-2 rounded-full shrink-0"
            style={{ backgroundColor: getLanguageColor(file.language) }}
            title={file.language}
          />
        )}
      </span>
      <span className="text-xs text-text-muted font-mono text-right tabular-nums self-center">
        {formatNumber(file.lines_count)}
      </span>
      <span className="text-xs text-text-muted font-mono text-right tabular-nums self-center">
        {file.functions.length || "—"}
      </span>
      <span className="text-xs text-text-muted font-mono text-right tabular-nums self-center">
        {file.classes.length || "—"}
      </span>
      <span className="text-xs text-text-muted font-mono text-right tabular-nums self-center">
        {file.imports.length || "—"}
      </span>
    </div>
  );
}

function formatDuration(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return s > 0 ? `${m}m ${s}s` : `${m}m`;
}
