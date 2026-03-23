import { Link } from "react-router";
import {
  MessageSquare,
  Database,
  FileCode2,
  Trash2,
  Play,
  GitBranch,
  ExternalLink,
} from "lucide-react";
import type { Project } from "~/lib/types";
import {
  cn,
  timeAgo,
  formatNumber,
  getLanguageColor,
  statusLabel,
} from "~/lib/utils";

interface ProjectCardProps {
  project: Project;
  onDelete?: (id: string) => void;
  onIndex?: (id: string) => void;
  className?: string;
  style?: React.CSSProperties;
}

export default function ProjectCard({
  project,
  onDelete,
  onIndex,
  className,
  style,
}: ProjectCardProps) {
  const {
    id,
    name,
    github_url,
    status,
    analysis,
    cloned_at,
    last_indexed,
  } = project;

  const languages = analysis?.languages
    ? Object.entries(analysis.languages).sort(
        ([, a], [, b]) => b.lines_of_code - a.lines_of_code
      )
    : [];
  const totalLOC = languages.reduce((s, [, l]) => s + l.lines_of_code, 0) || 1;

  const isReady = status === "indexed";
  const isWorking = status === "cloning" || status === "indexing";
  const isError = status === "error";

  return (
    <div
      className={cn(
        "group relative flex flex-col rounded-xl border border-border-subtle bg-surface",
        "transition-all duration-300 hover:border-border-bright hover:shadow-lg hover:shadow-accent/5",
        isWorking && "border-amber/20",
        isError && "border-rose/20",
        className
      )}
      style={style}
    >
      {/* Header */}
      <div className="flex items-start gap-3 p-4 pb-2">
        <div
          className={cn(
            "flex items-center justify-center w-9 h-9 rounded-lg shrink-0 mt-0.5",
            isReady && "bg-green/10",
            isWorking && "bg-amber/10",
            isError && "bg-rose/10",
            status === "cloned" && "bg-accent/10"
          )}
        >
          <GitBranch
            className={cn(
              "w-4.5 h-4.5",
              isReady && "text-green",
              isWorking && "text-amber",
              isError && "text-rose",
              status === "cloned" && "text-accent"
            )}
          />
        </div>
        <div className="flex-1 min-w-0">
          <h3 className="font-display font-600 text-[0.95rem] text-text-primary truncate leading-tight">
            {name}
          </h3>
          <a
            href={github_url}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1 text-xs text-text-muted hover:text-accent transition-colors mt-0.5"
          >
            <span className="truncate max-w-[180px]">
              {github_url.replace("https://github.com/", "")}
            </span>
            <ExternalLink className="w-3 h-3 shrink-0" />
          </a>
        </div>
      </div>

      {/* Status badge */}
      <div className="px-4 pb-2">
        <span
          className={cn(
            "inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-xs font-medium",
            isReady && "bg-green/10 text-green",
            isWorking && "bg-amber/10 text-amber",
            isError && "bg-rose/10 text-rose",
            status === "cloned" && "bg-accent/10 text-accent"
          )}
        >
          {isWorking && (
            <span className="w-1.5 h-1.5 rounded-full bg-current animate-pulse" />
          )}
          {statusLabel(status)}
        </span>
      </div>

      {/* Stats */}
      {analysis && (
        <div className="grid grid-cols-3 gap-px mx-4 mb-3 rounded-lg overflow-hidden border border-border-subtle">
          <div className="flex flex-col items-center py-2 bg-elevated/50">
            <span className="text-xs text-text-muted">Files</span>
            <span className="text-sm font-mono font-600 text-text-primary">
              {formatNumber(analysis.total_files)}
            </span>
          </div>
          <div className="flex flex-col items-center py-2 bg-elevated/50">
            <span className="text-xs text-text-muted">Lines</span>
            <span className="text-sm font-mono font-600 text-text-primary">
              {formatNumber(analysis.total_lines)}
            </span>
          </div>
          <div className="flex flex-col items-center py-2 bg-elevated/50">
            <span className="text-xs text-text-muted">Size</span>
            <span className="text-sm font-mono font-600 text-text-primary">
              {analysis.repository_size_mb.toFixed(1)}MB
            </span>
          </div>
        </div>
      )}

      {/* Language bar */}
      {languages.length > 0 && (
        <div className="px-4 pb-3">
          <div className="flex h-1.5 rounded-full overflow-hidden bg-void">
            {languages.slice(0, 5).map(([lang, stats]) => (
              <div
                key={lang}
                className="h-full transition-all duration-500"
                style={{
                  width: `${(stats.lines_of_code / totalLOC) * 100}%`,
                  backgroundColor: getLanguageColor(lang),
                  minWidth: "3px",
                }}
                title={`${lang}: ${formatNumber(stats.lines_of_code)} lines`}
              />
            ))}
          </div>
          <div className="flex flex-wrap gap-x-3 gap-y-1 mt-2">
            {languages.slice(0, 3).map(([lang, stats]) => (
              <span
                key={lang}
                className="inline-flex items-center gap-1.5 text-xs text-text-muted"
              >
                <span
                  className="w-2 h-2 rounded-full shrink-0"
                  style={{ backgroundColor: getLanguageColor(lang) }}
                />
                {lang}
                <span className="text-text-ghost">
                  {((stats.lines_of_code / totalLOC) * 100).toFixed(0)}%
                </span>
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Footer actions */}
      <div className="flex items-center gap-2 px-4 py-3 mt-auto border-t border-border-subtle">
        {isReady && (
          <Link
            to={`/chat/${id}`}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-accent/10 text-accent hover:bg-accent/20 transition-colors"
          >
            <MessageSquare className="w-3.5 h-3.5" />
            Chat
          </Link>
        )}
        {status === "cloned" && onIndex && (
          <button
            onClick={() => onIndex(id)}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-green/10 text-green hover:bg-green/20 transition-colors"
          >
            <Database className="w-3.5 h-3.5" />
            Index
          </button>
        )}
        {isReady && onIndex && (
          <button
            onClick={() => onIndex(id)}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-elevated text-text-secondary hover:text-text-primary hover:bg-hover transition-colors"
          >
            <Database className="w-3.5 h-3.5" />
            Re-index
          </button>
        )}
        <span className="flex-1" />
        <span className="text-xs text-text-ghost">
          {timeAgo(last_indexed || cloned_at)}
        </span>
        {onDelete && (
          <button
            onClick={() => onDelete(id)}
            className="p-1.5 rounded-md text-text-ghost hover:text-rose hover:bg-rose/10 transition-colors sm:opacity-0 sm:group-hover:opacity-100"
            title="Delete project"
          >
            <Trash2 className="w-3.5 h-3.5" />
          </button>
        )}
      </div>
    </div>
  );
}
