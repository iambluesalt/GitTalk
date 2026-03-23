import { cn } from "~/lib/utils";

interface LogLine {
  type: "info" | "success" | "warning" | "error" | "dim";
  text: string;
  timestamp?: string;
}

interface TerminalProgressProps {
  lines: LogLine[];
  title?: string;
  percent?: number;
  isRunning?: boolean;
  className?: string;
  elapsedSeconds?: number;
  etaSeconds?: number | null;
}

function formatDuration(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return s > 0 ? `${m}m ${s}s` : `${m}m`;
}

export default function TerminalProgress({
  lines,
  title,
  percent,
  isRunning,
  className,
  elapsedSeconds,
  etaSeconds,
}: TerminalProgressProps) {
  const colorClass = {
    info: "info",
    success: "prompt",
    warning: "warn",
    error: "err",
    dim: "dim",
  };

  return (
    <div
      className={cn(
        "rounded-xl border border-border-subtle bg-base overflow-hidden",
        className
      )}
    >
      {/* Title bar */}
      <div className="flex items-center gap-2 px-4 py-2.5 bg-surface border-b border-border-subtle">
        <div className="flex gap-1.5">
          <span className="w-3 h-3 rounded-full bg-rose/60" />
          <span className="w-3 h-3 rounded-full bg-amber/60" />
          <span className="w-3 h-3 rounded-full bg-green/60" />
        </div>
        {title && (
          <span className="ml-2 text-xs font-mono text-text-muted">{title}</span>
        )}
        {/* Timing info */}
        <span className="ml-auto flex items-center gap-3">
          {elapsedSeconds != null && elapsedSeconds > 0 && (
            <span className="text-[11px] font-mono text-text-muted">
              {formatDuration(elapsedSeconds)}
            </span>
          )}
          {isRunning && etaSeconds != null && etaSeconds > 0 && (
            <span className="text-[11px] font-mono text-text-ghost">
              ~{formatDuration(etaSeconds)} left
            </span>
          )}
          {isRunning && (
            <span className="flex items-center gap-1.5 text-xs text-amber">
              <span className="w-1.5 h-1.5 rounded-full bg-amber animate-pulse" />
              Running
            </span>
          )}
        </span>
      </div>

      {/* Progress bar */}
      {percent != null && (
        <div className="h-1 bg-void">
          <div
            className={cn(
              "h-full transition-all duration-500 ease-out",
              percent >= 100
                ? "bg-green"
                : "bg-gradient-to-r from-accent to-purple"
            )}
            style={{ width: `${Math.min(percent, 100)}%` }}
          />
        </div>
      )}

      {/* Terminal output */}
      <div className="terminal-text p-4 max-h-80 overflow-y-auto">
        {lines.map((line, i) => (
          <div key={i} className="flex gap-2 animate-fade-in" style={{ animationDelay: `${Math.min(i * 30, 300)}ms` }}>
            {line.timestamp && (
              <span className="dim shrink-0">[{line.timestamp}]</span>
            )}
            <span className={colorClass[line.type]}>{line.text}</span>
          </div>
        ))}
        {isRunning && (
          <span className="inline-block w-2 h-4 bg-accent/80 animate-typewriter-blink" />
        )}
      </div>
    </div>
  );
}
