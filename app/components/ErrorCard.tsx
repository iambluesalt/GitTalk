import { Link } from "react-router";
import { AlertTriangle, ArrowRight } from "lucide-react";
import type { HumanError } from "~/lib/errors";
import { cn } from "~/lib/utils";

interface ErrorCardProps {
  error: HumanError;
  className?: string;
}

export default function ErrorCard({ error, className }: ErrorCardProps) {
  return (
    <div
      className={cn(
        "rounded-xl border border-rose/20 bg-rose/5 p-4 animate-fade-in",
        className
      )}
    >
      <div className="flex items-start gap-3">
        <div className="flex items-center justify-center w-8 h-8 rounded-lg bg-rose/10 shrink-0 mt-0.5">
          <AlertTriangle className="w-4 h-4 text-rose" />
        </div>
        <div className="flex-1 min-w-0">
          <p className="text-sm font-medium text-text-primary">
            {error.title}
          </p>
          <p className="text-xs text-text-secondary mt-1 leading-relaxed">
            {error.message}
          </p>
          {error.hint && (
            <div className="mt-2 p-2 rounded-lg bg-base border border-border-subtle">
              <p className="text-xs text-text-secondary leading-relaxed">
                <span className="font-medium text-amber">Tip: </span>
                <code className="font-mono text-[11px] text-accent/80">
                  {error.hint}
                </code>
              </p>
            </div>
          )}
          {error.action && (
            <Link
              to={error.action.to}
              className="inline-flex items-center gap-1.5 mt-2.5 px-3 py-1.5 rounded-lg bg-accent/10 text-accent text-xs font-medium hover:bg-accent/20 transition-colors"
            >
              {error.action.label}
              <ArrowRight className="w-3 h-3" />
            </Link>
          )}
        </div>
      </div>
    </div>
  );
}
