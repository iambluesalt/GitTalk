import { Link, useNavigate, useLocation } from "react-router";
import { useState, useEffect } from "react";
import {
  ArrowRight,
  FolderGit2,
  MessageSquareCode,
  Zap,
  Database,
  GitBranch,
  Search,
  Sparkles,
} from "lucide-react";
import type { Route } from "./+types/home";
import type { Project } from "~/lib/types";
import { getProjects } from "~/lib/api";
import ProjectCard from "~/components/ProjectCard";
import { cn, formatNumber } from "~/lib/utils";

export function meta({}: Route.MetaArgs) {
  return [
    { title: "GitTalk — Chat with any codebase" },
    {
      name: "description",
      content:
        "AI-powered codebase analysis with GitHub integration. Clone, index, and chat with any repository.",
    },
  ];
}

export default function Home() {
  const navigate = useNavigate();
  const location = useLocation();
  const [projects, setProjects] = useState<Project[]>([]);
  const [recentProjects, setRecentProjects] = useState<Project[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [quickUrl, setQuickUrl] = useState("");

  // Re-fetch whenever the user navigates to this page
  useEffect(() => {
    setLoading(true);
    setError(null);
    getProjects()
      .then((data) => {
        setProjects(data.projects);
        setRecentProjects(data.recent_projects);
      })
      .catch((err) => {
        setError(err instanceof Error ? err.message : "Failed to load projects");
      })
      .finally(() => setLoading(false));
  }, [location.key]);

  const totalFiles = projects.reduce(
    (s, p) => s + (p.analysis?.total_files || 0),
    0
  );
  const totalLines = projects.reduce(
    (s, p) => s + (p.analysis?.total_lines || 0),
    0
  );
  const indexedCount = projects.filter((p) => p.status === "indexed").length;

  const handleQuickClone = (e: React.FormEvent) => {
    e.preventDefault();
    if (quickUrl.trim()) {
      navigate(`/clone?url=${encodeURIComponent(quickUrl.trim())}`);
    }
  };

  return (
    <div className="h-full overflow-y-auto">
      <div className="max-w-5xl mx-auto px-6 py-8">
        {/* Hero */}
        <section className="relative mb-12">
          {/* Background glow — layered for depth */}
          <div className="absolute -top-24 left-1/2 -translate-x-1/2 w-[600px] h-[350px] bg-accent/6 rounded-full blur-[120px] pointer-events-none animate-breathe" />
          <div className="absolute -top-12 left-1/3 w-[350px] h-[250px] bg-purple/5 rounded-full blur-[100px] pointer-events-none" />
          <div className="absolute -top-8 right-1/4 w-[200px] h-[200px] bg-green/3 rounded-full blur-[80px] pointer-events-none" />

          <div className="relative text-center pt-10 pb-4">
            <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full border border-border-subtle bg-surface/80 backdrop-blur-sm text-xs text-text-secondary mb-6">
              <Sparkles className="w-3 h-3 text-amber" />
              AI-powered codebase analysis
            </div>
            <h1 className="font-display font-800 text-4xl sm:text-5xl tracking-tight mb-4">
              <span className="text-text-primary">Chat with </span>
              <span className="gradient-text">any codebase</span>
            </h1>
            <p className="text-text-secondary text-lg max-w-xl mx-auto mb-8 leading-relaxed">
              Clone a GitHub repository, index it with AST-aware chunking, and
              ask questions about the code using local AI.
            </p>

            {/* Quick clone input */}
            <form
              onSubmit={handleQuickClone}
              className="flex items-center max-w-lg mx-auto gap-2"
            >
              <div className="relative flex-1">
                <GitBranch className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-text-muted" />
                <input
                  type="text"
                  value={quickUrl}
                  onChange={(e) => setQuickUrl(e.target.value)}
                  placeholder="https://github.com/owner/repo"
                  className="w-full pl-10 pr-4 py-3 rounded-xl bg-surface border border-border-default text-text-primary placeholder:text-text-ghost text-sm font-mono focus:outline-none focus:border-accent/40 focus:shadow-[0_0_0_3px_rgba(0,212,255,0.1)] transition-all"
                />
              </div>
              <button
                type="submit"
                disabled={!quickUrl.trim()}
                className="flex items-center gap-2 px-5 py-3 rounded-xl bg-accent text-void font-display font-600 text-sm hover:bg-accent/90 disabled:opacity-40 disabled:cursor-not-allowed transition-all hover:shadow-lg hover:shadow-accent/20"
              >
                Clone
                <ArrowRight className="w-4 h-4" />
              </button>
            </form>
          </div>
        </section>

        {/* Stats row */}
        {projects.length > 0 && (
          <section className="grid grid-cols-3 gap-4 mb-10">
            <StatCard
              icon={FolderGit2}
              label="Projects"
              value={projects.length.toString()}
              accent="accent"
              index={0}
            />
            <StatCard
              icon={Database}
              label="Indexed"
              value={indexedCount.toString()}
              accent="green"
              index={1}
            />
            <StatCard
              icon={Zap}
              label="Lines Indexed"
              value={formatNumber(totalLines)}
              accent="purple"
              index={2}
            />
          </section>
        )}

        {/* Recent projects */}
        {!loading && recentProjects.length > 0 && (
          <section className="mb-10">
            <div className="flex items-center justify-between mb-4">
              <h2 className="font-display font-700 text-lg text-text-primary">
                Recent Projects
              </h2>
              <Link
                to="/projects"
                className="text-sm text-text-muted hover:text-accent transition-colors"
              >
                View all
                <ArrowRight className="w-3.5 h-3.5 inline ml-1" />
              </Link>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {recentProjects.slice(0, 3).map((project, i) => (
                <ProjectCard
                  key={project.id}
                  project={project}
                  className={cn("animate-fade-in-up", `stagger-${i + 1}`)}
                />
              ))}
            </div>
          </section>
        )}

        {/* Error state */}
        {!loading && error && (
          <section className="text-center py-12">
            <div className="inline-flex items-center justify-center w-14 h-14 rounded-2xl bg-rose/10 border border-rose/20 mb-4">
              <Zap className="w-7 h-7 text-rose" />
            </div>
            <h2 className="font-display font-700 text-lg text-text-primary mb-1">
              Backend unavailable
            </h2>
            <p className="text-sm text-text-secondary max-w-sm mx-auto">
              Could not connect. Make sure the backend is running on port 8000.
            </p>
          </section>
        )}

        {/* Empty state */}
        {!loading && !error && projects.length === 0 && (
          <section className="text-center py-16">
            <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-surface border border-border-subtle mb-6">
              <MessageSquareCode className="w-8 h-8 text-text-muted" />
            </div>
            <h2 className="font-display font-700 text-xl text-text-primary mb-2">
              No projects yet
            </h2>
            <p className="text-text-secondary mb-6 max-w-sm mx-auto">
              Clone a GitHub repository to get started. GitTalk will analyze the
              code and let you chat with it using AI.
            </p>
            <Link
              to="/clone"
              className="inline-flex items-center gap-2 px-6 py-3 rounded-xl bg-accent text-void font-display font-600 text-sm hover:bg-accent/90 transition-all hover:shadow-lg hover:shadow-accent/20"
            >
              Clone your first repo
              <ArrowRight className="w-4 h-4" />
            </Link>
          </section>
        )}

        {/* How it works */}
        <section className="mt-4 mb-8">
          <h2 className="font-display font-700 text-lg text-text-primary mb-6 text-center">
            How it works
          </h2>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            {[
              {
                icon: GitBranch,
                title: "1. Clone",
                desc: "Paste a GitHub URL. GitTalk shallow-clones the repository locally.",
                color: "accent",
              },
              {
                icon: Search,
                title: "2. Index",
                desc: "AST parsing, chunking, and embedding create a searchable code index.",
                color: "purple",
              },
              {
                icon: MessageSquareCode,
                title: "3. Chat",
                desc: "Ask questions. Hybrid search + RAG delivers context-aware answers.",
                color: "green",
              },
            ].map((step, i) => (
              <div
                key={step.title}
                className={cn(
                  "glass-card rounded-xl p-5 animate-fade-in-up group/step hover:border-border-bright transition-all duration-300",
                  `stagger-${i + 2}`
                )}
              >
                <div className="flex items-center gap-3 mb-3">
                  <div
                    className="inline-flex items-center justify-center w-10 h-10 rounded-lg shrink-0 transition-transform duration-300 group-hover/step:scale-110"
                    style={{
                      backgroundColor: `color-mix(in srgb, var(--color-${step.color}) 10%, transparent)`,
                    }}
                  >
                    <step.icon
                      className="w-5 h-5"
                      style={{ color: `var(--color-${step.color})` }}
                    />
                  </div>
                  <h3 className="font-display font-600 text-sm text-text-primary">
                    {step.title}
                  </h3>
                </div>
                <p className="text-xs text-text-secondary leading-relaxed">
                  {step.desc}
                </p>
              </div>
            ))}
          </div>
        </section>
      </div>
    </div>
  );
}

function StatCard({
  icon: Icon,
  label,
  value,
  accent,
  index,
}: {
  icon: React.ComponentType<{ className?: string; style?: React.CSSProperties }>;
  label: string;
  value: string;
  accent: string;
  index: number;
}) {
  return (
    <div className={cn(
      "stat-card glass-card rounded-xl p-4 flex items-center gap-3 hover:border-border-bright transition-all duration-300 animate-fade-in-up",
      `stagger-${index + 1}`
    )}>
      <div
        className="flex items-center justify-center w-11 h-11 rounded-xl shrink-0"
        style={{ backgroundColor: `color-mix(in srgb, var(--color-${accent}) 12%, transparent)` }}
      >
        <Icon
          className="w-5 h-5"
          style={{ color: `var(--color-${accent})` }}
        />
      </div>
      <div>
        <div className="text-2xl font-mono font-700 text-text-primary leading-tight tracking-tight">
          {value}
        </div>
        <div className="text-[11px] text-text-muted uppercase tracking-wider font-medium mt-0.5">{label}</div>
      </div>
    </div>
  );
}
