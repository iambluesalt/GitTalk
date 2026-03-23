import { useState, useEffect, useMemo } from "react";
import { useNavigate, useLocation } from "react-router";
import { Search, Plus, Trash2, Filter, FolderGit2 } from "lucide-react";
import type { Route } from "./+types/projects";
import type { Project } from "~/lib/types";
import { getProjects, deleteProject, indexProject, parseSSEStream } from "~/lib/api";
import ProjectCard from "~/components/ProjectCard";
import { cn } from "~/lib/utils";

export function meta({}: Route.MetaArgs) {
  return [{ title: "Projects — GitTalk" }];
}

type StatusFilter = "all" | "indexed" | "cloned" | "error";

export default function Projects() {
  const navigate = useNavigate();
  const location = useLocation();
  const [projects, setProjects] = useState<Project[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");
  const [filter, setFilter] = useState<StatusFilter>("all");
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = () => {
    setLoading(true);
    setError(null);
    getProjects()
      .then((data) => setProjects(data.projects))
      .catch((err) => {
        setError(err instanceof Error ? err.message : "Failed to load projects");
      })
      .finally(() => setLoading(false));
  };

  // Re-fetch whenever the user navigates to this page
  useEffect(() => { load(); }, [location.key]);

  const filtered = useMemo(() => {
    let result = projects;
    if (filter !== "all") {
      result = result.filter((p) => p.status === filter);
    }
    if (search.trim()) {
      const q = search.toLowerCase();
      result = result.filter(
        (p) =>
          p.name.toLowerCase().includes(q) ||
          p.github_url.toLowerCase().includes(q) ||
          (p.analysis?.primary_language || "").toLowerCase().includes(q)
      );
    }
    return result;
  }, [projects, filter, search]);

  const handleDelete = async (id: string) => {
    setDeletingId(id);
    try {
      await deleteProject(id);
      setProjects((prev) => prev.filter((p) => p.id !== id));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to delete project");
    } finally {
      setDeletingId(null);
    }
  };

  const handleIndex = async (id: string) => {
    navigate(`/clone?index=${id}`);
  };

  const counts = {
    all: projects.length,
    indexed: projects.filter((p) => p.status === "indexed").length,
    cloned: projects.filter((p) => p.status === "cloned").length,
    error: projects.filter((p) => p.status === "error").length,
  };

  return (
    <div className="h-full overflow-y-auto">
      <div className="max-w-5xl mx-auto px-6 py-8">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div>
            <h1 className="font-display font-800 text-2xl text-text-primary">
              Projects
            </h1>
            <p className="text-sm text-text-secondary mt-1">
              {projects.length} repositories cloned
            </p>
          </div>
          <button
            onClick={() => navigate("/clone")}
            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-accent text-void font-display font-600 text-sm hover:bg-accent/90 transition-all hover:shadow-lg hover:shadow-accent/20"
          >
            <Plus className="w-4 h-4" />
            Clone New
          </button>
        </div>

        {/* Filters */}
        <div className="flex flex-col sm:flex-row gap-3 mb-6">
          {/* Search */}
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-text-muted" />
            <input
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search projects…"
              className="w-full pl-9 pr-4 py-2.5 rounded-lg bg-surface border border-border-default text-sm text-text-primary placeholder:text-text-ghost focus:outline-none focus:border-accent/40 focus:shadow-[0_0_0_3px_rgba(0,212,255,0.1)] transition-all"
            />
          </div>

          {/* Status tabs */}
          <div className="flex gap-1 p-1 rounded-lg bg-surface border border-border-subtle">
            {(["all", "indexed", "cloned", "error"] as StatusFilter[]).map(
              (status) => (
                <button
                  key={status}
                  onClick={() => setFilter(status)}
                  className={cn(
                    "flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-colors",
                    filter === status
                      ? "bg-elevated text-text-primary shadow-sm"
                      : "text-text-muted hover:text-text-secondary"
                  )}
                >
                  {status.charAt(0).toUpperCase() + status.slice(1)}
                  <span
                    className={cn(
                      "text-[10px] px-1.5 py-0.5 rounded-full",
                      filter === status
                        ? "bg-accent/15 text-accent"
                        : "bg-border-subtle text-text-ghost"
                    )}
                  >
                    {counts[status]}
                  </span>
                </button>
              )
            )}
          </div>
        </div>

        {/* Error banner */}
        {error && (
          <div className="mb-4 p-3 rounded-lg border border-rose/20 bg-rose/5 text-sm text-rose">
            {error}
          </div>
        )}

        {/* Grid */}
        {loading ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {[1, 2, 3].map((i) => (
              <div
                key={i}
                className="h-52 rounded-xl border border-border-subtle bg-surface animate-shimmer bg-gradient-to-r from-surface via-elevated to-surface"
              />
            ))}
          </div>
        ) : filtered.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {filtered.map((project, i) => (
              <ProjectCard
                key={project.id}
                project={project}
                onDelete={handleDelete}
                onIndex={handleIndex}
                className={cn("animate-fade-in-up", `stagger-${Math.min(i + 1, 6)}`)}
              />
            ))}
          </div>
        ) : (
          <div className="text-center py-16">
            <FolderGit2 className="w-12 h-12 text-text-ghost mx-auto mb-4" />
            <p className="text-text-secondary">
              {search || filter !== "all"
                ? "No projects match your filters"
                : "No projects yet"}
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
