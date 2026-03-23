import { NavLink, Outlet, useLocation } from "react-router";
import {
  FolderGit2,
  MessageSquareCode,
  Home,
  Plus,
  PanelLeftOpen,
  PanelLeftClose,
  Settings,
  CircleCheck,
  CircleX,
  CircleDot,
  Server,
  Brain,
  Cloud,
  Database,
  ChevronDown,
  ChevronUp,
  Sun,
  Moon,
} from "lucide-react";
import { useState, useEffect, useCallback } from "react";
import { cn } from "~/lib/utils";
import { getHealth } from "~/lib/api";
import { useTheme } from "~/lib/theme";
import type { HealthResponse } from "~/lib/types";

const NAV_ITEMS = [
  { to: "/", icon: Home, label: "Dashboard", end: true },
  { to: "/projects", icon: FolderGit2, label: "Projects", end: false },
  { to: "/clone", icon: Plus, label: "Clone Repo", end: false },
];

const SERVICE_META: Record<
  string,
  { label: string; icon: typeof Server; critical?: boolean; description: string }
> = {
  metadata_db: {
    label: "Database",
    icon: Database,
    critical: true,
    description: "SQLite metadata store",
  },
  vector_db: {
    label: "Vector DB",
    icon: Database,
    critical: true,
    description: "LanceDB vector store",
  },
  ollama: {
    label: "Ollama",
    icon: Brain,
    description: "Local LLM inference",
  },
  cloud_api: {
    label: "Cloud API",
    icon: Cloud,
    description: "Cloud LLM fallback",
  },
  ollama_embed: {
    label: "Embeddings",
    icon: Server,
    description: "Nomic embedding model",
  },
};

export default function AppLayout() {
  const [collapsed, setCollapsed] = useState(false);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [backendDown, setBackendDown] = useState(false);
  const [servicesExpanded, setServicesExpanded] = useState(false);
  const location = useLocation();
  const { theme, toggle: toggleTheme } = useTheme();

  const fetchHealth = useCallback(() => {
    getHealth()
      .then((h) => {
        setHealth(h);
        setBackendDown(false);
      })
      .catch(() => {
        setHealth(null);
        setBackendDown(true);
      });
  }, []);

  // Poll health every 30s
  useEffect(() => {
    fetchHealth();
    const interval = setInterval(fetchHealth, 30_000);
    return () => clearInterval(interval);
  }, [fetchHealth]);

  const isChatRoute = location.pathname.startsWith("/chat/");

  // Derive service summary
  const services = health?.services || {};
  const llmAvailable = services.ollama === true || services.cloud_api === true;
  const canChat = llmAvailable && services.metadata_db === true && services.vector_db === true;
  const canIndex =
    services.metadata_db === true &&
    services.vector_db === true &&
    (services.ollama_embed === true || services.ollama_embed === null);

  return (
    <div className="flex h-screen overflow-hidden bg-void">
      {/* Sidebar */}
      <aside
        className={cn(
          "relative flex flex-col border-r border-border-subtle bg-base transition-all duration-300 ease-out shrink-0",
          collapsed ? "w-0 overflow-hidden border-r-0" : "w-60",
          isChatRoute && "max-lg:hidden"
        )}
      >
        {/* Logo */}
        <div className="flex items-center gap-3 px-4 h-13 border-b border-border-subtle shrink-0">
          <div className="flex items-center justify-center w-8 h-8 rounded-lg bg-accent/10 shrink-0">
            <MessageSquareCode className="w-4.5 h-4.5 text-accent" />
          </div>
          <span className="font-display font-700 text-lg tracking-tight text-text-primary whitespace-nowrap">
            GitTalk
          </span>
        </div>

        {/* Navigation */}
        <nav className="flex-1 px-2 py-3 space-y-1 overflow-y-auto">
          {NAV_ITEMS.map(({ to, icon: Icon, label, end }) => (
            <NavLink
              key={to}
              to={to}
              end={end}
              className={({ isActive }) =>
                cn(
                  "flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-all duration-200 whitespace-nowrap",
                  isActive
                    ? "bg-accent/10 text-accent shadow-[inset_0_0_0_1px_rgba(0,212,255,0.15)]"
                    : "text-text-secondary hover:text-text-primary hover:bg-hover"
                )
              }
            >
              <Icon className="w-4.5 h-4.5 shrink-0" />
              <span>{label}</span>
            </NavLink>
          ))}

          {/* Settings nav */}
          <NavLink
            to="/settings"
            className={({ isActive }) =>
              cn(
                "flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-all duration-200 whitespace-nowrap",
                isActive
                  ? "bg-accent/10 text-accent shadow-[inset_0_0_0_1px_rgba(0,212,255,0.15)]"
                  : "text-text-secondary hover:text-text-primary hover:bg-hover"
              )
            }
          >
            <Settings className="w-4.5 h-4.5 shrink-0" />
            <span>Settings</span>
          </NavLink>

          {/* Theme toggle */}
          <button
            onClick={toggleTheme}
            className="flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium text-text-secondary hover:text-text-primary hover:bg-hover transition-all duration-200 w-full whitespace-nowrap"
            title={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
          >
            {theme === "dark" ? (
              <Sun className="w-4.5 h-4.5 shrink-0" />
            ) : (
              <Moon className="w-4.5 h-4.5 shrink-0" />
            )}
            <span>{theme === "dark" ? "Light Mode" : "Dark Mode"}</span>
          </button>
        </nav>

        {/* Service status footer */}
        <div className="border-t border-border-subtle shrink-0">
          {/* Summary row */}
          <button
            onClick={() => setServicesExpanded(!servicesExpanded)}
            className="w-full flex items-center gap-2 px-3 py-2.5 text-left hover:bg-hover/50 transition-colors"
          >
            <StatusDot status={backendDown ? "offline" : health?.status === "healthy" ? "ok" : "warn"} />
            <span className="text-xs text-text-muted flex-1 truncate">
              {backendDown
                ? "Backend offline"
                : !llmAvailable
                ? "No LLM connected"
                : canChat
                ? "All systems ready"
                : "Partial services"}
            </span>
            {servicesExpanded ? (
              <ChevronUp className="w-3 h-3 text-text-ghost" />
            ) : (
              <ChevronDown className="w-3 h-3 text-text-ghost" />
            )}
          </button>

          {/* Expanded service list */}
          {servicesExpanded && (
            <div className="px-2 pb-2 space-y-0.5">
              {Object.entries(SERVICE_META).map(
                ([key, { label, icon: SvcIcon, critical, description }]) => {
                  const value = services[key];
                  const status =
                    value === true
                      ? "ok"
                      : value === false
                      ? "offline"
                      : "unknown";
                  return (
                    <div
                      key={key}
                      className="flex items-center gap-2 px-2 py-1.5 rounded-md"
                      title={description}
                    >
                      <StatusDot status={status} size="sm" />
                      <SvcIcon className="w-3 h-3 text-text-ghost" />
                      <span className="text-[11px] text-text-muted flex-1 truncate">
                        {label}
                      </span>
                      <span
                        className={cn(
                          "text-[10px] font-mono",
                          status === "ok" && "text-green",
                          status === "offline" && (critical ? "text-rose" : "text-amber"),
                          status === "unknown" && "text-text-ghost"
                        )}
                      >
                        {status === "ok"
                          ? "online"
                          : status === "offline"
                          ? "offline"
                          : "n/a"}
                      </span>
                    </div>
                  );
                }
              )}

              {/* Capability summary */}
              <div className="mt-1.5 pt-1.5 border-t border-border-subtle space-y-1 px-1">
                <CapabilityRow label="Chat" ready={canChat} />
                <CapabilityRow label="Clone" ready={services.metadata_db === true} />
                <CapabilityRow label="Index" ready={canIndex} />
              </div>
            </div>
          )}
        </div>
      </aside>

      {/* Main content */}
      <main className="flex-1 overflow-hidden relative">
        <button
          onClick={() => setCollapsed(!collapsed)}
          className="absolute top-3 left-3 z-20 p-1.5 rounded-md text-text-muted hover:text-text-secondary hover:bg-hover transition-colors"
          title={collapsed ? "Open sidebar" : "Close sidebar"}
        >
          {collapsed ? (
            <PanelLeftOpen className="w-4 h-4" />
          ) : (
            <PanelLeftClose className="w-4 h-4" />
          )}
        </button>
        <Outlet />
      </main>
    </div>
  );
}

function StatusDot({
  status,
  size = "md",
}: {
  status: "ok" | "warn" | "offline" | "unknown";
  size?: "sm" | "md";
}) {
  const dim = size === "sm" ? "w-1.5 h-1.5" : "w-2 h-2";
  return (
    <span
      className={cn(
        "rounded-full shrink-0",
        dim,
        status === "ok" && "bg-green",
        status === "warn" && "bg-amber animate-pulse",
        status === "offline" && "bg-rose",
        status === "unknown" && "bg-text-ghost"
      )}
    />
  );
}

function CapabilityRow({ label, ready }: { label: string; ready: boolean }) {
  return (
    <div className="flex items-center gap-1.5">
      {ready ? (
        <CircleCheck className="w-3 h-3 text-green" />
      ) : (
        <CircleX className="w-3 h-3 text-rose/60" />
      )}
      <span
        className={cn(
          "text-[10px]",
          ready ? "text-text-muted" : "text-text-ghost"
        )}
      >
        {label}
      </span>
    </div>
  );
}
