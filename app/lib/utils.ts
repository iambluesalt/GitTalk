import clsx, { type ClassValue } from "clsx";

export function cn(...inputs: ClassValue[]) {
  return clsx(inputs);
}

export function timeAgo(dateStr: string): string {
  const date = new Date(dateStr);
  const now = new Date();
  const seconds = Math.floor((now.getTime() - date.getTime()) / 1000);

  if (seconds < 60) return "just now";
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
  if (seconds < 604800) return `${Math.floor(seconds / 86400)}d ago`;
  return date.toLocaleDateString();
}

export function formatNumber(n: number | null | undefined): string {
  if (n == null) return "0";
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return n.toString();
}

export function getLanguageColor(lang: string): string {
  const colors: Record<string, string> = {
    python: "#3572A5",
    javascript: "#f1e05a",
    typescript: "#3178c6",
    go: "#00ADD8",
    rust: "#dea584",
    java: "#b07219",
    c: "#555555",
    "c++": "#f34b7d",
    cpp: "#f34b7d",
    ruby: "#701516",
    php: "#4F5D95",
    swift: "#F05138",
    kotlin: "#A97BFF",
    html: "#e34c26",
    css: "#563d7c",
    shell: "#89e051",
    markdown: "#083fa1",
  };
  return colors[lang.toLowerCase()] || "#8585a0";
}

export function statusLabel(status: string): string {
  const labels: Record<string, string> = {
    cloning: "Cloning…",
    cloned: "Ready to Index",
    indexing: "Indexing…",
    indexed: "Indexed",
    error: "Error",
  };
  return labels[status] || status;
}
