export type ProjectStatus = "cloning" | "cloned" | "indexing" | "indexed" | "error";

export interface LanguageStats {
  file_count: number;
  lines_of_code: number;
  functions: number;
  classes: number;
  imports: number;
}

export interface RepositoryAnalysis {
  total_files: number;
  total_lines: number;
  repository_size_mb: number;
  file_types: Record<string, number>;
  languages: Record<string, LanguageStats>;
  primary_language: string | null;
}

export interface ProjectIndexCounts {
  files_indexed: number;
  chunks_created: number;
}

export interface Project {
  id: string;
  name: string;
  github_url: string;
  clone_path: string;
  status: ProjectStatus;
  cloned_at: string;
  last_indexed: string | null;
  last_used: string | null;
  analysis: RepositoryAnalysis | null;
  repo_map: string | null;
  error_message: string | null;
  index_counts?: ProjectIndexCounts;
}

export interface ProjectListResponse {
  projects: Project[];
  total: number;
  recent_projects: Project[];
}

export interface Conversation {
  id: string;
  project_id: string;
  title: string;
  created_at: string;
  updated_at: string;
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  created_at: string;
  sources: CodeReference[];
}

export interface CodeReference {
  file_path: string;
  line_start: number;
  line_end: number;
  code_snippet: string;
  relevance_score: number;
}

export interface SSEMessage {
  event: string;
  data: Record<string, unknown>;
}

export interface ChatModel {
  id: string;
  name: string;
  provider: string;
}

export interface IndexStats {
  files_indexed: number;
  total_lines: number;
  total_functions: number;
  total_classes: number;
  total_imports: number;
  languages_count: number;
  chunks_created: number;
}

export interface PreflightInfo {
  name: string;
  full_name: string;
  description: string | null;
  size_kb: number;
  size_mb: number;
  stars: number;
  forks: number;
  default_branch: string;
  language: string | null;
  updated_at: string | null;
  private: boolean;
  max_size_mb: number;
  size_warning: "ok" | "medium" | "large" | "too_large";
  size_warning_message: string;
}

export interface CloneProgress {
  phase: string;
  message: string;
  percent?: number;
  current_file?: string;
}

export interface IndexProgress {
  files_processed: number;
  total_files: number;
  current_file: string;
  chunks_created: number;
  percent: number;
}

export interface HealthResponse {
  status: "healthy" | "unhealthy";
  timestamp: string;
  version: string;
  services: Record<string, boolean | null>;
}

export interface AppConfig {
  app_name: string;
  llm_provider: "ollama" | "cloud" | "hybrid";
  ollama_base_url: string;
  ollama_model: string;
  ollama_embed_model: string;
  ollama_timeout: number;
  cloud_api_configured: boolean;
  cloud_api_provider: string | null;
  cloud_model: string | null;
  cloud_api_base_url: string | null;
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
  github_token_configured: boolean;
}

/** Partial config update — only include fields you want to change. */
export type ConfigUpdate = Partial<{
  llm_provider: "ollama" | "cloud" | "hybrid";
  ollama_base_url: string;
  ollama_model: string;
  ollama_embed_model: string;
  ollama_timeout: number;
  cloud_api_provider: string | null;
  cloud_api_key: string | null;
  cloud_api_base_url: string | null;
  cloud_model: string | null;
  github_token: string | null;
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
}>;
