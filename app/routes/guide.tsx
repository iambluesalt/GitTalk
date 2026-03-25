import { Link } from "react-router";
import {
  GitBranch,
  Search,
  MessageSquareCode,
  Settings,
  FolderGit2,
  ArrowRight,
  Terminal,
  Cpu,
  Brain,
  Cloud,
  Database,
  Trash2,
  RefreshCw,
  ChevronRight,
  Sparkles,
  Zap,
  Eye,
  FileCode2,
  BookOpen,
  Lightbulb,
  CheckCircle2,
  AlertTriangle,
  Globe,
  Key,
} from "lucide-react";
import type { Route } from "./+types/guide";
import { cn } from "~/lib/utils";

export function meta({}: Route.MetaArgs) {
  return [{ title: "Guide — GitTalk" }];
}

export default function Guide() {
  return (
    <div className="h-full overflow-y-auto">
      <div className="max-w-4xl mx-auto px-6 py-8">
        {/* Header */}
        <section className="relative mb-12">
          <div className="absolute -top-20 left-1/2 -translate-x-1/2 w-[500px] h-[300px] bg-purple/5 rounded-full blur-[120px] pointer-events-none" />
          <div className="absolute -top-10 right-1/3 w-[250px] h-[200px] bg-accent/4 rounded-full blur-[100px] pointer-events-none" />

          <div className="relative pt-10 pb-2">
            <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full border border-border-subtle bg-surface/80 backdrop-blur-sm text-xs text-text-secondary mb-6">
              <BookOpen className="w-3 h-3 text-purple" />
              User Guide
            </div>
            <h1 className="font-display font-800 text-3xl sm:text-4xl tracking-tight mb-3 text-text-primary">
              Getting started with{" "}
              <span className="gradient-text">GitTalk</span>
            </h1>
            <p className="text-text-secondary text-lg max-w-2xl leading-relaxed">
              Clone any GitHub repository, index it with AST-aware code parsing,
              and chat with the codebase using local or cloud AI models.
            </p>
          </div>
        </section>

        {/* Quick nav */}
        <nav className="glass-card rounded-xl p-4 mb-10">
          <p className="text-xs font-medium text-text-muted uppercase tracking-wider mb-3">
            Jump to
          </p>
          <div className="flex flex-wrap gap-2">
            {[
              { label: "Prerequisites", href: "#prerequisites" },
              { label: "Clone a Repo", href: "#clone" },
              { label: "Index the Code", href: "#index" },
              { label: "Chat with Code", href: "#chat" },
              { label: "Manage Projects", href: "#projects" },
              { label: "Settings", href: "#settings" },
              { label: "Tips", href: "#tips" },
            ].map((item) => (
              <a
                key={item.href}
                href={item.href}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-surface border border-border-subtle text-xs text-text-secondary hover:text-accent hover:border-accent/30 transition-all"
              >
                <ChevronRight className="w-3 h-3" />
                {item.label}
              </a>
            ))}
          </div>
        </nav>

        {/* Prerequisites */}
        <GuideSection
          id="prerequisites"
          icon={Zap}
          color="amber"
          title="Prerequisites"
          step={0}
        >
          <p className="text-text-secondary text-sm leading-relaxed mb-4">
            Before using GitTalk, make sure you have the following set up:
          </p>
          <div className="grid gap-3">
            <PrereqCard
              icon={Terminal}
              title="Python Backend"
              description="The FastAPI backend must be running. Start it from the app/backend directory."
              command="cd app/backend && python main.py"
            />
            <PrereqCard
              icon={Brain}
              title="Ollama (recommended)"
              description="Install Ollama for local LLM inference and embeddings. Pull a chat model and the embedding model."
              command="ollama pull llama3.2 && ollama pull nomic-embed-text"
            />
            <PrereqCard
              icon={Cloud}
              title="Cloud API (alternative)"
              description="Alternatively, configure a cloud LLM provider (OpenAI, Anthropic, etc.) in Settings with your API key."
            />
            <PrereqCard
              icon={Globe}
              title="Frontend Dev Server"
              description="Start the React frontend development server."
              command="npm run dev"
            />
          </div>
          <div className="mt-4 flex items-start gap-2 p-3 rounded-lg bg-amber/5 border border-amber/10">
            <AlertTriangle className="w-4 h-4 text-amber shrink-0 mt-0.5" />
            <p className="text-xs text-text-secondary leading-relaxed">
              Check the{" "}
              <Link
                to="/settings"
                className="text-accent hover:underline"
              >
                Settings page
              </Link>{" "}
              to verify all services are online. The sidebar status indicator
              will show green when everything is ready.
            </p>
          </div>
        </GuideSection>

        {/* Step 1: Clone */}
        <GuideSection
          id="clone"
          icon={GitBranch}
          color="accent"
          title="Step 1 — Clone a Repository"
          step={1}
        >
          <StepList
            steps={[
              {
                title: "Navigate to Clone",
                description: (
                  <>
                    Go to the{" "}
                    <Link
                      to="/clone"
                      className="text-accent hover:underline"
                    >
                      Clone Repo
                    </Link>{" "}
                    page from the sidebar, or use the quick-clone input on the
                    Dashboard.
                  </>
                ),
              },
              {
                title: "Enter a GitHub URL",
                description:
                  "Paste any public GitHub repository URL. Private repos are supported if you add a GitHub token in Settings.",
              },
              {
                title: "Review the preflight check",
                description:
                  "GitTalk fetches repository metadata (size, stars, description) and shows warnings if the repo is very large. Repos over 500 MB are blocked by default.",
              },
              {
                title: "Start cloning",
                description:
                  'Click "Clone Repository" to begin. A terminal-style progress view shows real-time streaming logs, elapsed time, and ETA. GitTalk uses shallow cloning (depth=1) for speed.',
              },
              {
                title: "Optionally auto-index",
                description:
                  'Check "Index after cloning" to automatically start indexing once the clone completes. Otherwise you can index later from the project detail page.',
              },
            ]}
          />
          <InfoBox color="accent">
            You can also re-clone a repository by using the same URL with the
            force option. This replaces the existing local copy.
          </InfoBox>
        </GuideSection>

        {/* Step 2: Index */}
        <GuideSection
          id="index"
          icon={Search}
          color="purple"
          title="Step 2 — Index the Code"
          step={2}
        >
          <p className="text-text-secondary text-sm leading-relaxed mb-4">
            Indexing transforms raw code into a searchable knowledge base. This
            is what makes GitTalk's answers context-aware and accurate.
          </p>
          <StepList
            steps={[
              {
                title: "Trigger indexing",
                description:
                  'From the project detail page or the projects list, click "Index" on any cloned repository. You can also check "Index after cloning" during the clone step.',
              },
              {
                title: "AST parsing & chunking",
                description:
                  "GitTalk uses Tree-sitter to parse code into its Abstract Syntax Tree, then intelligently chunks by functions, classes, and logical blocks rather than arbitrary line counts.",
              },
              {
                title: "Embedding generation",
                description:
                  "Each chunk is embedded using a local model (nomic-embed-text by default via Ollama) and stored in LanceDB for fast vector search.",
              },
              {
                title: "Monitor progress",
                description:
                  "The terminal view shows files processed, chunks created, and timing statistics in real-time via SSE streaming.",
              },
            ]}
          />
          <div className="grid sm:grid-cols-3 gap-3 mt-4">
            <MiniStat
              icon={FileCode2}
              label="AST-Aware"
              desc="Chunks respect code structure"
            />
            <MiniStat
              icon={Database}
              label="Vector Storage"
              desc="LanceDB for fast retrieval"
            />
            <MiniStat
              icon={Cpu}
              label="Local Embeddings"
              desc="No data leaves your machine"
            />
          </div>
          <InfoBox color="purple">
            Re-indexing a project clears the previous index and rebuilds from
            scratch. This is useful after the repo has changed or if you want to
            try different settings.
          </InfoBox>
        </GuideSection>

        {/* Step 3: Chat */}
        <GuideSection
          id="chat"
          icon={MessageSquareCode}
          color="green"
          title="Step 3 — Chat with Your Code"
          step={3}
        >
          <p className="text-text-secondary text-sm leading-relaxed mb-4">
            Once a project is indexed, you can ask questions and get
            context-aware answers grounded in the actual source code.
          </p>
          <StepList
            steps={[
              {
                title: "Open the chat",
                description: (
                  <>
                    Click <strong>Chat</strong> on any indexed project card, or
                    navigate directly from the project detail page.
                  </>
                ),
              },
              {
                title: "Choose a model",
                description:
                  "Use the model selector dropdown at the top to pick which LLM to use (e.g., llama3.2 via Ollama, or a cloud model).",
              },
              {
                title: "Ask a question",
                description:
                  'Type your question in the input area and press Enter or click Send. Try things like "How does authentication work?" or "Explain the database schema".',
              },
              {
                title: "View source references",
                description:
                  'Each response includes expandable source attribution. Click "X sources referenced" to see the exact code snippets, file paths, line numbers, and relevance scores that informed the answer.',
              },
              {
                title: "Manage conversations",
                description:
                  "The left sidebar shows your conversation history. Create new conversations, switch between them, or delete old ones.",
              },
            ]}
          />
          <div className="mt-4 glass-card rounded-xl p-4">
            <p className="text-xs font-medium text-text-muted uppercase tracking-wider mb-3">
              Example questions to try
            </p>
            <div className="space-y-2">
              {[
                "What does this project do? Give me a high-level overview.",
                "How are API routes organized and what endpoints exist?",
                "Walk me through the error handling strategy.",
                "Find potential security issues in the authentication code.",
                "How would I add a new feature to the settings page?",
              ].map((q) => (
                <div
                  key={q}
                  className="flex items-start gap-2 text-sm text-text-secondary"
                >
                  <ChevronRight className="w-3.5 h-3.5 text-green shrink-0 mt-0.5" />
                  <span>{q}</span>
                </div>
              ))}
            </div>
          </div>
          <InfoBox color="green">
            Responses stream in real-time. If a model supports reasoning (like
            thinking blocks), you can expand them to see the model's
            step-by-step thought process.
          </InfoBox>
        </GuideSection>

        {/* Managing projects */}
        <GuideSection
          id="projects"
          icon={FolderGit2}
          color="accent"
          title="Managing Projects"
          step={0}
        >
          <p className="text-text-secondary text-sm leading-relaxed mb-4">
            The{" "}
            <Link to="/projects" className="text-accent hover:underline">
              Projects
            </Link>{" "}
            page gives you an overview of all cloned repositories.
          </p>
          <div className="grid sm:grid-cols-2 gap-3">
            <FeatureCard
              icon={Eye}
              title="Project Detail"
              description="Click any project card to see full analysis details, browse indexed files, and view statistics."
            />
            <FeatureCard
              icon={Search}
              title="Filter & Search"
              description="Filter projects by status (All, Indexed, Cloned, Error) and search by name."
            />
            <FeatureCard
              icon={RefreshCw}
              title="Re-index"
              description="Rebuild the search index for any project. Useful after code changes or config updates."
            />
            <FeatureCard
              icon={Trash2}
              title="Delete"
              description="Remove a project and all its data (cloned files, index, conversations)."
            />
          </div>
          <InfoBox color="accent">
            The project detail page includes a file browser tab where you can
            explore the indexed files and their directory structure.
          </InfoBox>
        </GuideSection>

        {/* Settings */}
        <GuideSection
          id="settings"
          icon={Settings}
          color="amber"
          title="Settings & Configuration"
          step={0}
        >
          <p className="text-text-secondary text-sm leading-relaxed mb-4">
            The{" "}
            <Link to="/settings" className="text-accent hover:underline">
              Settings
            </Link>{" "}
            page lets you configure LLM providers, check service health, and
            tune advanced parameters.
          </p>
          <div className="space-y-4">
            <div className="glass-card rounded-xl p-4">
              <h4 className="font-display font-600 text-sm text-text-primary mb-3 flex items-center gap-2">
                <Brain className="w-4 h-4 text-purple" />
                LLM Provider
              </h4>
              <p className="text-xs text-text-secondary leading-relaxed mb-2">
                Choose between <strong>Ollama</strong> (local, private, free) or
                a <strong>Cloud API</strong> (OpenAI, Anthropic, etc.). You can
                configure both and switch between them.
              </p>
              <div className="grid sm:grid-cols-2 gap-2 mt-3">
                <div className="p-2.5 rounded-lg bg-surface border border-border-subtle">
                  <p className="text-xs font-mono text-green mb-1">Ollama</p>
                  <p className="text-[11px] text-text-muted">
                    Base URL, model name, embedding model, timeout
                  </p>
                </div>
                <div className="p-2.5 rounded-lg bg-surface border border-border-subtle">
                  <p className="text-xs font-mono text-accent mb-1">Cloud API</p>
                  <p className="text-[11px] text-text-muted">
                    Provider, API key, base URL, model
                  </p>
                </div>
              </div>
            </div>

            <div className="glass-card rounded-xl p-4">
              <h4 className="font-display font-600 text-sm text-text-primary mb-3 flex items-center gap-2">
                <Key className="w-4 h-4 text-amber" />
                GitHub Token
              </h4>
              <p className="text-xs text-text-secondary leading-relaxed">
                Add a personal access token to clone private repositories and
                increase GitHub API rate limits. The token is stored locally and
                never shared.
              </p>
            </div>

            <div className="glass-card rounded-xl p-4">
              <h4 className="font-display font-600 text-sm text-text-primary mb-3 flex items-center gap-2">
                <Sparkles className="w-4 h-4 text-accent" />
                Service Health Monitor
              </h4>
              <p className="text-xs text-text-secondary leading-relaxed">
                The health panel shows real-time status of all services: Database,
                Vector DB, Ollama, Cloud API, and Embeddings. The sidebar also
                shows a summary indicator that polls every 30 seconds.
              </p>
            </div>
          </div>
        </GuideSection>

        {/* Tips */}
        <GuideSection
          id="tips"
          icon={Lightbulb}
          color="green"
          title="Tips & Best Practices"
          step={0}
        >
          <div className="space-y-3">
            {[
              {
                title: "Start with smaller repos",
                description:
                  "Repos under 100 MB index fastest. Try GitTalk on a few small projects first to get a feel for how it works.",
              },
              {
                title: "Be specific in your questions",
                description:
                  'Instead of "explain the code", try "how does the authentication middleware validate JWT tokens?" — specific questions get better source retrieval.',
              },
              {
                title: "Check the sources",
                description:
                  "Always expand the source references to verify the answer is grounded in actual code. High relevance scores (green) indicate strong matches.",
              },
              {
                title: "Use conversations for different topics",
                description:
                  "Create separate conversations for different areas of investigation. This keeps context focused and makes it easier to revisit.",
              },
              {
                title: "Re-index after config changes",
                description:
                  "If you change embedding models or chunk settings, re-index your projects to take advantage of the new configuration.",
              },
              {
                title: "Keep Ollama running",
                description:
                  "Ollama needs to be running in the background for local inference and embeddings. The sidebar health indicator will show you if it goes offline.",
              },
            ].map((tip, i) => (
              <div
                key={tip.title}
                className="flex items-start gap-3 p-3 rounded-lg bg-surface/50 border border-border-subtle hover:border-border-default transition-colors"
              >
                <div className="flex items-center justify-center w-6 h-6 rounded-md bg-green/10 shrink-0 mt-0.5">
                  <span className="text-xs font-mono font-bold text-green">
                    {i + 1}
                  </span>
                </div>
                <div>
                  <p className="text-sm font-medium text-text-primary mb-0.5">
                    {tip.title}
                  </p>
                  <p className="text-xs text-text-secondary leading-relaxed">
                    {tip.description}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </GuideSection>

        {/* CTA */}
        <section className="text-center py-10 mt-4">
          <h2 className="font-display font-700 text-xl text-text-primary mb-2">
            Ready to get started?
          </h2>
          <p className="text-sm text-text-secondary mb-6 max-w-md mx-auto">
            Clone your first repository and start chatting with code in minutes.
          </p>
          <Link
            to="/clone"
            className="inline-flex items-center gap-2 px-6 py-3 rounded-xl bg-accent text-void font-display font-600 text-sm hover:bg-accent/90 transition-all hover:shadow-lg hover:shadow-accent/20"
          >
            Clone a Repository
            <ArrowRight className="w-4 h-4" />
          </Link>
        </section>
      </div>
    </div>
  );
}

/* ── Sub-components ── */

function GuideSection({
  id,
  icon: Icon,
  color,
  title,
  step,
  children,
}: {
  id: string;
  icon: React.ComponentType<{ className?: string; style?: React.CSSProperties }>;
  color: string;
  title: string;
  step: number;
  children: React.ReactNode;
}) {
  return (
    <section id={id} className="mb-12 scroll-mt-8">
      <div className="flex items-center gap-3 mb-5">
        <div
          className="flex items-center justify-center w-10 h-10 rounded-xl shrink-0"
          style={{
            backgroundColor: `color-mix(in srgb, var(--color-${color}) 12%, transparent)`,
          }}
        >
          <Icon
            className="w-5 h-5"
            style={{ color: `var(--color-${color})` }}
          />
        </div>
        <h2 className="font-display font-700 text-xl text-text-primary">
          {title}
        </h2>
      </div>
      {children}
    </section>
  );
}

function StepList({
  steps,
}: {
  steps: { title: string; description: React.ReactNode }[];
}) {
  return (
    <div className="space-y-3">
      {steps.map((step, i) => (
        <div key={i} className="flex gap-3">
          <div className="flex flex-col items-center">
            <div className="flex items-center justify-center w-7 h-7 rounded-lg bg-surface border border-border-subtle shrink-0">
              <span className="text-xs font-mono font-bold text-text-secondary">
                {i + 1}
              </span>
            </div>
            {i < steps.length - 1 && (
              <div className="w-px flex-1 bg-border-subtle mt-1" />
            )}
          </div>
          <div className="pb-4">
            <p className="text-sm font-medium text-text-primary mb-1">
              {step.title}
            </p>
            <p className="text-xs text-text-secondary leading-relaxed">
              {step.description}
            </p>
          </div>
        </div>
      ))}
    </div>
  );
}

function PrereqCard({
  icon: Icon,
  title,
  description,
  command,
}: {
  icon: React.ComponentType<{ className?: string }>;
  title: string;
  description: string;
  command?: string;
}) {
  return (
    <div className="glass-card rounded-xl p-4">
      <div className="flex items-center gap-2 mb-2">
        <Icon className="w-4 h-4 text-text-muted" />
        <h4 className="text-sm font-medium text-text-primary">{title}</h4>
      </div>
      <p className="text-xs text-text-secondary leading-relaxed mb-2">
        {description}
      </p>
      {command && (
        <code className="block text-xs font-mono bg-base border border-border-subtle rounded-lg px-3 py-2 text-accent">
          {command}
        </code>
      )}
    </div>
  );
}

function FeatureCard({
  icon: Icon,
  title,
  description,
}: {
  icon: React.ComponentType<{ className?: string }>;
  title: string;
  description: string;
}) {
  return (
    <div className="glass-card rounded-xl p-4 hover:border-border-bright transition-colors">
      <div className="flex items-center gap-2 mb-2">
        <Icon className="w-4 h-4 text-accent" />
        <h4 className="text-sm font-medium text-text-primary">{title}</h4>
      </div>
      <p className="text-xs text-text-secondary leading-relaxed">
        {description}
      </p>
    </div>
  );
}

function MiniStat({
  icon: Icon,
  label,
  desc,
}: {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  desc: string;
}) {
  return (
    <div className="glass-card rounded-xl p-3 text-center">
      <Icon className="w-5 h-5 text-purple mx-auto mb-1.5" />
      <p className="text-xs font-medium text-text-primary">{label}</p>
      <p className="text-[11px] text-text-muted">{desc}</p>
    </div>
  );
}

function InfoBox({
  color,
  children,
}: {
  color: string;
  children: React.ReactNode;
}) {
  return (
    <div
      className="mt-4 flex items-start gap-2 p-3 rounded-lg"
      style={{
        backgroundColor: `color-mix(in srgb, var(--color-${color}) 5%, transparent)`,
        borderWidth: 1,
        borderStyle: "solid",
        borderColor: `color-mix(in srgb, var(--color-${color}) 10%, transparent)`,
      }}
    >
      <Lightbulb
        className="w-4 h-4 shrink-0 mt-0.5"
        style={{ color: `var(--color-${color})` }}
      />
      <p className="text-xs text-text-secondary leading-relaxed">{children}</p>
    </div>
  );
}
