import { useMemo } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import {
  User,
  Bot,
  Copy,
  Check,
  FileCode2,
  ChevronDown,
  X,
  Search,
  Brain,
  Loader2,
} from "lucide-react";
import { useState, useRef, useEffect } from "react";
import { cn } from "~/lib/utils";
import type { CodeReference } from "~/lib/types";

interface ChatMessageProps {
  role: "user" | "assistant";
  content: string;
  sources?: CodeReference[];
  isStreaming?: boolean;
  isSearching?: boolean;
  className?: string;
}

/**
 * Parse <think>...</think> blocks from reasoning models.
 * Handles both complete and partial (streaming) think blocks.
 */
function parseThinkingBlocks(
  text: string,
  isStreaming: boolean
): {
  thinking: string | null;
  answer: string;
  isThinking: boolean; // true when still inside an unclosed <think> block
} {
  // Check for unclosed <think> block (streaming)
  const openIdx = text.lastIndexOf("<think>");
  const closeIdx = text.lastIndexOf("</think>");

  if (isStreaming && openIdx !== -1 && (closeIdx === -1 || closeIdx < openIdx)) {
    // We're inside an unclosed think block
    const thinkContent = text.slice(openIdx + 7); // after <think>
    const beforeThink = text.slice(0, openIdx).trim();
    return {
      thinking: thinkContent.trim() || null,
      answer: beforeThink,
      isThinking: true,
    };
  }

  // Complete think blocks — extract all
  const thinkRegex = /<think>([\s\S]*?)<\/think>/gi;
  const thinkParts: string[] = [];
  const answer = text
    .replace(thinkRegex, (_, content) => {
      thinkParts.push(content.trim());
      return "";
    })
    .trim();

  return {
    thinking: thinkParts.length > 0 ? thinkParts.join("\n\n") : null,
    answer: answer || (isStreaming ? "" : text),
    isThinking: false,
  };
}

export default function ChatMessage({
  role,
  content,
  sources,
  isStreaming,
  isSearching,
  className,
}: ChatMessageProps) {
  const isUser = role === "user";
  const { thinking, answer, isThinking } = useMemo(
    () =>
      isUser
        ? { thinking: null, answer: content, isThinking: false }
        : parseThinkingBlocks(content, !!isStreaming),
    [content, isUser, isStreaming]
  );
  const [showThinking, setShowThinking] = useState(false);
  const [sourcesOpen, setSourcesOpen] = useState(false);

  const hasSources = !isUser && sources && sources.length > 0;

  return (
    <div
      className={cn(
        "flex gap-3 animate-fade-in",
        isUser ? "flex-row-reverse" : "flex-row",
        className
      )}
    >
      {/* Avatar */}
      <div
        className={cn(
          "flex items-center justify-center w-8 h-8 rounded-lg shrink-0 mt-1",
          isUser ? "bg-accent/10 text-accent" : "bg-purple/10 text-purple"
        )}
      >
        {isUser ? <User className="w-4 h-4" /> : <Bot className="w-4 h-4" />}
      </div>

      {/* Content */}
      <div
        className={cn(
          "flex flex-col max-w-[85%] min-w-0",
          isUser ? "items-end" : "items-start"
        )}
      >
        {/* Retrieval indicator — shown while searching or when sources arrive during streaming */}
        {!isUser && isSearching && (
          <div className="flex items-center gap-2 px-3 py-2 mb-2 rounded-lg bg-accent/5 border border-accent/10 animate-fade-in">
            <Search className="w-3.5 h-3.5 text-accent animate-pulse" />
            <span className="text-xs text-accent font-medium">
              Searching codebase...
            </span>
          </div>
        )}

        {/* Retrieved files indicator — show file names when sources arrive and streaming */}
        {!isUser && !isSearching && isStreaming && hasSources && !sourcesOpen && (
          <div className="flex items-center gap-2 px-3 py-1.5 mb-2 rounded-lg bg-surface border border-border-subtle animate-fade-in">
            <FileCode2 className="w-3.5 h-3.5 text-accent shrink-0" />
            <div className="flex items-center gap-1.5 flex-wrap min-w-0">
              <span className="text-[11px] text-text-muted shrink-0">Context from</span>
              {sources!.slice(0, 4).map((src, i) => (
                <span
                  key={i}
                  className="inline-flex items-center px-1.5 py-0.5 rounded bg-accent/8 text-[10px] font-mono text-accent/80 truncate max-w-[140px]"
                >
                  {src.file_path.split("/").pop()}
                </span>
              ))}
              {sources!.length > 4 && (
                <span className="text-[10px] text-text-ghost">
                  +{sources!.length - 4} more
                </span>
              )}
            </div>
          </div>
        )}

        {/* Message bubble */}
        <div
          className={cn(
            "rounded-2xl px-4 py-3",
            isUser
              ? "bg-accent/10 border border-accent/15 rounded-tr-md"
              : "bg-surface border border-border-subtle rounded-tl-md"
          )}
        >
          {isUser ? (
            <p className="text-[0.9375rem] leading-relaxed text-text-primary whitespace-pre-wrap">
              {content}
            </p>
          ) : (
            <div className="markdown-body">
              {/* Thinking indicator — shown while model is actively reasoning */}
              {isThinking && (
                <div className="mb-3 animate-fade-in">
                  <div className="flex items-center gap-2 mb-2">
                    <Brain className="w-3.5 h-3.5 text-purple animate-pulse" />
                    <span className="text-xs text-purple font-medium">
                      Thinking
                      <span className="inline-flex ml-0.5">
                        <span className="animate-bounce" style={{ animationDelay: "0ms" }}>.</span>
                        <span className="animate-bounce" style={{ animationDelay: "150ms" }}>.</span>
                        <span className="animate-bounce" style={{ animationDelay: "300ms" }}>.</span>
                      </span>
                    </span>
                  </div>
                  {thinking && (
                    <div className="pl-3 border-l-2 border-purple/20 text-xs text-text-muted leading-relaxed whitespace-pre-wrap max-h-32 overflow-y-auto">
                      {thinking}
                    </div>
                  )}
                </div>
              )}

              {/* Completed thinking block — collapsible */}
              {!isThinking && thinking && (
                <div className="mb-3">
                  <button
                    onClick={() => setShowThinking(!showThinking)}
                    className="flex items-center gap-1.5 text-xs text-text-muted hover:text-text-secondary transition-colors group"
                  >
                    <Brain className="w-3 h-3 text-purple/50 group-hover:text-purple/70" />
                    <span
                      className={cn(
                        "inline-block transition-transform text-[10px]",
                        showThinking ? "rotate-90" : ""
                      )}
                    >
                      &#9654;
                    </span>
                    <span>
                      Thought for a moment
                    </span>
                  </button>
                  {showThinking && (
                    <div className="mt-1.5 pl-3 border-l-2 border-purple/15 text-xs text-text-muted leading-relaxed whitespace-pre-wrap max-h-64 overflow-y-auto animate-fade-in">
                      {thinking}
                    </div>
                  )}
                </div>
              )}

              {/* Main answer */}
              {answer ? (
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  rehypePlugins={[rehypeHighlight]}
                  components={{
                    pre: ({ children, ...props }) => (
                      <PreBlock {...props}>{children}</PreBlock>
                    ),
                  }}
                >
                  {answer}
                </ReactMarkdown>
              ) : (
                // Show spinner if no answer yet and not in thinking mode
                !isThinking && isStreaming && !isSearching && (
                  <div className="flex items-center gap-2 py-1">
                    <Loader2 className="w-3.5 h-3.5 text-purple/50 animate-spin" />
                    <span className="text-xs text-text-muted">Generating response...</span>
                  </div>
                )
              )}

              {isStreaming && answer && (
                <span className="inline-block w-2 h-4 ml-0.5 bg-purple/80 animate-typewriter-blink align-middle" />
              )}
            </div>
          )}
        </div>

        {/* Sources trigger — compact pill below message */}
        {hasSources && !isStreaming && (
          <SourcesDrawer
            sources={sources!}
            open={sourcesOpen}
            onToggle={() => setSourcesOpen(!sourcesOpen)}
          />
        )}
      </div>
    </div>
  );
}

/* --- Sources Drawer --- */

function SourcesDrawer({
  sources,
  open,
  onToggle,
}: {
  sources: CodeReference[];
  open: boolean;
  onToggle: () => void;
}) {
  const top5 = sources.slice(0, 5);
  const drawerRef = useRef<HTMLDivElement>(null);

  // Animate height on open/close
  useEffect(() => {
    const el = drawerRef.current;
    if (!el) return;
    if (open) {
      el.style.maxHeight = el.scrollHeight + "px";
      el.style.opacity = "1";
    } else {
      el.style.maxHeight = "0px";
      el.style.opacity = "0";
    }
  }, [open, sources]);

  return (
    <div className="mt-1.5 w-full">
      {/* Trigger pill */}
      <button
        onClick={onToggle}
        className={cn(
          "inline-flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs transition-all",
          open
            ? "bg-accent/10 text-accent border border-accent/20"
            : "bg-elevated/60 text-text-muted border border-border-subtle hover:text-text-secondary hover:border-border-default hover:bg-elevated"
        )}
      >
        <FileCode2 className="w-3.5 h-3.5" />
        <span className="font-medium">
          {top5.length} source{top5.length !== 1 ? "s" : ""} referenced
        </span>
        <ChevronDown
          className={cn(
            "w-3 h-3 transition-transform duration-200",
            open && "rotate-180"
          )}
        />
      </button>

      {/* Expandable drawer */}
      <div
        ref={drawerRef}
        className="overflow-hidden transition-all duration-300 ease-out"
        style={{ maxHeight: 0, opacity: 0 }}
      >
        <div className="mt-2 rounded-xl border border-border-subtle bg-base overflow-hidden">
          {/* Header */}
          <div className="flex items-center justify-between px-3.5 py-2 bg-surface/50 border-b border-border-subtle">
            <span className="text-[10px] font-medium uppercase tracking-wider text-text-ghost">
              Retrieved Context
            </span>
            <button
              onClick={onToggle}
              className="p-0.5 rounded text-text-ghost hover:text-text-muted transition-colors"
            >
              <X className="w-3 h-3" />
            </button>
          </div>

          {/* Source cards */}
          <div className="divide-y divide-border-subtle">
            {top5.map((src, i) => (
              <SourceCard key={i} source={src} rank={i + 1} />
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

/* --- Source Card --- */

function SourceCard({
  source,
  rank,
}: {
  source: CodeReference;
  rank: number;
}) {
  const [copied, setCopied] = useState(false);
  const [expanded, setExpanded] = useState(false);

  const fileName = source.file_path.split("/").pop() || source.file_path;
  const dirPath = source.file_path.includes("/")
    ? source.file_path.substring(0, source.file_path.lastIndexOf("/"))
    : "";
  const relevance = Math.round(source.relevance_score * 100);
  const snippet = source.code_snippet?.trim();
  const previewLines = snippet ? snippet.split("\n").slice(0, 4) : [];
  const fullLines = snippet ? snippet.split("\n") : [];
  const hasMore = fullLines.length > 4;

  const handleCopy = async () => {
    const text = snippet || `${source.file_path}:${source.line_start}`;
    await navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="group/card px-3.5 py-3 hover:bg-elevated/30 transition-colors">
      {/* File info row */}
      <div className="flex items-center gap-2 mb-2">
        <span className="flex items-center justify-center w-5 h-5 rounded text-[10px] font-mono font-bold bg-accent/8 text-accent/70 shrink-0">
          {rank}
        </span>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-1.5">
            <span className="text-xs font-mono font-medium text-text-primary truncate">
              {fileName}
            </span>
            <span className="text-[10px] font-mono text-text-ghost shrink-0">
              L{source.line_start}–{source.line_end}
            </span>
          </div>
          {dirPath && (
            <span className="text-[10px] text-text-ghost truncate block">
              {dirPath}/
            </span>
          )}
        </div>

        {/* Relevance bar */}
        <div className="flex items-center gap-1.5 shrink-0">
          <div className="w-12 h-1.5 rounded-full bg-void overflow-hidden">
            <div
              className="h-full rounded-full transition-all duration-500"
              style={{
                width: `${relevance}%`,
                background:
                  relevance >= 70
                    ? "var(--color-green)"
                    : relevance >= 40
                      ? "var(--color-accent)"
                      : "var(--color-amber)",
              }}
            />
          </div>
          <span className="text-[10px] font-mono text-text-ghost w-7 text-right">
            {relevance}%
          </span>
        </div>

        {/* Copy button */}
        <button
          onClick={handleCopy}
          className={cn(
            "p-1 rounded transition-all shrink-0",
            copied
              ? "text-green"
              : "text-text-ghost hover:text-text-muted"
          )}
          title={copied ? "Copied!" : "Copy snippet"}
        >
          {copied ? (
            <Check className="w-3 h-3" />
          ) : (
            <Copy className="w-3 h-3" />
          )}
        </button>
      </div>

      {/* Code snippet preview */}
      {snippet && (
        <div className="relative">
          <pre className="text-[11px] font-mono leading-relaxed text-text-secondary bg-void/50 rounded-lg px-3 py-2 overflow-x-auto whitespace-pre">
            {(expanded ? fullLines : previewLines).join("\n")}
          </pre>
          {hasMore && (
            <button
              onClick={() => setExpanded(!expanded)}
              className="mt-1 text-[10px] text-accent/70 hover:text-accent transition-colors font-medium"
            >
              {expanded
                ? "Show less"
                : `+${fullLines.length - 4} more lines`}
            </button>
          )}
        </div>
      )}
    </div>
  );
}

/* --- Code Block with Copy --- */

function PreBlock({
  children,
  ...props
}: React.HTMLAttributes<HTMLPreElement> & { children?: React.ReactNode }) {
  const [copied, setCopied] = useState(false);

  // Extract raw text for clipboard
  const code = useMemo(() => {
    const extractText = (node: React.ReactNode): string => {
      if (typeof node === "string") return node;
      if (Array.isArray(node)) return node.map(extractText).join("");
      if (node && typeof node === "object" && "props" in node) {
        const el = node as React.ReactElement<{ children?: React.ReactNode }>;
        return extractText(el.props.children);
      }
      return "";
    };
    return extractText(children);
  }, [children]);

  // Extract language from className (e.g. "language-python" -> "python")
  const language = useMemo(() => {
    const extractLang = (node: React.ReactNode): string => {
      if (node && typeof node === "object" && "props" in node) {
        const el = node as React.ReactElement<{
          className?: string;
          children?: React.ReactNode;
        }>;
        const cls = el.props.className || "";
        const match = cls.match(/language-(\w+)/);
        if (match) return match[1];
        return extractLang(el.props.children);
      }
      return "";
    };
    return extractLang(children);
  }, [children]);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="relative group/code">
      {language && (
        <div className="flex items-center justify-between px-3 py-1.5 bg-surface border-b border-border-subtle rounded-t-lg -mb-[1px]">
          <span className="text-[10px] font-mono text-text-muted uppercase tracking-wider">
            {language}
          </span>
        </div>
      )}
      <pre
        {...props}
        className={cn(
          props.className,
          language && "!rounded-t-none !border-t-0"
        )}
      >
        {children}
      </pre>
      <button
        onClick={handleCopy}
        className={cn(
          "absolute top-2 right-2 p-1.5 rounded-md border transition-all",
          copied
            ? "bg-green/10 border-green/20 text-green"
            : "bg-elevated/80 border-border-subtle text-text-muted hover:text-text-primary sm:opacity-0 sm:group-hover/code:opacity-100"
        )}
        title="Copy code"
      >
        {copied ? (
          <Check className="w-3.5 h-3.5" />
        ) : (
          <Copy className="w-3.5 h-3.5" />
        )}
      </button>
    </div>
  );
}
