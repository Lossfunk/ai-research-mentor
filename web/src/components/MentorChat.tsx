import { useState, useRef, useEffect } from 'react';
import { Rnd } from 'react-rnd';
import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";
import { useEditor, EditorContent } from '@tiptap/react';
import StarterKit from '@tiptap/starter-kit';
import Placeholder from '@tiptap/extension-placeholder';
import {
  X,
  Send,
  Sparkles,
  Bot,
  User,
  ChevronRight,
  ChevronDown,
  PanelRightOpen,
  SidebarClose,
  Maximize2,
  Minimize2,
  GripHorizontal,
  ImagePlus,
  Terminal,
  Loader2,
  CheckCircle2,
  AlertCircle,
  Activity,
  Settings,
  Download,
  Bold,
  Italic,
  List,
  MoreVertical,
  Globe,
  BookOpen,
} from 'lucide-react';
import { MarkdownRenderer } from './MarkdownRenderer';
import { useChatStore, ChatAttachment, ToolCall } from '@/store/useChatStore';
import { useDocumentStore } from '@/store/useDocumentStore';

// Utility for tailwind class merging
function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

// --- Components ---

const ExportMenu = ({ 
  isOpen, 
  onClose, 
  onExport 
}: { 
  isOpen: boolean; 
  onClose: () => void; 
  onExport: (format: 'txt' | 'json') => void 
}) => {
  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        onClose();
      }
    };
    if (isOpen) document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  return (
    <div 
      ref={menuRef}
      className="absolute top-full right-0 mt-1 w-48 bg-white border border-stone-200 rounded-md shadow-lg z-50 py-1 text-sm font-mono"
    >
      <button 
        onClick={() => onExport('txt')}
        className="w-full text-left px-4 py-2 hover:bg-stone-50 text-stone-700"
      >
        Export as Text (.txt)
      </button>
      <button 
        onClick={() => onExport('json')}
        className="w-full text-left px-4 py-2 hover:bg-stone-50 text-stone-700"
      >
        Export as JSON (.json)
      </button>
    </div>
  );
};

// --- Existing Components (ToolCallBlock, ThinkingBlock, etc) ---

const TOOL_STATUS_CONFIG = {
  calling: {
    icon: Activity,
    color: "text-[#E69F00]", // Okabe-Ito Orange
    borderColor: "border-[#E69F00]",
    bgColor: "bg-[#E69F00]/5",
    iconClass: "animate-pulse"
  },
  executing: {
    icon: Loader2,
    color: "text-[#56B4E9]", // Okabe-Ito Sky Blue
    borderColor: "border-[#56B4E9]",
    bgColor: "bg-[#56B4E9]/5",
    iconClass: "animate-spin"
  },
  completed: {
    icon: CheckCircle2,
    color: "text-[#009E73]", // Okabe-Ito Bluish Green
    borderColor: "border-[#009E73]",
    bgColor: "bg-[#009E73]/5",
    iconClass: ""
  },
  error: {
    icon: AlertCircle,
    color: "text-[#D55E00]", // Okabe-Ito Vermilion
    borderColor: "border-[#D55E00]",
    bgColor: "bg-[#D55E00]/5",
    iconClass: ""
  },
  default: {
    icon: Settings,
    color: "text-stone-500",
    borderColor: "border-stone-300",
    bgColor: "bg-stone-50",
    iconClass: ""
  }
};

const ToolCallBlock = ({ toolCall }: { toolCall: ToolCall }) => {
  const [isExpanded, setIsExpanded] = useState(false);
  
  const statusConfig = TOOL_STATUS_CONFIG[toolCall.status as keyof typeof TOOL_STATUS_CONFIG] || TOOL_STATUS_CONFIG.default;
  const Icon = statusConfig.icon;
  const hasResult = !!toolCall.result;

  // Ensure we don't expand if there is no content, but allow user to see status
  const handleToggle = () => {
    if (hasResult) setIsExpanded(!isExpanded);
  };

  const kind = toolCall.name.toLowerCase().includes('web') ? 'web' : toolCall.name.toLowerCase().includes('arxiv') ? 'arxiv' : 'default';

  const parsed = parseStructuredResult(toolCall.result);

  return (
    <div className={cn(
      "mb-2 rounded-md border overflow-hidden transition-all duration-300",
      statusConfig.borderColor,
      statusConfig.bgColor,
      statusConfig.color
    )}>
      <button 
        onClick={handleToggle}
        disabled={!hasResult}
        className={cn(
          "w-full flex items-center gap-2 px-3 py-1.5 transition-colors",
          hasResult ? "hover:bg-black/5 cursor-pointer" : "cursor-default"
        )}
      >
        <Icon size={12} className={statusConfig.iconClass} />
        <span className="text-[11px] font-mono font-semibold tracking-wide uppercase truncate">
          {toolCall.name}
        </span>
        <span className="text-[10px] opacity-70 ml-2 uppercase font-mono flex-shrink-0">
          {toolCall.status}
        </span>
        <div className="ml-auto flex-shrink-0">
           {hasResult && (
             <ChevronDown size={12} className={cn("transition-transform", isExpanded && "rotate-180")} />
           )}
        </div>
      </button>
      {isExpanded && hasResult && (
        <div className="p-3 bg-white/60 border-t border-inherit border-opacity-30">
          {(kind === 'web' || kind === 'arxiv') && parsed?.items ? (
            <SearchResults items={parsed.items} kind={kind} />
          ) : (
            <pre className="text-xs font-mono whitespace-pre-wrap text-stone-700">{toolCall.result}</pre>
          )}
        </div>
      )}
    </div>
  );
};

type ParsedSearchResult = {
  title?: string;
  url?: string;
  summary?: string;
  snippet?: string;
};

function parseStructuredResult(raw?: string): { items?: ParsedSearchResult[] } | null {
  if (!raw) return null;
  try {
    const parsed = JSON.parse(raw);
    if (Array.isArray(parsed)) {
      return { items: parsed };
    }
    if (parsed?.results && Array.isArray(parsed.results)) {
      return { items: parsed.results };
    }
  } catch (e) {
    return null;
  }
  return null;
}

const okabeIto = {
  arxiv: {
    accent: "#E69F00",
    badge: "bg-[#E69F00]/10 text-[#E69F00]",
    border: "border-[#E69F00]/60",
  },
  web: {
    accent: "#56B4E9",
    badge: "bg-[#56B4E9]/10 text-[#56B4E9]",
    border: "border-[#56B4E9]/60",
  },
};

const SearchResults = ({ items, kind }: { items: ParsedSearchResult[]; kind: 'web' | 'arxiv' }) => {
  const palette = okabeIto[kind];
  const Icon = kind === 'web' ? Globe : BookOpen;
  return (
    <div className="space-y-2">
      {items.map((item, idx) => (
        <div
          key={idx}
          className={cn(
            "rounded-lg border bg-white p-3 shadow-sm", 
            palette.border
          )}
        >
          <div className="flex items-start gap-2">
            <div className={cn("p-1.5 rounded-md", palette.badge)}>
              <Icon size={14} />
            </div>
            <div className="min-w-0 space-y-1">
              {item.title && (
                <div className="text-sm font-semibold text-stone-800 line-clamp-2">{item.title}</div>
              )}
              {(item.summary || item.snippet) && (
                <div className="text-xs text-stone-600 line-clamp-3">{item.summary || item.snippet}</div>
              )}
              {item.url && (
                <a
                  href={item.url}
                  target="_blank"
                  rel="noreferrer"
                  className="text-[11px] font-mono text-blue-600 hover:underline break-all"
                >
                  {item.url}
                </a>
              )}
            </div>
          </div>
        </div>
      ))}
    </div>
  );
};

const ThinkingBlock = ({ content, defaultExpanded = false }: { content: string; defaultExpanded?: boolean }) => {
  const [isExpanded, setIsExpanded] = useState(defaultExpanded);
  if (!content) return null;

  return (
    <div className="mb-4 rounded-md overflow-hidden border border-stone-200/60 bg-stone-50/50 shadow-sm group/trace">
      <button 
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full flex items-center gap-2 px-3 py-1.5 hover:bg-stone-100/50 transition-colors group"
      >
        <div className="relative">
           <Terminal size={12} className="text-stone-400 group-hover:text-stone-600 transition-colors" />
           {!isExpanded && <span className="absolute top-0 right-0 w-1.5 h-1.5 bg-amber-400 rounded-full animate-pulse" />}
        </div>
        <span className="text-[10px] font-mono text-stone-500 uppercase tracking-wider font-medium">
          System_Trace
        </span>
        <div className="ml-auto flex items-center gap-2">
           <span className="text-[10px] text-stone-400 group-hover:text-stone-500 font-mono opacity-0 group-hover/trace:opacity-100 transition-all">
             {isExpanded ? 'HIDE' : 'VIEW LOGS'}
           </span>
        </div>
      </button>
      {isExpanded && (
        <div className="p-3 text-xs font-mono text-[#A8A29E] bg-[#1C1917] overflow-x-auto leading-relaxed border-t border-stone-200 max-h-[300px] overflow-y-auto custom-scrollbar animate-slide-down">
          {content}
        </div>
      )}
    </div>
  );
};

const CollapsibleMessage = ({ content }: { content: string }) => {
  return (
    <div className="text-[15px] leading-relaxed text-stone-900">
      <MarkdownRenderer content={content} />
    </div>
  );
};

const ImageAttachmentStrip = ({ attachments }: { attachments?: ChatAttachment[] }) => {
  if (!attachments || attachments.length === 0) return null;

  return (
    <div className="mt-3 flex flex-wrap gap-2">
      {attachments.map((att) => (
        <div 
          key={att.id} 
          className="relative w-24 h-24 rounded-md border border-stone-200 overflow-hidden bg-stone-50 shadow-[1px_1px_0px_rgba(0,0,0,0.05)]"
        >
          <img src={att.dataUrl} alt={att.name} className="w-full h-full object-cover" />
          <div className="absolute bottom-0 left-0 right-0 bg-white/85 backdrop-blur-sm px-1 py-0.5 text-[10px] text-stone-600 truncate">
            {att.name}
          </div>
        </div>
      ))}
    </div>
  );
};

export const MentorChat = ({ 
    isOpen, 
    onClose, 
    mode, 
    onToggleMode,
    isFullscreen,
    onToggleFullscreen
}: { 
    isOpen: boolean; 
    onClose: () => void;
    mode: 'floating' | 'docked';
    onToggleMode: () => void;
    isFullscreen?: boolean;
    onToggleFullscreen?: () => void;
}) => {
  const [input, setInput] = useState(""); 
  const [pendingImages, setPendingImages] = useState<ChatAttachment[]>([]);
  const [isMobile, setIsMobile] = useState(false);
  
  // Export Menu State
  const [isExportMenuOpen, setExportMenuOpen] = useState(false);

  const { 
    messages, 
    addUserMessage, 
    addAiMessage, 
    isLoading, 
    setLoading, 
    isStreaming, 
    setStreaming, 
    streamingContent, 
    streamingReasoning,
    streamingToolCalls,
    appendContent,
    appendReasoning,
    addToolCall,
    updateToolCall,
    finalizeStream,
    conversations,
    currentConversationId,
    startNewChat,
    loadConversation,
    deleteConversation,
    buildImageContext,
  } = useChatStore();
  const { getSelectedContent } = useDocumentStore();
  const scrollRef = useRef<HTMLDivElement>(null);
  const imageInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const checkMobile = () => setIsMobile(window.innerWidth < 768);
    checkMobile();
    window.addEventListener('resize', checkMobile);
    return () => window.removeEventListener('resize', checkMobile);
  }, []);

  useEffect(() => {
    if (isOpen) {
      // Dispatch custom event when chat opens (for onboarding tour)
      window.dispatchEvent(new Event('mentor-chat-opened'));
    }
  }, [isOpen]);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, streamingContent, streamingReasoning]);

  // ... existing helpers (parseResponse, readFileAsDataUrl, handleImagesSelected, removePendingImage, buildContentParts, stripThinkingTags) ...
  
  const parseResponse = (fullResponse: string): { thinking?: string, content: string } => {
    const thinkingMatch = fullResponse.match(/<thinking>([\s\S]*?)<\/thinking>/i);
    if (thinkingMatch) {
      const thinking = thinkingMatch[1].trim();
      const content = fullResponse.replace(/<thinking>[\s\S]*?<\/thinking>/i, '').trim();
      return { thinking, content };
    }
    return { content: fullResponse };
  };

  const readFileAsDataUrl = (file: File): Promise<string> => new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = () => reject(new Error('Unable to read file'));
    reader.readAsDataURL(file);
  });

  const handleImagesSelected = async (files: FileList | null) => {
    if (!files || files.length === 0) return;
    const validImages = Array.from(files).filter((file) => file.type.startsWith('image/'));
    if (validImages.length === 0) return;

    const attachments: ChatAttachment[] = [];
    for (const file of validImages) {
      try {
        const dataUrl = await readFileAsDataUrl(file);
        attachments.push({
          id: `img-${Date.now()}-${Math.random().toString(36).slice(2)}`,
          name: file.name,
          type: 'image',
          dataUrl,
          size: file.size,
          uploadedAt: Date.now(),
        });
      } catch (err) {
        console.warn('Image read failed', err);
      }
    }

    if (attachments.length) {
      setPendingImages((prev) => [...prev, ...attachments]);
    }
  };

  const removePendingImage = (id: string) => {
    setPendingImages((prev) => prev.filter((img) => img.id !== id));
  };

  const buildContentParts = (text: string, context: string, images: ChatAttachment[]) => {
    const parts: any[] = [];
    if (context) {
      parts.push({ type: 'text', text: `Context:\n${context}` });
    }
    parts.push({ type: 'text', text });
    images.forEach((img) => {
      parts.push({ 
        type: 'image_url', 
        image_url: { url: img.dataUrl } 
      });
    });
    return parts;
  };

  const stripThinkingTags = (text: string) => text.replace(/<\/?thinking>/gi, '');

  // New Export Handler
  const handleExport = (format: 'txt' | 'json') => {
    const convo = conversations.find(c => c.id === currentConversationId);
    if (!convo) return;

    let content = "";
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const filename = `chat-export-${timestamp}.${format}`;

    if (format === 'json') {
      content = JSON.stringify(convo, null, 2);
    } else {
      content = `Chat Export: ${convo.title}\nDate: ${new Date(convo.updatedAt).toLocaleString()}\n\n`;
      content += convo.messages.map(m => {
        const role = m.role === 'user' ? 'USER' : 'MENTOR';
        let msgText = `[${role}]: ${m.content}\n`;
        if (m.thinking) {
          msgText += `[Thinking]: ${m.thinking}\n`;
        }
        if (m.toolCalls) {
          m.toolCalls.forEach(tc => {
             msgText += `[Tool: ${tc.name} (${tc.status})]: ${tc.result || ''}\n`;
          });
        }
        return msgText;
      }).join('\n-------------------\n\n');
    }

    const blob = new Blob([content], { type: format === 'json' ? 'application/json' : 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    setExportMenuOpen(false);
  };

  // Modified Submit Handler (accepts content arg)
  const handleSubmit = async (e: React.FormEvent | React.KeyboardEvent) => {
    if (e && 'preventDefault' in e) e.preventDefault();
    if (!input.trim()) return;

    const userMsg = input;
    const imagesForMessage = pendingImages;
    setInput(""); // Clear input immediately
    setPendingImages([]); // Clear pending images immediately
    
    addUserMessage(userMsg, imagesForMessage);
    setLoading(true);

    const documentContext = getSelectedContent();
    const imageContext = buildImageContext(imagesForMessage);
    const combinedContext = [documentContext, imageContext ? `Images available this chat:\n${imageContext}` : ""]
      .filter(Boolean)
      .join("\n\n");
    const contentParts = buildContentParts(userMsg, combinedContext, imagesForMessage);

    try {
      const streamRes = await fetch('http://localhost:8000/api/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          prompt: userMsg,
          document_context: combinedContext || undefined,
          content_parts: contentParts,
        }),
      });

      if (!streamRes.ok || !streamRes.body) {
        throw new Error('Streaming unavailable');
      }

      setStreaming(true);
      setLoading(false);
      
      const reader = streamRes.body.getReader();
      const decoder = new TextDecoder();
      let buffer = ""; 
      let isThinking = false; 
      let eventBuffer = ""; 
      
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        
        eventBuffer += decoder.decode(value, { stream: true });
        const lines = eventBuffer.split('\n\n');
        eventBuffer = lines.pop() || ""; 
        
        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          
          try {
            const event = JSON.parse(line.slice(6));
            
            if (event.type === 'tool') {
               if (event.status === 'calling') {
                 addToolCall({
                   id: `tool-${Date.now()}-${Math.random().toString(36).slice(2)}`,
                   name: event.name,
                   status: 'calling'
                 });
               } else if (event.status === 'executing') {
                 updateToolCall(event.name, 'executing');
               } else if (event.status === 'completed') {
                 updateToolCall(event.name, 'completed', event.result);
               } else if (event.status === 'error') {
                 updateToolCall(event.name, 'error', event.result);
               }
            } else if (event.type === 'reasoning' && event.content) {
              appendReasoning(event.content);
            } else if (event.type === 'content' && event.content) {
              // Robust parsing for embedded <thinking> tags
              buffer += event.content;
              
              while (true) {
                if (!isThinking) {
                  const startIdx = buffer.indexOf('<thinking>');
                  if (startIdx !== -1) {
                    if (startIdx > 0) {
                      appendContent(buffer.slice(0, startIdx));
                    }
                    buffer = buffer.slice(startIdx + 10);
                    isThinking = true;
                  } else {
                    const partialMatch = buffer.match(/<(?:t(?:h(?:i(?:n(?:k(?:i(?:n(?:g(?:>)?)?)?)?)?)?)?)?)?$/i);
                    if (partialMatch) {
                      if (partialMatch.index && partialMatch.index > 0) {
                         appendContent(buffer.slice(0, partialMatch.index));
                         buffer = buffer.slice(partialMatch.index);
                      }
                      break;
                    } else {
                      appendContent(buffer);
                      buffer = "";
                      break;
                    }
                  }
                } else {
                  const endIdx = buffer.indexOf('</thinking>');
                  if (endIdx !== -1) {
                    if (endIdx > 0) {
                      appendReasoning(buffer.slice(0, endIdx));
                    }
                    buffer = buffer.slice(endIdx + 11);
                    isThinking = false;
                  } else {
                     const partialMatch = buffer.match(/<\/?(?:t(?:h(?:i(?:n(?:k(?:i(?:n(?:g(?:>)?)?)?)?)?)?)?)?)?$/i);
                     if (partialMatch) {
                        if (partialMatch.index && partialMatch.index > 0) {
                          appendReasoning(buffer.slice(0, partialMatch.index));
                          buffer = buffer.slice(partialMatch.index);
                        }
                        break;
                     } else {
                       appendReasoning(buffer);
                       buffer = "";
                       break;
                     }
                  }
                }
              }
              
            } else if (event.type === 'done') {
              finalizeStream();
            } else if (event.type === 'error') {
              console.error('Stream error:', event.content);
              addAiMessage(`Error: ${event.content}`);
              setStreaming(false);
            }
          } catch (parseErr) {
            console.warn('Failed to parse SSE event:', line);
          }
        }
      }
      
      finalizeStream();
      
    } catch (error) {
      console.error('Streaming failed:', error);
      try {
        // Fallback
        const res = await fetch('http://localhost:8000/api/chat', {
          method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          prompt: userMsg,
          document_context: combinedContext || undefined,
          content_parts: contentParts,
        }),
      });
        
        if (!res.ok) throw new Error('Failed to fetch');
        
        const json = await res.json();
        const explicitThinking = json.reasoning as string | undefined;
        const { thinking: parsedThinking, content } = parseResponse(json.response);
        const thinking = explicitThinking || parsedThinking;

        addAiMessage(content, thinking);
      } catch (fallbackError) {
        addAiMessage("Sorry, I encountered an error connecting to the backend.");
      } finally {
        setStreaming(false);
        setLoading(false);
      }
    }
  };

  if (!isOpen) return null;

  const ChatContent = (
    <div className={`
      h-full w-full bg-white flex flex-col overflow-hidden shadow-[4px_4px_0px_0px_rgba(0,0,0,0.1)] border border-stone-800
      ${isMobile && mode === 'floating' ? 'fixed inset-0 z-[60] rounded-none border-0' : 'rounded-xl'}
    `}>
      {/* Header */}
      <div className={`
        flex flex-col gap-3 p-4 border-b border-stone-200 bg-stone-50/80 backdrop-blur-sm
        ${mode === 'floating' && !isMobile ? 'cursor-move drag-handle' : ''}
      `}>
        <div className="flex items-center justify-between gap-2 min-w-0">
          <div className="flex items-center gap-2.5 font-mono text-stone-900 select-none">
            <div className="bg-stone-900 p-1 rounded-sm">
              <Sparkles size={12} className="text-white" />
            </div>
            <span className="text-sm font-bold tracking-tight uppercase truncate">Research_Mentor</span>
            {mode === 'floating' && !isMobile && <GripHorizontal size={14} className="text-stone-300 ml-1" />}
          </div>
          <div className="flex items-center gap-1.5 flex-shrink-0">
              {mode === 'docked' && onToggleFullscreen && !isMobile && (
                 <button 
                   onClick={onToggleFullscreen}
                   className="p-1.5 text-stone-400 hover:text-stone-900 hover:bg-stone-200/50 rounded transition-colors touch-target h-auto w-auto"
                   title={isFullscreen ? "Exit Fullscreen" : "Fullscreen"}
                 >
                   {isFullscreen ? <Minimize2 size={16} /> : <Maximize2 size={16} />}
                 </button>
              )}
              <button 
                onClick={onClose} 
                className="p-1.5 text-stone-400 hover:text-stone-900 hover:bg-stone-200/50 rounded transition-colors touch-target h-auto w-auto"
              >
                  <X size={18} />
              </button>
          </div>
        </div>

        <div className="flex flex-wrap items-center justify-between gap-2 min-w-0">
          <select
            className="text-xs font-mono bg-white border border-stone-200 rounded px-2 py-1 text-stone-700 focus:outline-none focus:border-stone-400 w-full sm:w-56 md:w-64 truncate"
            value={currentConversationId || ''}
            onChange={(e) => {
              if (!e.target.value) return;
              loadConversation(e.target.value);
            }}
          >
            <option value="" disabled>Select chat…</option>
            {conversations
              .slice()
              .sort((a, b) => b.updatedAt - a.updatedAt)
              .map(convo => (
                <option key={convo.id} value={convo.id}>{convo.title}</option>
              ))}
          </select>
          
          <div className="flex items-center gap-1 relative">
            <button
              onClick={() => startNewChat()}
              className="px-2.5 py-1 text-[11px] font-mono rounded border border-stone-200 text-stone-600 hover:text-stone-900 hover:border-stone-400 transition-colors"
              title="New chat"
            >
              New
            </button>
            
            {/* Export Dropdown */}
            <div className="relative">
              <button
                onClick={() => setExportMenuOpen(!isExportMenuOpen)}
                className={cn(
                  "px-2 py-1 text-[11px] font-mono rounded border transition-colors flex items-center justify-center",
                  isExportMenuOpen 
                    ? "bg-stone-100 text-stone-900 border-stone-400" 
                    : "border-stone-200 text-stone-600 hover:text-stone-900 hover:border-stone-400"
                )}
                title="Export chat"
              >
                <Download size={12} />
              </button>
              <ExportMenu 
                isOpen={isExportMenuOpen} 
                onClose={() => setExportMenuOpen(false)} 
                onExport={handleExport} 
              />
            </div>

            {currentConversationId && (
              <button
                onClick={() => deleteConversation(currentConversationId)}
                className="px-2 py-1 text-[11px] font-mono rounded border border-red-200 text-red-600 hover:text-white hover:bg-red-500 hover:border-red-500 transition-colors"
                title="Delete chat"
              >
                ✕
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Messages */}
      <div 
        data-tour-id="chat-toolcalls"
        className="flex-1 overflow-y-auto p-4 md:p-5 space-y-6 bg-[#FAFAF9]" 
        ref={scrollRef}
      >
        {messages.map((msg, idx) => (
          <div key={idx} className={`flex gap-5 animate-slide-up ${msg.role === 'user' ? 'flex-row-reverse' : ''}`}>
            <div className={`
              w-8 h-8 rounded-sm flex items-center justify-center shrink-0 border
              ${msg.role === 'ai' ? 'bg-white border-stone-800 text-stone-900 shadow-[2px_2px_0px_0px_rgba(0,0,0,1)]' : 'bg-stone-900 border-stone-900 text-white'}
            `}>
              {msg.role === 'ai' ? <Bot size={16} /> : <User size={16} />}
            </div>
            <div className={`flex flex-col max-w-[85%] ${msg.role === 'user' ? 'items-end' : 'items-start'}`}>
                {msg.role === 'ai' && msg.toolCalls && msg.toolCalls.length > 0 && (
                  <div className="w-full mb-2">
                    {msg.toolCalls.map((toolCall, tcIdx) => (
                      <ToolCallBlock key={tcIdx} toolCall={toolCall} />
                    ))}
                  </div>
                )}
                {msg.role === 'ai' && msg.thinking && (
                  <ThinkingBlock content={msg.thinking} defaultExpanded={idx === messages.length - 1} />
                )}
                <div className={`
                  rounded-lg px-5 py-3.5 min-w-0 text-[15px] leading-relaxed border
                  ${msg.role === 'ai' 
                    ? 'bg-white border-stone-200 text-stone-900 shadow-[2px_2px_0px_0px_rgba(0,0,0,0.05)]' 
                    : 'bg-[#E5E5E0] border-stone-200 text-stone-800'
                  }
                `}>
                  {msg.role === 'ai' ? (
                    <CollapsibleMessage content={msg.content} />
                  ) : (
                    msg.content
                  )}
                  {msg.attachments?.length ? (
                    <ImageAttachmentStrip attachments={msg.attachments} />
                  ) : null}
                </div>
            </div>
          </div>
        ))}
        
        {/* Streaming Indicator */}
        {isStreaming && (
          <div className="flex gap-3 animate-slide-up">
             <div className="w-8 h-8 rounded-sm bg-white border border-stone-800 text-stone-900 shadow-[2px_2px_0px_0px_rgba(0,0,0,1)] flex items-center justify-center shrink-0">
                <Bot size={16} className="animate-pulse" />
             </div>
             <div className="flex flex-col max-w-[85%]">
                {streamingToolCalls.length > 0 && (
                  <div className="w-full mb-2">
                    {streamingToolCalls.map((toolCall, tcIdx) => (
                      <ToolCallBlock key={tcIdx} toolCall={toolCall} />
                    ))}
                  </div>
                )}
                <div className="mb-4 rounded-md overflow-hidden border border-stone-800 bg-[#1C1917] shadow-sm">
                  <div className="flex items-center gap-2 px-3 py-1.5 bg-[#292524] border-b border-stone-800/50">
                    <span className="relative flex h-2 w-2">
                      <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-amber-500 opacity-75"></span>
                      <span className="relative inline-flex rounded-full h-2 w-2 bg-amber-500"></span>
                    </span>
                    <span className="text-[10px] font-mono text-amber-500 uppercase tracking-wider font-medium">
                      System_Trace: Live
                    </span>
                  </div>
                  <div className="p-3 text-xs font-mono text-[#A8A29E] overflow-x-auto leading-relaxed max-h-48">
                    {streamingReasoning || (
                      <span className="text-stone-500">Calibrating reasoning channel…</span>
                    )}
                    <span className="inline-block w-1.5 h-3 ml-1 bg-amber-500/50 animate-pulse align-middle"></span>
                  </div>
                </div>
                {streamingContent && (
                  <div className="bg-white border border-stone-200 px-5 py-3.5 rounded-lg text-[15px] leading-relaxed text-stone-900 shadow-[2px_2px_0px_0px_rgba(0,0,0,0.05)] min-w-0">
                    <MarkdownRenderer content={streamingContent} />
                  </div>
                )}
             </div>
          </div>
        )}
        
        {/* Loading Indicator */}
        {isLoading && !isStreaming && (
          <div className="flex gap-3 animate-slide-up">
             <div className="w-8 h-8 rounded-sm bg-white border border-stone-800 text-stone-900 shadow-[2px_2px_0px_0px_rgba(0,0,0,1)] flex items-center justify-center shrink-0">
                <Bot size={16} className="animate-pulse" />
             </div>
             <div className="bg-white border border-stone-200 px-4 py-2 rounded text-xs font-mono text-stone-500 shadow-[2px_2px_0px_0px_rgba(0,0,0,0.05)] flex items-center gap-2">
                <span className="animate-pulse">●</span>
                ESTABLISHING_UPLINK...
             </div>
          </div>
        )}
      </div>

      {/* Input Area */}
      <div className="p-4 bg-white border-t border-stone-200 pb-safe">
        <form onSubmit={handleSubmit} className="relative space-y-2">
          {pendingImages.length > 0 && (
            <div className="flex flex-wrap gap-2">
              {pendingImages.map((img) => (
                <div 
                  key={img.id} 
                  className="relative w-16 h-16 rounded border border-stone-200 overflow-hidden shadow-[1px_1px_0px_rgba(0,0,0,0.08)]"
                >
                  <img src={img.dataUrl} alt={img.name} className="w-full h-full object-cover" />
                  <button
                    type="button"
                    onClick={() => removePendingImage(img.id)}
                    className="absolute -top-1 -right-1 bg-white border border-stone-200 rounded-full p-0.5 text-stone-500 hover:text-stone-800 shadow-sm"
                    title="Remove image"
                  >
                    <X size={10} />
                  </button>
                </div>
              ))}
            </div>
          )}
          <div className="relative flex items-end gap-2 bg-stone-50 border border-stone-200 rounded-xl p-2 shadow-inner focus-within:ring-2 focus-within:ring-stone-400/20 focus-within:border-stone-400 transition-all">
            <button
              type="button"
              onClick={() => imageInputRef.current?.click()}
              className="p-2 text-stone-500 hover:text-stone-800 hover:bg-stone-200/50 rounded-lg transition-colors flex-shrink-0 h-10 w-10 flex items-center justify-center"
              title="Attach image"
            >
              <ImagePlus size={20} />
            </button>
            <textarea
              value={input}
              onChange={(e) => {
                setInput(e.target.value);
                // Auto-grow
                e.target.style.height = 'auto';
                e.target.style.height = Math.min(e.target.scrollHeight, 200) + 'px';
              }}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSubmit(e);
                }
              }}
              placeholder="Ask your research mentor..."
              className="w-full bg-transparent border-none text-[15px] leading-relaxed font-sans outline-none placeholder-stone-400 resize-none py-2 min-h-[40px] max-h-[200px]"
              rows={1}
            />
            <button 
              type="submit"
              disabled={!input.trim() || isLoading || isStreaming}
              className="p-2 bg-stone-900 text-white rounded-lg hover:bg-stone-700 disabled:opacity-50 disabled:hover:bg-stone-900 transition-colors shadow-sm flex-shrink-0 h-10 w-10 flex items-center justify-center mb-[1px]"
            >
              <Send size={16} />
            </button>
            <input
              ref={imageInputRef}
              type="file"
              accept="image/*"
              multiple
              className="hidden"
              onChange={(e) => {
                handleImagesSelected(e.target.files);
                e.target.value = '';
              }}
            />
          </div>
        </form>
      </div>
    </div>
  );

  // If mobile, override floating mode to be full screen fixed
  if (mode === 'floating' && !isMobile) {
    return (
      <Rnd
        default={{
          x: window.innerWidth - 450,
          y: 80,
          width: 400,
          height: 600,
        }}
        minWidth={320}
        minHeight={400}
        bounds="window"
        className="z-50"
        dragHandleClassName="drag-handle"
        enableResizing={{
           top:false, right:false, bottom:true, left:true, 
           topRight:false, bottomRight:true, bottomLeft:true, topLeft:true 
        }}
      >
        {ChatContent}
      </Rnd>
    );
  }

  return <div className={isMobile && mode === 'floating' ? "fixed inset-0 z-[60]" : "h-full w-full"}>{ChatContent}</div>;
};
