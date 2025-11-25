import { useState, useRef, useEffect } from 'react';
import { Rnd } from 'react-rnd';
import {
  X,
  Send,
  Sparkles,
  Bot,
  User,
  ChevronRight,
  ChevronDown,
  ChevronUp,
  PanelRightClose,
  PanelRightOpen,
  SidebarClose,
  Maximize2,
  Minimize2,
  GripHorizontal,
} from 'lucide-react';
import { MarkdownRenderer } from './MarkdownRenderer';
import { useChatStore } from '@/store/useChatStore';
import { useDocumentStore } from '@/store/useDocumentStore';

interface Message {
  role: 'user' | 'ai';
  content: string;
  thinking?: string;
}

const ThinkingBlock = ({ content, defaultExpanded = false }: { content: string; defaultExpanded?: boolean }) => {
  const [isExpanded, setIsExpanded] = useState(defaultExpanded);
  if (!content) return null;

  return (
    <div className="mb-2 rounded-lg border border-amber-200 bg-amber-50/70 overflow-hidden shadow-[0_1px_0_rgba(255,193,7,0.25)]">
      <button 
        onClick={() => setIsExpanded(!isExpanded)}
        className="flex items-center gap-2 w-full px-3 py-2 text-xs font-semibold text-amber-800 hover:bg-amber-100/70 transition-colors"
      >
        {isExpanded ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
        <span className="text-[10px] uppercase tracking-[0.08em] bg-white/70 border border-amber-200 px-2 py-0.5 rounded-full text-amber-700">Thinking</span>
        <span className="text-amber-900">Mentor's scratchpad</span>
      </button>
      {isExpanded && (
        <div className="px-3 py-2 text-xs text-amber-900 border-t border-amber-200 bg-white font-mono whitespace-pre-wrap">
          {content}
        </div>
      )}
    </div>
  );
};

const CollapsibleMessage = ({ content }: { content: string }) => {
  const [expanded, setExpanded] = useState(false);
  const isLong = content && content.length > 900;

  return (
    <div className="text-[15px] leading-relaxed text-stone-900">
      <div className={`relative ${expanded ? '' : 'max-h-64 overflow-hidden'}`}>
        <MarkdownRenderer content={content} />
        {!expanded && isLong && (
          <div className="pointer-events-none absolute inset-x-0 bottom-0 h-16 bg-gradient-to-t from-white via-white/85 to-transparent" />
        )}
      </div>
      {isLong && (
        <button
          onClick={() => setExpanded(!expanded)}
          className="mt-2 inline-flex items-center gap-1 rounded-full bg-stone-100 px-3 py-1 text-[11px] font-semibold text-stone-600 hover:bg-stone-200 transition-colors"
        >
          {expanded ? (
            <>
              <ChevronUp size={12} />
              Collapse
            </>
          ) : (
            <>
              <ChevronDown size={12} />
              Show full reply
            </>
          )}
        </button>
      )}
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
    appendContent,
    appendReasoning,
    finalizeStream,
  } = useChatStore();
  const { getSelectedContent } = useDocumentStore();
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, streamingContent, streamingReasoning]);

  // Robust regex that handles newlines and whitespace attributes in thinking tags
  const parseResponse = (fullResponse: string): { thinking?: string, content: string } => {
    // Match <thinking>...</thinking> across multiple lines, non-greedy
    const thinkingMatch = fullResponse.match(/<thinking>([\s\S]*?)<\/thinking>/i);
    if (thinkingMatch) {
      const thinking = thinkingMatch[1].trim();
      const content = fullResponse.replace(/<thinking>[\s\S]*?<\/thinking>/i, '').trim();
      return { thinking, content };
    }
    return { content: fullResponse };
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMsg = input;
    setInput("");
    addUserMessage(userMsg);
    setLoading(true);

    try {
      // Get document context from selected documents
      const documentContext = getSelectedContent();
      
      // Use SSE streaming endpoint for real-time reasoning + content
      const streamRes = await fetch('http://localhost:8000/api/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          prompt: userMsg,
          document_context: documentContext || undefined,
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
      
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        
        buffer += decoder.decode(value, { stream: true });
        
        // Process SSE events (format: "data: {...}\n\n")
        const lines = buffer.split('\n\n');
        buffer = lines.pop() || ""; // Keep incomplete event in buffer
        
        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          
          try {
            const event = JSON.parse(line.slice(6)); // Remove "data: " prefix
            
            if (event.type === 'reasoning' && event.content) {
              appendReasoning(event.content);
            } else if (event.type === 'content' && event.content) {
              appendContent(event.content);
            } else if (event.type === 'done') {
              finalizeStream();
            } else if (event.type === 'error') {
              console.error('Stream error:', event.content);
              addAiMessage(`Error: ${event.content}`);
              setStreaming(false);
            }
          } catch (parseErr) {
            // Skip malformed events
            console.warn('Failed to parse SSE event:', line);
          }
        }
      }
      
      // Finalize any remaining content
      finalizeStream();
      
    } catch (error) {
      console.error('Streaming failed:', error);
      // Fallback to non-streaming endpoint
      try {
        const res = await fetch('http://localhost:8000/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ prompt: userMsg }),
        });
        
        if (!res.ok) throw new Error('Failed to fetch');
        
        const json = await res.json();
        console.log("Raw LLM Response:", json);

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
    <div className="h-full w-full bg-white flex flex-col overflow-hidden rounded-xl shadow-sm border border-stone-200">
      {/* Header */}
      <div className={`flex items-center justify-between p-4 border-b border-stone-100 bg-stone-50/80 backdrop-blur-sm h-14 ${mode === 'floating' ? 'cursor-move drag-handle' : ''}`}>
        <div className="flex items-center gap-2 font-medium text-stone-700 select-none">
          <Sparkles size={16} className="text-yellow-500" />
          Research Mentor
          {mode === 'floating' && <GripHorizontal size={14} className="text-stone-300 ml-2" />}
        </div>
        <div className="flex items-center gap-1">
            {mode === 'docked' && onToggleFullscreen && (
               <button 
                 onClick={onToggleFullscreen}
                 className="p-1 text-stone-400 hover:text-stone-600 hover:bg-stone-200/50 rounded transition-colors"
                 title={isFullscreen ? "Exit Fullscreen" : "Fullscreen"}
               >
                 {isFullscreen ? <Minimize2 size={16} /> : <Maximize2 size={16} />}
               </button>
            )}
            <button 
                onClick={onToggleMode} 
                className="p-1 text-stone-400 hover:text-stone-600 hover:bg-stone-200/50 rounded transition-colors"
                title={mode === 'floating' ? "Dock to side" : "Float window"}
            >
                {mode === 'floating' ? <PanelRightOpen size={16} /> : <SidebarClose size={16} />}
            </button>
            <button onClick={onClose} className="p-1 text-stone-400 hover:text-stone-600 hover:bg-stone-200/50 rounded transition-colors">
                <X size={18} />
            </button>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-6 bg-stone-50/30" ref={scrollRef}>
        {messages.map((msg, idx) => (
          <div key={idx} className={`flex gap-3 ${msg.role === 'user' ? 'flex-row-reverse' : ''}`}>
            <div className={`
              w-8 h-8 rounded-full flex items-center justify-center shrink-0 shadow-sm
              ${msg.role === 'ai' ? 'bg-white text-indigo-500 border border-stone-100' : 'bg-stone-800 text-white'}
            `}>
              {msg.role === 'ai' ? <Bot size={16} /> : <User size={16} />}
            </div>
            <div className={`flex flex-col max-w-[85%] ${msg.role === 'user' ? 'items-end' : 'items-start'}`}>
                {msg.role === 'ai' && msg.thinking && (
                  <ThinkingBlock content={msg.thinking} defaultExpanded={idx === messages.length - 1} />
                )}
                <div className={`
                  rounded-2xl px-4 py-3 shadow-sm min-w-0
                  ${msg.role === 'ai' ? 'bg-white border border-stone-200 text-stone-900' : 'bg-stone-800 text-white text-sm'}
                `}>
                  {msg.role === 'ai' ? (
                    <CollapsibleMessage content={msg.content} />
                  ) : (
                    msg.content
                  )}
                </div>
            </div>
          </div>
        ))}
        {/* Streaming in-progress bubble with separate reasoning and content */}
        {isStreaming && (streamingReasoning || streamingContent) && (
          <div className="flex gap-3">
             <div className="w-8 h-8 rounded-full bg-white border border-stone-100 flex items-center justify-center shadow-sm shrink-0">
                <Bot size={16} className="text-indigo-500 animate-pulse" />
             </div>
             <div className="flex flex-col max-w-[85%]">
                {/* Live reasoning stream */}
                {streamingReasoning && (
                  <div className="mb-2 rounded-lg border border-amber-200 bg-amber-50/70 overflow-hidden shadow-[0_1px_0_rgba(255,193,7,0.25)]">
                    <div className="flex items-center gap-2 px-3 py-2 text-xs font-semibold text-amber-800">
                      <span className="relative flex h-2 w-2">
                        <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-amber-400 opacity-75" />
                        <span className="relative inline-flex rounded-full h-2 w-2 bg-amber-500" />
                      </span>
                      <span className="text-[10px] uppercase tracking-[0.08em] bg-white/70 border border-amber-200 px-2 py-0.5 rounded-full text-amber-700">Thinking</span>
                      <span className="text-amber-900">Reasoning in progress...</span>
                    </div>
                    <div className="px-3 py-2 text-xs text-amber-900 border-t border-amber-200 bg-white font-mono whitespace-pre-wrap max-h-48 overflow-y-auto">
                      {streamingReasoning}
                    </div>
                  </div>
                )}
                {/* Live content stream */}
                {streamingContent && (
                  <div className="bg-white border border-stone-200 px-4 py-3 rounded-2xl text-[15px] leading-relaxed text-stone-900 shadow-sm min-w-0">
                    <MarkdownRenderer content={streamingContent} />
                  </div>
                )}
             </div>
          </div>
        )}
        {/* Loading indicator when waiting for first token */}
        {isLoading && !isStreaming && (
          <div className="flex gap-3">
             <div className="w-8 h-8 rounded-full bg-white border border-stone-100 flex items-center justify-center shadow-sm">
                <Bot size={16} className="text-indigo-500 animate-pulse" />
             </div>
             <div className="bg-white border border-stone-100 px-4 py-3 rounded-2xl text-sm text-stone-500 shadow-sm flex items-center gap-2">
                <span className="relative flex h-2 w-2">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-amber-400 opacity-75" />
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-amber-500" />
                </span>
                Connecting to mentor...
             </div>
          </div>
        )}
      </div>

      {/* Input */}
      <div className="p-4 bg-white border-t border-stone-100">
        <form onSubmit={handleSubmit} className="relative">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask follow-up questions..."
            className="w-full bg-stone-50 border border-stone-200 rounded-xl py-3 pl-4 pr-12 text-sm outline-none focus:ring-2 focus:ring-stone-200 transition-all placeholder-stone-400 focus:bg-white"
            onMouseDown={(e) => e.stopPropagation()} // Prevent drag when clicking input
          />
          <button 
            type="submit"
            disabled={!input.trim() || isLoading || isStreaming}
            className="absolute right-2 top-2 p-1.5 bg-stone-800 text-white rounded-lg hover:bg-stone-700 disabled:opacity-50 disabled:hover:bg-stone-800 transition-colors shadow-sm"
          >
            <Send size={14} />
          </button>
        </form>
      </div>
    </div>
  );

  if (mode === 'floating') {
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

  return <div className="h-full w-full">{ChatContent}</div>;
};
