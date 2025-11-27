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
    <div className="mb-4 pl-3 border-l-2 border-stone-300">
      <button 
        onClick={() => setIsExpanded(!isExpanded)}
        className="flex items-center gap-2 w-full text-xs font-mono text-stone-500 hover:text-stone-800 transition-colors"
      >
        <span className="text-[10px] uppercase tracking-wider font-bold">
          {isExpanded ? '▼ SYSTEM_LOG' : '▶ SYSTEM_LOG'}
        </span>
      </button>
      {isExpanded && (
        <div className="mt-2 text-xs font-mono text-stone-600 whitespace-pre-wrap leading-relaxed bg-stone-50 p-3 rounded border border-stone-100">
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
  const [isMobile, setIsMobile] = useState(false);
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
    const checkMobile = () => setIsMobile(window.innerWidth < 768);
    checkMobile();
    window.addEventListener('resize', checkMobile);
    return () => window.removeEventListener('resize', checkMobile);
  }, []);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, streamingContent, streamingReasoning]);

  const parseResponse = (fullResponse: string): { thinking?: string, content: string } => {
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
      const documentContext = getSelectedContent();
      
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
        
        const lines = buffer.split('\n\n');
        buffer = lines.pop() || ""; 
        
        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          
          try {
            const event = JSON.parse(line.slice(6));
            
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
            console.warn('Failed to parse SSE event:', line);
          }
        }
      }
      
      finalizeStream();
      
    } catch (error) {
      console.error('Streaming failed:', error);
      try {
        const res = await fetch('http://localhost:8000/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ prompt: userMsg }),
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
        flex items-center justify-between p-4 border-b border-stone-200 bg-stone-50/80 backdrop-blur-sm h-14
        ${mode === 'floating' && !isMobile ? 'cursor-move drag-handle' : ''}
      `}>
        <div className="flex items-center gap-2.5 font-mono text-stone-900 select-none">
          <div className="bg-stone-900 p-1 rounded-sm">
            <Sparkles size={12} className="text-white" />
          </div>
          <span className="text-sm font-bold tracking-tight uppercase">Research_Mentor</span>
          {mode === 'floating' && !isMobile && <GripHorizontal size={14} className="text-stone-300 ml-2" />}
        </div>
        <div className="flex items-center gap-1.5">
            {mode === 'docked' && onToggleFullscreen && !isMobile && (
               <button 
                 onClick={onToggleFullscreen}
                 className="p-1.5 text-stone-400 hover:text-stone-900 hover:bg-stone-200/50 rounded transition-colors touch-target h-auto w-auto"
                 title={isFullscreen ? "Exit Fullscreen" : "Fullscreen"}
               >
                 {isFullscreen ? <Minimize2 size={16} /> : <Maximize2 size={16} />}
               </button>
            )}
            {!isMobile && (
              <button 
                  onClick={onToggleMode} 
                  className="p-1.5 text-stone-400 hover:text-stone-900 hover:bg-stone-200/50 rounded transition-colors touch-target h-auto w-auto"
                  title={mode === 'floating' ? "Dock to side" : "Float window"}
              >
                  {mode === 'floating' ? <PanelRightOpen size={16} /> : <SidebarClose size={16} />}
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

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-6 bg-[#FAFAF9]" ref={scrollRef}>
        {messages.map((msg, idx) => (
          <div key={idx} className={`flex gap-3 animate-slide-up ${msg.role === 'user' ? 'flex-row-reverse' : ''}`}>
            <div className={`
              w-8 h-8 rounded-sm flex items-center justify-center shrink-0 border
              ${msg.role === 'ai' ? 'bg-white border-stone-800 text-stone-900 shadow-[2px_2px_0px_0px_rgba(0,0,0,1)]' : 'bg-stone-900 border-stone-900 text-white'}
            `}>
              {msg.role === 'ai' ? <Bot size={16} /> : <User size={16} />}
            </div>
            <div className={`flex flex-col max-w-[85%] ${msg.role === 'user' ? 'items-end' : 'items-start'}`}>
                {msg.role === 'ai' && msg.thinking && (
                  <ThinkingBlock content={msg.thinking} defaultExpanded={idx === messages.length - 1} />
                )}
                <div className={`
                  rounded-lg px-5 py-3.5 min-w-0 text-[15px] leading-relaxed border
                  ${msg.role === 'ai' 
                    ? 'bg-white border-stone-200 text-stone-900 shadow-[2px_2px_0px_0px_rgba(0,0,0,0.05)]' 
                    : 'bg-stone-100 border-stone-200 text-stone-800'
                  }
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
        
        {/* Streaming Indicator */}
        {isStreaming && (streamingReasoning || streamingContent) && (
          <div className="flex gap-3 animate-slide-up">
             <div className="w-8 h-8 rounded-sm bg-white border border-stone-800 text-stone-900 shadow-[2px_2px_0px_0px_rgba(0,0,0,1)] flex items-center justify-center shrink-0">
                <Bot size={16} className="animate-pulse" />
             </div>
             <div className="flex flex-col max-w-[85%]">
                {streamingReasoning && (
                  <div className="mb-4 pl-3 border-l-2 border-amber-400">
                    <div className="flex items-center gap-2 text-xs font-mono text-amber-600 animate-pulse mb-1">
                      <span className="w-2 h-2 bg-amber-500 rounded-full" />
                      PROCESSING...
                    </div>
                    <div className="text-xs font-mono text-stone-600 whitespace-pre-wrap max-h-48 overflow-y-auto leading-relaxed bg-stone-50 p-3 rounded border border-stone-100">
                      {streamingReasoning}
                    </div>
                  </div>
                )}
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
        <form onSubmit={handleSubmit} className="relative">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="> Enter command or query..."
            className="w-full bg-stone-50 border border-stone-200 rounded py-3.5 pl-4 pr-14 text-[14px] font-mono outline-none focus:border-stone-400 focus:bg-white transition-all placeholder-stone-400 shadow-inner"
            onMouseDown={(e) => e.stopPropagation()}
          />
          <button 
            type="submit"
            disabled={!input.trim() || isLoading || isStreaming}
            className="absolute right-2 top-2 p-1.5 bg-stone-900 text-white rounded hover:bg-stone-700 disabled:opacity-50 disabled:hover:bg-stone-900 transition-colors shadow-sm touch-target h-auto w-auto"
          >
            <Send size={14} />
          </button>
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
