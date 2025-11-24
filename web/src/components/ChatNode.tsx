import { memo, useState } from 'react';
import { Handle, Position, NodeProps, Node } from '@xyflow/react';
import { MessageSquare, Bot, Send, Loader2, ChevronDown, ChevronUp } from 'lucide-react';
import { MarkdownRenderer } from './MarkdownRenderer';

type ChatNodeData = Node<{
  prompt: string;
  response: string;
}>;

const ChatNode = ({ data, isConnectable }: NodeProps<ChatNodeData>) => {
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [prompt, setPrompt] = useState(data.prompt);
  const [response, setResponse] = useState(data.response);
  const [isExpanded, setIsExpanded] = useState(false);
  
  // If this is a new node (placeholder text), allow editing
  const isNew = prompt === "New Research Thread...";

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;

    setIsLoading(true);
    setPrompt(input); // Lock in the prompt
    setResponse("Thinking...");

    try {
      const res = await fetch('http://localhost:8000/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: input }),
      });
      
      if (!res.ok) throw new Error('Failed to fetch');
      
      const json = await res.json();
      setResponse(json.response);
    } catch (error) {
      setResponse("Error: Could not connect to mentor agent.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="group relative flex w-[600px] flex-col overflow-hidden rounded-2xl border border-white/10 bg-slate-950/60 shadow-2xl backdrop-blur-2xl transition-all hover:border-indigo-500/30">
      {/* Input Handle */}
      <Handle
        type="target"
        position={Position.Top}
        isConnectable={isConnectable}
        className="!h-3 !w-3 !bg-slate-500 transition-colors group-hover:!bg-blue-500"
      />

      {/* User Prompt Section - Header */}
      <div className="flex gap-3 border-b border-white/5 bg-slate-900/50 p-4">
        <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-slate-800/80 text-slate-400 ring-1 ring-white/5">
          <MessageSquare size={14} />
        </div>
        
        {isNew && !isLoading ? (
          <form onSubmit={handleSubmit} className="flex w-full gap-2">
            <input
              autoFocus
              className="w-full bg-transparent text-sm text-slate-200 placeholder-slate-500 outline-none"
              placeholder="Ask a research question..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
            />
            <button 
                type="submit" 
                disabled={!input.trim()}
                className="text-slate-500 hover:text-blue-400 disabled:opacity-50 transition-colors"
            >
                <Send size={14} />
            </button>
          </form>
        ) : (
          <div className="text-sm font-medium leading-relaxed text-slate-200">
            {prompt}
          </div>
        )}
      </div>

      {/* AI Response Section - Body */}
      <div className="relative flex gap-4 bg-gradient-to-b from-transparent to-slate-900/20 p-5">
        <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-indigo-500/10 text-indigo-400 shadow-[0_0_15px_rgba(99,102,241,0.15)]">
            {isLoading ? <Loader2 size={16} className="animate-spin" /> : <Bot size={16} />}
        </div>
        
        <div className="min-w-0 flex-1">
            <div className={`
                relative transition-all duration-300 ease-in-out
                ${isExpanded ? 'max-h-none' : 'max-h-[400px] overflow-hidden'}
            `}>
                <MarkdownRenderer content={response} />
                
                {/* Fade out gradient if collapsed and content is long */}
                {!isExpanded && response.length > 500 && (
                     <div className="absolute bottom-0 left-0 right-0 h-24 bg-gradient-to-t from-slate-950/90 to-transparent pointer-events-none" />
                )}
            </div>

            {/* Expand/Collapse Toggle */}
            {response.length > 500 && (
                <button 
                    onClick={() => setIsExpanded(!isExpanded)}
                    className="mt-3 flex items-center gap-1 text-xs font-medium text-indigo-400 hover:text-indigo-300 transition-colors"
                >
                    {isExpanded ? (
                        <>Show less <ChevronUp size={12} /></>
                    ) : (
                        <>Read full response <ChevronDown size={12} /></>
                    )}
                </button>
            )}
        </div>
      </div>

      {/* Output Handle */}
      <Handle
        type="source"
        position={Position.Bottom}
        isConnectable={isConnectable}
        className="!h-3 !w-3 !bg-slate-500 transition-colors group-hover:!bg-blue-500"
      />
    </div>
  );
};

export default memo(ChatNode);
