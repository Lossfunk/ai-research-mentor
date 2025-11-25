import { create } from 'zustand';

type Message = {
  role: 'user' | 'ai';
  content: string;
  thinking?: string;
};

type ChatState = {
  messages: Message[];
  isLoading: boolean;
  isStreaming: boolean;
  streamingContent: string;
  streamingReasoning: string;
  setLoading: (v: boolean) => void;
  setStreaming: (v: boolean) => void;
  appendContent: (chunk: string) => void;
  appendReasoning: (chunk: string) => void;
  addUserMessage: (content: string) => void;
  addAiMessage: (content: string, thinking?: string) => void;
  finalizeStream: () => void;
  reset: () => void;
};

export const useChatStore = create<ChatState>((set, get) => ({
  messages: [
    { role: 'ai', content: "Hello! I'm your research mentor. How can I help you refine your hypothesis today?" },
  ],
  isLoading: false,
  isStreaming: false,
  streamingContent: "",
  streamingReasoning: "",
  setLoading: (v) => set({ isLoading: v }),
  setStreaming: (v) => set({ 
    isStreaming: v, 
    streamingContent: v ? "" : get().streamingContent,
    streamingReasoning: v ? "" : get().streamingReasoning,
  }),
  appendContent: (chunk) => set((state) => ({ streamingContent: state.streamingContent + chunk })),
  appendReasoning: (chunk) => set((state) => ({ streamingReasoning: state.streamingReasoning + chunk })),
  addUserMessage: (content) => set((state) => ({ messages: [...state.messages, { role: 'user', content }] })),
  addAiMessage: (content, thinking) => set((state) => ({ messages: [...state.messages, { role: 'ai', content, thinking }] })),
  finalizeStream: () => {
    const state = get();
    if (state.streamingContent.trim() || state.streamingReasoning.trim()) {
      let finalContent = state.streamingContent.trim();
      let finalReasoning = state.streamingReasoning.trim();
      
      // Fallback: Parse <thinking> tags from content if no reasoning was streamed
      if (!finalReasoning && finalContent) {
        const thinkMatch = finalContent.match(/<thinking>([\s\S]*?)<\/thinking>/i);
        if (thinkMatch) {
          finalReasoning = thinkMatch[1].trim();
          finalContent = finalContent.replace(/<thinking>[\s\S]*?<\/thinking>/gi, '').trim();
        }
      }
      
      set((s) => ({
        messages: [...s.messages, {
          role: 'ai',
          content: finalContent || "(No content)",
          thinking: finalReasoning || undefined,
        }],
        streamingContent: "",
        streamingReasoning: "",
        isStreaming: false,
      }));
    }
  },
  reset: () => set({
    messages: [
      { role: 'ai', content: "Hello! I'm your research mentor. How can I help you refine your hypothesis today?" },
    ],
    isLoading: false,
    isStreaming: false,
    streamingContent: "",
    streamingReasoning: "",
  }),
}));
