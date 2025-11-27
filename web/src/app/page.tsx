"use client";

import { useState } from "react";
import dynamic from "next/dynamic";
import { Sidebar } from "@/components/Sidebar";
import { Notebook } from "@/components/Notebook";
import { MentorChat } from "@/components/MentorChat";
import { PenTool, Layout, Sparkles, Menu } from "lucide-react";
import { ResizableHandle, ResizablePanel, ResizablePanelGroup } from "@/components/ui/resizable";

// Dynamically import Tldraw with SSR disabled
const Whiteboard = dynamic(() => import("@/components/Whiteboard").then(mod => mod.Whiteboard), { 
  ssr: false,
  loading: () => <div className="flex h-full w-full items-center justify-center text-stone-400">Loading Canvas...</div>
});

export default function Home() {
  const [view, setView] = useState<'notebook' | 'whiteboard'>('notebook');
  const [isChatOpen, setIsChatOpen] = useState(false);
  const [chatMode, setChatMode] = useState<'floating' | 'docked'>('floating');
  const [isMobileSidebarOpen, setIsMobileSidebarOpen] = useState(false);

  const ToolbarContent = () => (
    <div className="flex items-center gap-3 p-2 bg-white/90 backdrop-blur-xl rounded-full border border-stone-200 shadow-[0_8px_30px_rgb(0,0,0,0.12)]">
      <div className="flex items-center gap-1 p-1 bg-stone-100/50 rounded-full border border-stone-200/50">
          <button 
              onClick={() => setView('notebook')}
              className={`flex items-center gap-2 px-4 py-2.5 rounded-full text-sm font-mono transition-all duration-300 ${view === 'notebook' ? 'bg-stone-900 text-white shadow-md scale-105' : 'text-stone-500 hover:text-stone-900 hover:bg-stone-200/50'}`}
          >
              <PenTool size={14} />
              <span className="hidden sm:inline">WRITE</span>
          </button>
          <button 
              onClick={() => setView('whiteboard')}
              className={`flex items-center gap-2 px-4 py-2.5 rounded-full text-sm font-mono transition-all duration-300 ${view === 'whiteboard' ? 'bg-stone-900 text-white shadow-md scale-105' : 'text-stone-500 hover:text-stone-900 hover:bg-stone-200/50'}`}
          >
              <Layout size={14} />
              <span className="hidden sm:inline">CANVAS</span>
          </button>
      </div>

      <div className="h-6 w-px bg-stone-200" />

      <button 
          onClick={() => setIsChatOpen(!isChatOpen)}
          className={`flex items-center gap-2 px-4 py-2.5 rounded-full text-sm font-mono transition-all duration-300 border ${
              isChatOpen 
              ? 'bg-stone-100 border-stone-300 text-stone-900' 
              : 'bg-white border-stone-200 text-stone-600 hover:border-stone-400 hover:text-stone-900 shadow-sm'
          }`}
      >
          <Sparkles size={14} className={isChatOpen ? "text-stone-900" : "text-amber-500"} />
          <span className="hidden sm:inline">{isChatOpen ? 'CLOSE_MENTOR' : 'ASK_MENTOR'}</span>
          <span className="sm:hidden">{isChatOpen ? 'CLOSE' : 'MENTOR'}</span>
      </button>
    </div>
  );

  return (
    <main className="h-screen w-screen overflow-hidden bg-[#F7F6F3] flex flex-col md:block font-sans selection:bg-amber-100 selection:text-amber-900">
       {/* Mobile Header */}
       <div className="md:hidden flex items-center justify-between px-4 py-3 pt-safe z-30 fixed top-0 left-0 right-0 pointer-events-none">
         <button 
           onClick={() => setIsMobileSidebarOpen(true)}
           className="pointer-events-auto p-2.5 bg-white/90 backdrop-blur border border-stone-200 rounded-full text-stone-500 shadow-sm hover:scale-105 transition-all"
         >
           <Menu size={20} />
         </button>
       </div>

       {/* Floating Dock (Desktop & Mobile) */}
       <div className="fixed bottom-8 left-1/2 -translate-x-1/2 z-40 animate-slide-up">
          <ToolbarContent />
       </div>

       {/* Mobile Content Area */}
       <div className="md:hidden flex-1 relative overflow-hidden h-full pt-safe">
          {view === 'notebook' ? (
              <div className="h-full overflow-y-auto scrollbar-hide pt-12 pb-24">
                  <Notebook />
              </div>
          ) : (
              <div className="h-full w-full bg-white">
                  <Whiteboard />
              </div>
          )}
       </div>

       {/* Mobile Sidebar Drawer */}
       {isMobileSidebarOpen && (
         <>
           <div 
             className="mobile-drawer-overlay" 
             onClick={() => setIsMobileSidebarOpen(false)}
           />
           <Sidebar 
             className="mobile-drawer" 
             onClose={() => setIsMobileSidebarOpen(false)}
           />
         </>
       )}

       {/* Desktop Layout */}
       <div className="hidden md:flex h-full w-full">
         <ResizablePanelGroup direction="horizontal" className="h-full w-full">
            
            {/* Sidebar Panel */}
            <ResizablePanel defaultSize={20} minSize={15} maxSize={25} collapsible={true} collapsedSize={4} className="min-w-[50px] overflow-visible">
               <Sidebar />
            </ResizablePanel>
            
            <ResizableHandle withHandle />

            {/* Main Content Panel */}
            <ResizablePanel defaultSize={isChatOpen && chatMode === 'docked' ? 50 : 80} minSize={30}>
               <div className="flex flex-col h-full relative bg-[#F7F6F3]">
                  {/* View Content */}
                  <div className="flex-1 relative overflow-hidden">
                      {view === 'notebook' ? (
                          <div className="h-full overflow-y-auto scrollbar-hide p-8 pb-24 max-w-3xl mx-auto">
                              <Notebook />
                          </div>
                      ) : (
                          <div className="h-full w-full bg-white">
                              <Whiteboard />
                          </div>
                      )}
                  </div>
               </div>
            </ResizablePanel>

            {/* Docked Mentor Panel */}
            {isChatOpen && chatMode === 'docked' && (
               <>
                  <ResizableHandle withHandle />
                  <ResizablePanel defaultSize={30} minSize={20} maxSize={50}>
                      <MentorChat 
                          isOpen={true} 
                          onClose={() => setIsChatOpen(false)} 
                          mode="docked"
                          onToggleMode={() => setChatMode('floating')}
                      />
                  </ResizablePanel>
               </>
            )}
         </ResizablePanelGroup>
       </div>

       {/* Floating Mentor Chat (Desktop & Mobile Overlay) */}
       {isChatOpen && chatMode === 'floating' && (
           <MentorChat 
             isOpen={true} 
             onClose={() => setIsChatOpen(false)} 
             mode="floating"
             onToggleMode={() => setChatMode('docked')}
           />
       )}
    </main>
  );
}
