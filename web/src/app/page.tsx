"use client";

import { useState } from "react";
import dynamic from "next/dynamic";
import { Sidebar } from "@/components/Sidebar";
import { Notebook } from "@/components/Notebook";
import { MentorChat } from "@/components/MentorChat";
import { PenTool, Layout, Sparkles, PanelRightClose, PanelRightOpen, Menu } from "lucide-react";
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
    <>
      <div className="flex items-center gap-1 p-1 bg-stone-100/50 rounded-xl border border-stone-200/60">
          <button 
              onClick={() => setView('notebook')}
              className={`flex items-center gap-2 px-3 py-2 md:py-1.5 rounded-lg text-sm font-medium transition-all duration-200 ${view === 'notebook' ? 'bg-white text-stone-800 shadow-sm ring-1 ring-black/5' : 'text-stone-500 hover:text-stone-700'}`}
          >
              <PenTool size={16} className="md:w-[14px] md:h-[14px]" />
              <span className="hidden sm:inline">Write</span>
          </button>
          <button 
              onClick={() => setView('whiteboard')}
              className={`flex items-center gap-2 px-3 py-2 md:py-1.5 rounded-lg text-sm font-medium transition-all duration-200 ${view === 'whiteboard' ? 'bg-white text-stone-800 shadow-sm ring-1 ring-black/5' : 'text-stone-500 hover:text-stone-700'}`}
          >
              <Layout size={16} className="md:w-[14px] md:h-[14px]" />
              <span className="hidden sm:inline">Canvas</span>
          </button>
      </div>

      <div className="flex items-center gap-2">
          <button 
              onClick={() => setIsChatOpen(!isChatOpen)}
              className={`flex items-center gap-2 px-4 py-2.5 md:px-3 md:py-1.5 rounded-full text-sm font-medium transition-all shadow-sm border duration-200 ${
                  isChatOpen 
                  ? 'bg-stone-100 border-stone-200 text-stone-800' 
                  : 'bg-stone-900 border-transparent text-white hover:bg-stone-800 hover:shadow-md'
              }`}
          >
              <Sparkles size={16} className={`md:w-[14px] md:h-[14px] ${isChatOpen ? "text-stone-600" : "text-yellow-400"}`} />
              <span className="hidden sm:inline">{isChatOpen ? 'Close Mentor' : 'Ask Mentor'}</span>
              <span className="sm:hidden">{isChatOpen ? 'Close' : 'Mentor'}</span>
          </button>
      </div>
    </>
  );

  return (
    <main className="h-screen w-screen overflow-hidden bg-stone-50 flex flex-col md:block">
       {/* Mobile Header */}
       <div className="md:hidden flex items-center justify-between px-4 py-3 pt-safe border-b border-stone-200/60 bg-white/90 backdrop-blur-lg z-30 sticky top-0">
         <button 
           onClick={() => setIsMobileSidebarOpen(true)}
           className="p-2.5 -ml-2 text-stone-500 hover:bg-stone-100 rounded-xl transition-colors"
         >
           <Menu size={22} />
         </button>
         <ToolbarContent />
       </div>

       {/* Mobile Content Area */}
       <div className="md:hidden flex-1 relative overflow-hidden">
          {view === 'notebook' ? (
              <div className="h-full overflow-y-auto scrollbar-hide">
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
            <ResizablePanel defaultSize={20} minSize={15} maxSize={25} collapsible={true} collapsedSize={4} className="min-w-[50px]">
               <Sidebar />
            </ResizablePanel>
            
            <ResizableHandle withHandle />

            {/* Main Content Panel */}
            <ResizablePanel defaultSize={isChatOpen && chatMode === 'docked' ? 50 : 80} minSize={30}>
               <div className="flex flex-col h-full relative bg-stone-50/50">
                  {/* Desktop Toolbar */}
                  <div className="sticky top-0 z-10 flex items-center justify-between px-6 py-3 border-b border-stone-200/60 bg-white/80 backdrop-blur-md">
                     <ToolbarContent />
                  </div>

                  {/* View Content */}
                  <div className="flex-1 relative overflow-hidden">
                      {view === 'notebook' ? (
                          <div className="h-full overflow-y-auto scrollbar-hide">
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
