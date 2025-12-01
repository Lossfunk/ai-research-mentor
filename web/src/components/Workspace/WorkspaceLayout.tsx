"use client";

import { useState } from "react";
import dynamic from "next/dynamic";
import { Sidebar } from "@/components/Sidebar";
import { Notebook } from "@/components/Notebook";
import { MentorChat } from "@/components/MentorChat";
import { PenTool, Layout, Sparkles, Menu } from "lucide-react";
import { ResizableHandle, ResizablePanel, ResizablePanelGroup } from "@/components/ui/resizable";

import { OnboardingTour } from "@/components/OnboardingTour";

// Dynamically import Tldraw with SSR disabled
const Whiteboard = dynamic(() => import("@/components/Whiteboard").then(mod => mod.Whiteboard), { 
  ssr: false,
  loading: () => <div className="flex h-full w-full items-center justify-center text-stone-400">Loading Canvas...</div>
});

export default function WorkspaceLayout() {
  const [view, setView] = useState<'notebook' | 'whiteboard'>('notebook');
  const [isChatOpen, setIsChatOpen] = useState(false);
  const [chatMode, setChatMode] = useState<'floating' | 'docked'>('docked');
  const [isMobileSidebarOpen, setIsMobileSidebarOpen] = useState(false);

  return (
    <div className="h-screen w-screen overflow-hidden bg-[#F7F6F3] flex flex-col md:block font-sans selection:bg-amber-100 selection:text-amber-900 animate-fade-in">
       <OnboardingTour />
       {/* Mobile Header */}
       <div className="md:hidden flex items-center justify-between px-4 py-3 pt-safe z-30 fixed top-0 left-0 right-0 pointer-events-none">
         <button 
           onClick={() => setIsMobileSidebarOpen(true)}
           className="pointer-events-auto p-2.5 bg-white/90 backdrop-blur border border-stone-200 rounded-full text-stone-500 shadow-sm hover:scale-105 transition-all"
         >
           <Menu size={20} />
         </button>
       </div>

       {/* Mobile Content Area */}
       <div className="md:hidden flex-1 relative overflow-hidden h-full pt-safe">
          {view === 'notebook' ? (
              <div className="h-full overflow-y-auto scrollbar-hide pt-28 pb-24">
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
         {/* Activity Rail - Leftmost vertical strip */}
         <div className="w-[60px] bg-[#1c1917] flex flex-col items-center py-6 gap-6 z-50 border-r border-stone-800 shrink-0 animate-slide-in-left">
            {/* Primary Controls (Top) */}
            <div className="flex flex-col gap-4">
               <button 
                 data-tour-id="activity-ask-mentor"
                 onClick={() => setIsChatOpen(!isChatOpen)}
                 className={`p-3 rounded-xl transition-all duration-200 ${isChatOpen ? 'bg-stone-800 text-white shadow-md' : 'text-amber-500 hover:text-stone-300 hover:bg-stone-800/50'}`}
                 title="Ask Mentor"
               >
                 <Sparkles size={20} />
               </button>
               <button 
                 data-tour-id="view-notebook"
                 onClick={() => setView('notebook')}
                 className={`p-3 rounded-xl transition-all duration-200 ${view === 'notebook' ? 'bg-stone-800 text-white shadow-md' : 'text-stone-500 hover:text-stone-300 hover:bg-stone-800/50'}`}
                 title="Write"
               >
                 <PenTool size={20} />
               </button>
               <button 
                 onClick={() => setView('whiteboard')}
                 className={`p-3 rounded-xl transition-all duration-200 ${view === 'whiteboard' ? 'bg-stone-800 text-white shadow-md' : 'text-stone-500 hover:text-stone-300 hover:bg-stone-800/50'}`}
                 title="Canvas"
               >
                 <Layout size={20} />
               </button>
            </div>

            <div className="flex-1" />
         </div>

         <ResizablePanelGroup direction="horizontal" className="h-full w-full animate-fade-in" style={{ animationDelay: '0.2s', animationFillMode: 'backwards' }}>
            
            {/* Sidebar Panel (Files) */}
            <ResizablePanel defaultSize={20} minSize={15} maxSize={25} collapsible={true} collapsedSize={0} className="min-w-0 overflow-hidden bg-[#F7F6F3] border-r border-stone-200">
               <Sidebar />
            </ResizablePanel>
            
            <ResizableHandle withHandle />

            {/* Main Content Panel */}
            <ResizablePanel defaultSize={isChatOpen ? 60 : 80} minSize={30}>
               <div className="flex flex-col h-full relative bg-[#F7F6F3]">
                  {/* View Content */}
                  <div className="flex-1 relative overflow-hidden">
                      {view === 'notebook' ? (
                          <div className="h-full overflow-hidden p-6 md:p-10 w-full">
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
            {isChatOpen && (
               <>
                  <ResizableHandle withHandle />
                  <ResizablePanel defaultSize={30} minSize={20} maxSize={50}>
                      <MentorChat 
                          isOpen={true} 
                          onClose={() => setIsChatOpen(false)} 
                          mode="docked"
                          onToggleMode={() => {}}
                      />
                  </ResizablePanel>
               </>
            )}
         </ResizablePanelGroup>
      </div>
    </div>
  );
}
