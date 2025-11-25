import { useState } from 'react';
import { Book, FileText, Hash, Search, GripVertical, FolderOpen, Plus } from 'lucide-react';
import { useLibraryStore } from '@/store/useLibraryStore';

export const Sidebar = () => {
  const [activeTab, setActiveTab] = useState<'context' | 'notes'>('context');
  const { papers, threads } = useLibraryStore();

  return (
    <aside className="flex h-screen w-72 flex-col border-r border-stone-200/60 bg-[#F7F6F3]">
      {/* Header */}
      <div className="p-4">
        <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2 text-stone-700 font-medium tracking-tight">
                <div className="w-4 h-4 rounded-md bg-gradient-to-br from-orange-400 to-red-500 shadow-sm" />
                Research OS
            </div>
            <button className="p-1 hover:bg-stone-200/60 rounded text-stone-400 hover:text-stone-600 transition-colors">
                <Plus size={14} />
            </button>
        </div>
        <div className="relative group">
          <Search size={13} className="absolute left-2.5 top-2.5 text-stone-400 group-focus-within:text-stone-600 transition-colors" />
          <input 
            className="w-full rounded-md bg-white border border-stone-200/60 py-1.5 pl-8 pr-3 text-xs text-stone-700 placeholder-stone-400 outline-none focus:border-stone-300 focus:ring-2 focus:ring-stone-100 transition-all shadow-sm"
            placeholder="Search knowledge..."
          />
        </div>
      </div>

      {/* Tabs */}
      <div className="flex gap-1 px-2 pb-2 border-b border-stone-200">
        <SidebarTab 
            label="Context" 
            icon={<FolderOpen size={14} />} 
            active={activeTab === 'context'} 
            onClick={() => setActiveTab('context')}
        />
        <SidebarTab 
            label="Notes" 
            icon={<FileText size={14} />} 
            active={activeTab === 'notes'} 
            onClick={() => setActiveTab('notes')}
        />
      </div>

      {/* List Content */}
      <div className="flex-1 overflow-y-auto p-2 space-y-1">
        {activeTab === 'context' ? (
            <>
                <div className="px-2 py-1 text-[10px] font-semibold uppercase tracking-wider text-stone-400">Papers</div>
                {papers.map(paper => (
                    <div 
                        key={paper.id}
                        className="group flex cursor-pointer items-start gap-3 rounded-md p-2 hover:bg-white hover:shadow-sm border border-transparent hover:border-stone-200 transition-all"
                    >
                        <div className="mt-1 text-stone-400"><Book size={14} /></div>
                        <div>
                            <div className="text-sm font-medium text-stone-700 line-clamp-1">{paper.title}</div>
                            <div className="text-xs text-stone-500">{paper.authors}</div>
                        </div>
                    </div>
                ))}
            </>
        ) : (
            <div className="p-4 text-center text-xs text-stone-400">
                No notes yet.
            </div>
        )}
      </div>

      {/* Footer */}
      <div className="border-t border-stone-200 p-3 text-xs text-stone-500 flex justify-between bg-stone-100/50">
         <span>Local Storage</span>
         <span>Synced</span>
      </div>
    </aside>
  );
};

const SidebarTab = ({ label, icon, active, onClick }: any) => (
    <button 
        onClick={onClick}
        className={`
            flex flex-1 items-center justify-center gap-2 rounded-md py-1.5 text-xs font-medium transition-all duration-200
            ${active ? 'bg-white text-stone-800 shadow-[0_1px_2px_rgba(0,0,0,0.04)] border border-stone-200/60' : 'text-stone-500 hover:text-stone-700 hover:bg-stone-200/40'}
        `}
    >
        {icon}
        {label}
    </button>
);

