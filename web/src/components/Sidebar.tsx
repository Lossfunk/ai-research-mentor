import { useState } from 'react';
import { Book, FileText, Hash, Search, GripVertical, FolderOpen, Plus } from 'lucide-react';
import { useLibraryStore } from '@/store/useLibraryStore';

export const Sidebar = () => {
  const [activeTab, setActiveTab] = useState<'context' | 'notes'>('context');
  const { papers, threads } = useLibraryStore();

  return (
    <aside className="flex h-screen w-72 flex-col border-r border-stone-200 bg-stone-50">
      {/* Header */}
      <div className="p-4">
        <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2 text-stone-800 font-serif font-semibold">
                <div className="w-3 h-3 rounded-full bg-stone-800" />
                Research OS
            </div>
            <button className="p-1 hover:bg-stone-200 rounded text-stone-500">
                <Plus size={16} />
            </button>
        </div>
        <div className="relative">
          <Search size={14} className="absolute left-3 top-2.5 text-stone-400" />
          <input 
            className="w-full rounded-lg bg-white border border-stone-200 py-2 pl-9 pr-3 text-xs text-stone-700 placeholder-stone-400 outline-none focus:border-stone-400 focus:ring-1 focus:ring-stone-400"
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
            flex flex-1 items-center justify-center gap-2 rounded-md py-1.5 text-xs font-medium transition-all
            ${active ? 'bg-white text-stone-800 shadow-sm border border-stone-200' : 'text-stone-500 hover:text-stone-700 hover:bg-stone-200/50'}
        `}
    >
        {icon}
        {label}
    </button>
);

