import { useState, useRef, useCallback, useEffect } from 'react';
import { 
  FileText, Search, FolderOpen, Plus, PanelLeftClose, PanelLeftOpen,
  Upload, File, Trash2, CheckSquare, Square, Loader2, AlertCircle, Brain, X
} from 'lucide-react';
import { useDocumentStore, UploadedDocument } from '@/store/useDocumentStore';

const ACCEPTED_TYPES = {
  'application/pdf': 'pdf',
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'docx',
  'text/plain': 'txt',
  'text/markdown': 'md',
} as const;

export const Sidebar = ({ 
  className = '', 
  onClose 
}: { 
  className?: string;
  onClose?: () => void;
}) => {
  const [activeTab, setActiveTab] = useState<'context' | 'notes'>('context');
  const [isDragOver, setIsDragOver] = useState(false);
  // Only relevant for mobile drawer mode
  const isMobile = !!onClose;
  
  const { 
    documents, 
    selectedDocumentIds, 
    isUploading,
    addDocument, 
    updateDocument,
    removeDocument,
    toggleDocumentSelection,
    selectAllDocuments,
    clearSelection,
    setUploading
  } = useDocumentStore();
  // ... rest of logic ...
  return (
    <aside 
      className={`
        relative flex h-full flex-col bg-[#F7F6F3] overflow-visible z-40
        ${isMobile ? 'w-full' : 'w-full'}
        ${!isMobile && 'border-r border-stone-200/60'}
        ${className}
      `}
    >
      {/* Mobile Close Button */}
      {isMobile && (
        <button 
          onClick={onClose}
          className="absolute right-4 top-4 z-20 p-2 rounded-full bg-stone-200/50 text-stone-500 hover:bg-stone-300/50 transition-all touch-target"
        >
          <X size={18} />
        </button>
      )}

      {/* Header */}
      <div className="p-4 flex flex-col transition-all">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2 text-stone-800 font-mono font-semibold tracking-tight">
            <div className="w-4 h-4 rounded-sm bg-stone-900 shadow-sm" />
            <span className="text-sm">RESEARCH_OS</span>
          </div>
          <button 
            onClick={() => fileInputRef.current?.click()}
            className="p-1.5 hover:bg-stone-200/60 rounded text-stone-500 hover:text-stone-800 transition-colors touch-target"
            title="Upload document"
          >
            <Plus size={16} />
          </button>
        </div>
        
        <div className="relative group">
          <Search size={13} className="absolute left-3 top-2.5 text-stone-400 group-focus-within:text-stone-600 transition-colors" />
          <input 
            className="w-full rounded bg-white border border-stone-200 py-1.5 pl-9 pr-3 text-xs font-mono text-stone-700 placeholder-stone-400 outline-none focus:border-stone-400 transition-all shadow-[2px_2px_0px_0px_rgba(0,0,0,0.05)] focus:shadow-[2px_2px_0px_0px_rgba(0,0,0,0.1)]"
            placeholder="SEARCH_FILES..."
          />
        </div>
      </div>

      {/* Tabs */}
      <div className="flex gap-1 px-3 pb-3 border-b border-stone-200/60">
        <SidebarTab 
          label="Context" 
          icon={<FolderOpen size={16} />} 
          active={activeTab === 'context'} 
          collapsed={false}
          onClick={() => setActiveTab('context')}
        />
        <SidebarTab 
          label="Notes" 
          icon={<FileText size={16} />} 
          active={activeTab === 'notes'} 
          collapsed={false}
          onClick={() => setActiveTab('notes')}
        />
      </div>

      {/* Hidden file input */}
      <input
        ref={fileInputRef}
        type="file"
        multiple
        accept=".pdf,.docx,.txt,.md"
        onChange={handleFileSelect}
        className="hidden"
      />

      {/* Content Area */}
      <div 
        className={`flex-1 overflow-y-auto p-3 space-y-2 scrollbar-hide ${isDragOver ? 'bg-blue-50/50' : ''}`}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
      >
        {activeTab === 'context' ? (
          <>
            {/* Selection controls */}
            {documents.length > 0 && (
              <div className="flex items-center justify-between px-1 py-1 mb-2">
                <span className="text-[10px] font-semibold uppercase tracking-wider text-stone-400">
                  Documents ({documents.length})
                </span>
                <div className="flex items-center gap-2">
                  {hasSelection && (
                    <span className="text-[10px] text-stone-500">
                      {selectedDocumentIds.size} selected
                    </span>
                  )}
                  <button
                    onClick={hasSelection ? clearSelection : selectAllDocuments}
                    className="text-[10px] text-blue-600 hover:text-blue-700 font-medium px-1.5 py-0.5 rounded hover:bg-blue-50 transition-colors"
                  >
                    {hasSelection ? 'Clear' : 'Select all'}
                  </button>
                </div>
              </div>
            )}

            {/* Document list */}
            <div className="space-y-1">
              {documents.map(doc => (
                <DocumentItem 
                  key={doc.id}
                  doc={doc}
                  isSelected={selectedDocumentIds.has(doc.id)}
                  isCollapsed={false}
                  onToggleSelect={() => toggleDocumentSelection(doc.id)}
                  onRemove={() => removeDocument(doc.id)}
                  formatFileSize={formatFileSize}
                />
              ))}
            </div>

            {/* Empty state / Drop zone */}
            {documents.length === 0 && (
              <div 
                className={`
                  mt-4 p-6 border-2 border-dashed rounded-xl text-center cursor-pointer
                  transition-all duration-200
                  ${isDragOver 
                    ? 'border-blue-400 bg-blue-50 scale-[1.02]' 
                    : 'border-stone-200 hover:border-stone-300 hover:bg-stone-50'
                  }
                `}
                onClick={() => fileInputRef.current?.click()}
              >
                <Upload className={`w-8 h-8 mx-auto mb-3 transition-colors ${isDragOver ? 'text-blue-500' : 'text-stone-400'}`} />
                <p className="text-sm text-stone-600 font-medium">
                  Drop files here
                </p>
                <p className="text-xs text-stone-400 mt-1">
                  or click to browse
                </p>
              </div>
            )}
          </>
        ) : (
          <div className="p-8 text-center text-sm text-stone-400 italic">
            "Notes feature coming soon..."
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="border-t border-stone-200 p-4 text-xs text-stone-500 bg-stone-100/50">
        <div className="flex justify-between items-center">
          <span className="font-medium">{selectedDocumentIds.size > 0 ? `${selectedDocumentIds.size} active` : 'No active context'}</span>
          <div className="flex items-center gap-3">
            {isUploading && (
              <span className="flex items-center gap-1.5 text-blue-600">
                <Loader2 size={12} className="animate-spin" />
                Uploading
              </span>
            )}
            <span 
              className={`flex items-center gap-1.5 px-2 py-1 rounded-full ${memoryConnected ? 'bg-green-100 text-green-700' : 'bg-stone-200 text-stone-500'}`}
              title={memoryConnected ? 'Supermemory connected' : 'Memory offline'}
            >
              <Brain size={12} />
              {memoryConnected ? 'Online' : 'Offline'}
            </span>
          </div>
        </div>
      </div>
    </aside>
  );
};

const DocumentItem = ({ 
  doc, 
  isSelected, 
  isCollapsed,
  onToggleSelect, 
  onRemove,
  formatFileSize 
}: { 
  doc: UploadedDocument;
  isSelected: boolean;
  isCollapsed: boolean;
  onToggleSelect: () => void;
  onRemove: () => void;
  formatFileSize: (bytes: number) => string;
}) => {
  const statusIcon = {
    uploading: <Loader2 size={16} className="animate-spin text-blue-500" />,
    processing: <Loader2 size={16} className="animate-spin text-amber-500" />,
    ready: <File size={16} />,
    error: <AlertCircle size={16} className="text-red-500" />,
  };

  const typeColors = {
    pdf: 'text-red-500',
    docx: 'text-blue-500',
    txt: 'text-stone-500',
    md: 'text-purple-500',
  };

  return (
    <div 
      className={`
        group flex cursor-pointer items-center gap-3 rounded p-2
        hover:bg-white hover:shadow-[2px_2px_0px_0px_rgba(0,0,0,0.05)] border transition-all duration-200
        ${isSelected ? 'bg-stone-100 border-stone-300 shadow-[2px_2px_0px_0px_rgba(0,0,0,0.1)]' : 'border-transparent hover:border-stone-200'}
        ${isCollapsed ? 'justify-center' : ''}
      `}
      title={isCollapsed ? doc.filename : undefined}
      onClick={doc.status === 'ready' ? onToggleSelect : undefined}
    >
      {/* Selection checkbox */}
      {!isCollapsed && doc.status === 'ready' && (
        <button 
          className="flex-shrink-0 touch-target p-0 h-auto w-auto"
          onClick={(e) => { e.stopPropagation(); onToggleSelect(); }}
        >
          {isSelected 
            ? <CheckSquare size={14} className="text-stone-800" /> 
            : <Square size={14} className="text-stone-300 group-hover:text-stone-400" />
          }
        </button>
      )}
      
      {/* File icon */}
      <div className={`flex-shrink-0 ${typeColors[doc.type]}`}>
        {statusIcon[doc.status]}
      </div>
      
      {/* File info */}
      {!isCollapsed && (
        <div className="min-w-0 flex-1">
          <div className={`text-xs font-mono ${isSelected ? 'text-stone-900 font-semibold' : 'text-stone-600'} line-clamp-1`}>
            {doc.filename}
          </div>
          <div className="text-[10px] text-stone-400 flex items-center gap-2 mt-0.5 font-mono">
            <span className="uppercase bg-stone-100 px-1 rounded text-[9px]">{doc.type}</span>
            <span>{formatFileSize(doc.size)}</span>
            {doc.status === 'error' && (
              <span className="text-red-500">{doc.error}</span>
            )}
          </div>
        </div>
      )}
      
      {/* Delete button */}
      {!isCollapsed && (
        <button 
          onClick={(e) => { e.stopPropagation(); onRemove(); }}
          className="flex-shrink-0 p-1.5 opacity-0 group-hover:opacity-100 hover:bg-red-50 rounded text-stone-400 hover:text-red-500 transition-all"
        >
          <Trash2 size={12} />
        </button>
      )}
    </div>
  );
};

const SidebarTab = ({ label, icon, active, collapsed, onClick }: {
  label: string;
  icon: React.ReactNode;
  active: boolean;
  collapsed: boolean;
  onClick: () => void;
}) => (
  <button 
    onClick={onClick}
    className={`
      flex items-center justify-center gap-2 rounded py-1.5 text-xs font-mono uppercase tracking-wide transition-all duration-200 touch-target h-auto
      ${active ? 'bg-stone-100 text-stone-900 border border-stone-300 shadow-[1px_1px_0px_0px_rgba(0,0,0,0.1)]' : 'text-stone-500 hover:text-stone-700 hover:bg-stone-200/40'}
      ${collapsed ? 'aspect-square w-full' : 'flex-1'}
    `}
    title={collapsed ? label : undefined}
  >
    {icon}
    {!collapsed && label}
  </button>
);
