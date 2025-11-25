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
  const [isCollapsed, setIsCollapsed] = useState(false);
  const [isDragOver, setIsDragOver] = useState(false);
  const [memoryConnected, setMemoryConnected] = useState<boolean | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Check memory status on mount
  useEffect(() => {
    const checkMemoryStatus = async () => {
      try {
        const res = await fetch('http://localhost:8000/api/memory/status');
        if (res.ok) {
          const data = await res.json();
          setMemoryConnected(data.connected);
        }
      } catch {
        setMemoryConnected(false);
      }
    };
    checkMemoryStatus();
  }, []);
  
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

  const uploadFile = async (file: File) => {
    const fileType = ACCEPTED_TYPES[file.type as keyof typeof ACCEPTED_TYPES];
    if (!fileType) {
      console.warn('Unsupported file type:', file.type);
      return;
    }

    const docId = `doc-${Date.now()}-${Math.random().toString(36).slice(2)}`;
    const newDoc: UploadedDocument = {
      id: docId,
      filename: file.name,
      type: fileType,
      size: file.size,
      uploadedAt: new Date(),
      status: 'uploading',
    };
    
    addDocument(newDoc);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch('http://localhost:8000/api/upload', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Upload failed: ${response.statusText}`);
      }

      const result = await response.json();
      
      updateDocument(docId, {
        status: 'ready',
        content: result.content,
      });
    } catch (error) {
      console.error('Upload error:', error);
      updateDocument(docId, {
        status: 'error',
        error: error instanceof Error ? error.message : 'Upload failed',
      });
    }
  };

  const handleFiles = useCallback(async (files: FileList | File[]) => {
    setUploading(true);
    const fileArray = Array.from(files);
    
    for (const file of fileArray) {
      await uploadFile(file);
    }
    
    setUploading(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    
    if (e.dataTransfer.files.length > 0) {
      handleFiles(e.dataTransfer.files);
    }
  }, [handleFiles]);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  }, []);

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFiles(e.target.files);
      e.target.value = ''; // Reset input
    }
  }, [handleFiles]);

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  const readyDocs = documents.filter(d => d.status === 'ready');
  const hasSelection = selectedDocumentIds.size > 0;
  
  // If onClose is provided, we are in "mobile drawer" mode, so we don't use internal collapse
  const isMobile = !!onClose;
  const effectiveCollapsed = isMobile ? false : isCollapsed;

  return (
    <aside 
      className={`
        relative flex h-full flex-col bg-[#F7F6F3]
        transition-all duration-300 ease-in-out
        ${effectiveCollapsed ? 'w-16' : 'w-full md:w-auto'}
        ${!isMobile && 'border-r border-stone-200/60'}
        ${className}
      `}
    >
      {/* Collapse Toggle (Desktop only) */}
      {!isMobile && (
        <button 
          onClick={() => setIsCollapsed(!isCollapsed)}
          className="absolute -right-3 top-4 z-20 flex h-6 w-6 items-center justify-center rounded-full border border-stone-200 bg-white text-stone-400 shadow-sm hover:text-stone-600 hover:scale-105 transition-all"
        >
          {isCollapsed ? <PanelLeftOpen size={12} /> : <PanelLeftClose size={12} />}
        </button>
      )}

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
      <div className={`p-4 ${effectiveCollapsed ? 'items-center' : ''} flex flex-col transition-all`}>
        <div className={`flex items-center justify-between ${effectiveCollapsed ? 'justify-center mb-4' : 'mb-4'}`}>
          {!effectiveCollapsed && (
            <div className="flex items-center gap-2 text-stone-700 font-medium tracking-tight">
              <div className="w-5 h-5 rounded-lg bg-gradient-to-br from-orange-400 to-red-500 shadow-sm" />
              <span className="text-base">Research OS</span>
            </div>
          )}
          {effectiveCollapsed && (
            <div className="w-6 h-6 rounded-lg bg-gradient-to-br from-orange-400 to-red-500 shadow-sm mb-2" />
          )}
          {!effectiveCollapsed && (
            <button 
              onClick={() => fileInputRef.current?.click()}
              className="p-1.5 hover:bg-stone-200/60 rounded-lg text-stone-400 hover:text-stone-600 transition-colors touch-target"
              title="Upload document"
            >
              <Plus size={16} />
            </button>
          )}
        </div>
        
        {!effectiveCollapsed && (
          <div className="relative group">
            <Search size={14} className="absolute left-3 top-2.5 text-stone-400 group-focus-within:text-stone-600 transition-colors" />
            <input 
              className="w-full rounded-xl bg-white border border-stone-200/60 py-2 pl-9 pr-3 text-sm text-stone-700 placeholder-stone-400 outline-none focus:border-stone-300 focus:ring-2 focus:ring-stone-100 transition-all shadow-sm"
              placeholder="Search documents..."
            />
          </div>
        )}
      </div>

      {/* Tabs */}
      <div className={`flex gap-1 px-3 pb-3 border-b border-stone-200/60 ${effectiveCollapsed ? 'flex-col' : ''}`}>
        <SidebarTab 
          label={effectiveCollapsed ? "" : "Context"} 
          icon={<FolderOpen size={16} />} 
          active={activeTab === 'context'} 
          collapsed={effectiveCollapsed}
          onClick={() => setActiveTab('context')}
        />
        <SidebarTab 
          label={effectiveCollapsed ? "" : "Notes"} 
          icon={<FileText size={16} />} 
          active={activeTab === 'notes'} 
          collapsed={effectiveCollapsed}
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
            {!effectiveCollapsed && documents.length > 0 && (
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
                  isCollapsed={effectiveCollapsed}
                  onToggleSelect={() => toggleDocumentSelection(doc.id)}
                  onRemove={() => removeDocument(doc.id)}
                  formatFileSize={formatFileSize}
                />
              ))}
            </div>

            {/* Empty state / Drop zone */}
            {documents.length === 0 && !effectiveCollapsed && (
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
            {!effectiveCollapsed && "Notes feature coming soon..."}
          </div>
        )}
      </div>

      {/* Footer */}
      {!effectiveCollapsed && (
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
      )}
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
        group flex cursor-pointer items-center gap-3 rounded-lg p-2.5
        hover:bg-white hover:shadow-sm border transition-all duration-200
        ${isSelected ? 'bg-blue-50 border-blue-200' : 'border-transparent hover:border-stone-200/60'}
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
            ? <CheckSquare size={16} className="text-blue-500" /> 
            : <Square size={16} className="text-stone-300 group-hover:text-stone-400" />
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
          <div className="text-sm font-medium text-stone-700 line-clamp-1">
            {doc.filename}
          </div>
          <div className="text-[10px] text-stone-400 flex items-center gap-2 mt-0.5">
            <span className="uppercase bg-stone-100 px-1 rounded">{doc.type}</span>
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
          className="flex-shrink-0 p-1.5 opacity-0 group-hover:opacity-100 hover:bg-red-50 rounded-md text-stone-400 hover:text-red-500 transition-all"
        >
          <Trash2 size={14} />
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
      flex items-center justify-center gap-2 rounded-lg py-2 text-sm font-medium transition-all duration-200 touch-target h-auto
      ${active ? 'bg-white text-stone-800 shadow-sm border border-stone-200/60' : 'text-stone-500 hover:text-stone-700 hover:bg-stone-200/40'}
      ${collapsed ? 'aspect-square w-full' : 'flex-1'}
    `}
    title={collapsed ? label : undefined}
  >
    {icon}
    {!collapsed && label}
  </button>
);
