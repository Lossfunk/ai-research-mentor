import { useState, useRef, useCallback, useEffect } from 'react';
import { 
  FileText, Search, FolderOpen, Plus, PanelLeftClose, PanelLeftOpen,
  Upload, File, Trash2, CheckSquare, Square, Loader2, AlertCircle, Brain
} from 'lucide-react';
import { useDocumentStore, UploadedDocument } from '@/store/useDocumentStore';

const ACCEPTED_TYPES = {
  'application/pdf': 'pdf',
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'docx',
  'text/plain': 'txt',
  'text/markdown': 'md',
} as const;

export const Sidebar = () => {
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

  return (
    <aside 
      className={`
        relative flex h-screen flex-col border-r border-stone-200/60 bg-[#F7F6F3]
        transition-all duration-300 ease-in-out
        ${isCollapsed ? 'w-16' : 'w-72'}
      `}
    >
      {/* Collapse Toggle */}
      <button 
        onClick={() => setIsCollapsed(!isCollapsed)}
        className="absolute -right-3 top-4 z-20 flex h-6 w-6 items-center justify-center rounded-full border border-stone-200 bg-white text-stone-400 shadow-sm hover:text-stone-600 hover:scale-105 transition-all"
      >
        {isCollapsed ? <PanelLeftOpen size={12} /> : <PanelLeftClose size={12} />}
      </button>

      {/* Header */}
      <div className={`p-4 ${isCollapsed ? 'items-center' : ''} flex flex-col transition-all`}>
        <div className={`flex items-center justify-between ${isCollapsed ? 'justify-center mb-4' : 'mb-4'}`}>
          {!isCollapsed && (
            <div className="flex items-center gap-2 text-stone-700 font-medium tracking-tight">
              <div className="w-4 h-4 rounded-md bg-gradient-to-br from-orange-400 to-red-500 shadow-sm" />
              Research OS
            </div>
          )}
          {isCollapsed && (
            <div className="w-6 h-6 rounded-md bg-gradient-to-br from-orange-400 to-red-500 shadow-sm mb-2" />
          )}
          {!isCollapsed && (
            <button 
              onClick={() => fileInputRef.current?.click()}
              className="p-1 hover:bg-stone-200/60 rounded text-stone-400 hover:text-stone-600 transition-colors"
              title="Upload document"
            >
              <Plus size={14} />
            </button>
          )}
        </div>
        
        {!isCollapsed && (
          <div className="relative group">
            <Search size={13} className="absolute left-2.5 top-2.5 text-stone-400 group-focus-within:text-stone-600 transition-colors" />
            <input 
              className="w-full rounded-md bg-white border border-stone-200/60 py-1.5 pl-8 pr-3 text-xs text-stone-700 placeholder-stone-400 outline-none focus:border-stone-300 focus:ring-2 focus:ring-stone-100 transition-all shadow-sm"
              placeholder="Search documents..."
            />
          </div>
        )}
      </div>

      {/* Tabs */}
      <div className={`flex gap-1 px-2 pb-2 border-b border-stone-200/60 ${isCollapsed ? 'flex-col' : ''}`}>
        <SidebarTab 
          label={isCollapsed ? "" : "Context"} 
          icon={<FolderOpen size={16} />} 
          active={activeTab === 'context'} 
          collapsed={isCollapsed}
          onClick={() => setActiveTab('context')}
        />
        <SidebarTab 
          label={isCollapsed ? "" : "Notes"} 
          icon={<FileText size={16} />} 
          active={activeTab === 'notes'} 
          collapsed={isCollapsed}
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
        className={`flex-1 overflow-y-auto p-2 space-y-1 scrollbar-hide ${isDragOver ? 'bg-blue-50' : ''}`}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
      >
        {activeTab === 'context' ? (
          <>
            {/* Selection controls */}
            {!isCollapsed && documents.length > 0 && (
              <div className="flex items-center justify-between px-2 py-1 mb-2">
                <span className="text-[10px] font-semibold uppercase tracking-wider text-stone-400">
                  Documents ({documents.length})
                </span>
                <div className="flex items-center gap-1">
                  {hasSelection && (
                    <span className="text-[10px] text-stone-500 mr-1">
                      {selectedDocumentIds.size} selected
                    </span>
                  )}
                  <button
                    onClick={hasSelection ? clearSelection : selectAllDocuments}
                    className="text-[10px] text-blue-600 hover:text-blue-700"
                  >
                    {hasSelection ? 'Clear' : 'Select all'}
                  </button>
                </div>
              </div>
            )}

            {/* Document list */}
            {documents.map(doc => (
              <DocumentItem 
                key={doc.id}
                doc={doc}
                isSelected={selectedDocumentIds.has(doc.id)}
                isCollapsed={isCollapsed}
                onToggleSelect={() => toggleDocumentSelection(doc.id)}
                onRemove={() => removeDocument(doc.id)}
                formatFileSize={formatFileSize}
              />
            ))}

            {/* Empty state / Drop zone */}
            {documents.length === 0 && !isCollapsed && (
              <div 
                className={`
                  mt-4 p-6 border-2 border-dashed rounded-lg text-center cursor-pointer
                  transition-colors
                  ${isDragOver 
                    ? 'border-blue-400 bg-blue-50' 
                    : 'border-stone-200 hover:border-stone-300 hover:bg-stone-50'
                  }
                `}
                onClick={() => fileInputRef.current?.click()}
              >
                <Upload className="w-8 h-8 mx-auto text-stone-400 mb-2" />
                <p className="text-sm text-stone-600 font-medium">
                  Drop files here
                </p>
                <p className="text-xs text-stone-400 mt-1">
                  or click to browse
                </p>
                <p className="text-[10px] text-stone-400 mt-2">
                  PDF, DOCX, TXT, MD
                </p>
              </div>
            )}

            {/* Upload indicator when dragging */}
            {isDragOver && documents.length > 0 && !isCollapsed && (
              <div className="p-4 border-2 border-dashed border-blue-400 rounded-lg bg-blue-50 text-center">
                <Upload className="w-6 h-6 mx-auto text-blue-500 mb-1" />
                <p className="text-xs text-blue-600">Drop to upload</p>
              </div>
            )}
          </>
        ) : (
          <div className="p-4 text-center text-xs text-stone-400">
            {!isCollapsed && "Notes coming soon..."}
          </div>
        )}
      </div>

      {/* Footer with context and memory indicator */}
      {!isCollapsed && (
        <div className="border-t border-stone-200 p-3 text-xs text-stone-500 bg-stone-100/50">
          <div className="flex justify-between items-center">
            <span>{selectedDocumentIds.size > 0 ? `${selectedDocumentIds.size} in context` : 'No context'}</span>
            <div className="flex items-center gap-2">
              {isUploading && (
                <span className="flex items-center gap-1">
                  <Loader2 size={10} className="animate-spin" />
                  Uploading...
                </span>
              )}
              {/* Memory status indicator */}
              <span 
                className={`flex items-center gap-1 ${memoryConnected ? 'text-green-600' : 'text-stone-400'}`}
                title={memoryConnected ? 'Supermemory connected' : 'Memory offline'}
              >
                <Brain size={12} />
                {memoryConnected ? 'Memory' : 'Offline'}
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
    uploading: <Loader2 size={14} className="animate-spin text-blue-500" />,
    processing: <Loader2 size={14} className="animate-spin text-amber-500" />,
    ready: <File size={14} />,
    error: <AlertCircle size={14} className="text-red-500" />,
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
        group flex cursor-pointer items-center gap-2 rounded-md p-2 
        hover:bg-white hover:shadow-sm border transition-all
        ${isSelected ? 'bg-blue-50 border-blue-200' : 'border-transparent hover:border-stone-200/60'}
        ${isCollapsed ? 'justify-center' : ''}
      `}
      title={isCollapsed ? doc.filename : undefined}
      onClick={doc.status === 'ready' ? onToggleSelect : undefined}
    >
      {/* Selection checkbox */}
      {!isCollapsed && doc.status === 'ready' && (
        <button 
          className="flex-shrink-0"
          onClick={(e) => { e.stopPropagation(); onToggleSelect(); }}
        >
          {isSelected 
            ? <CheckSquare size={14} className="text-blue-500" /> 
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
          <div className="text-sm font-medium text-stone-700 line-clamp-1">
            {doc.filename}
          </div>
          <div className="text-[10px] text-stone-400 flex items-center gap-2">
            <span className="uppercase">{doc.type}</span>
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
          className="flex-shrink-0 p-1 opacity-0 group-hover:opacity-100 hover:bg-red-100 rounded text-stone-400 hover:text-red-500 transition-all"
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
      flex items-center justify-center gap-2 rounded-md py-1.5 text-xs font-medium transition-all duration-200
      ${active ? 'bg-white text-stone-800 shadow-[0_1px_2px_rgba(0,0,0,0.04)] border border-stone-200/60' : 'text-stone-500 hover:text-stone-700 hover:bg-stone-200/40'}
      ${collapsed ? 'aspect-square w-full' : 'flex-1'}
    `}
    title={collapsed ? label : undefined}
  >
    {icon}
    {!collapsed && label}
  </button>
);
