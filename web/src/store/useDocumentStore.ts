import { create } from 'zustand';

export type UploadedDocument = {
  id: string;
  filename: string;
  type: 'pdf' | 'docx' | 'txt' | 'md';
  size: number;
  uploadedAt: Date;
  status: 'uploading' | 'processing' | 'ready' | 'error';
  content?: string; // Parsed text content
  error?: string;
};

type DocumentState = {
  documents: UploadedDocument[];
  selectedDocumentIds: Set<string>;
  isUploading: boolean;
  
  addDocument: (doc: UploadedDocument) => void;
  updateDocument: (id: string, updates: Partial<UploadedDocument>) => void;
  removeDocument: (id: string) => void;
  toggleDocumentSelection: (id: string) => void;
  selectAllDocuments: () => void;
  clearSelection: () => void;
  setUploading: (v: boolean) => void;
  getSelectedContent: () => string;
};

export const useDocumentStore = create<DocumentState>((set, get) => ({
  documents: [],
  selectedDocumentIds: new Set<string>(),
  isUploading: false,

  addDocument: (doc) => set((state) => ({ 
    documents: [...state.documents, doc] 
  })),

  updateDocument: (id, updates) => set((state) => ({
    documents: state.documents.map(d => 
      d.id === id ? { ...d, ...updates } : d
    )
  })),

  removeDocument: (id) => set((state) => {
    const newSelected = new Set(state.selectedDocumentIds);
    newSelected.delete(id);
    return {
      documents: state.documents.filter(d => d.id !== id),
      selectedDocumentIds: newSelected
    };
  }),

  toggleDocumentSelection: (id) => set((state) => {
    const newSelected = new Set(state.selectedDocumentIds);
    if (newSelected.has(id)) {
      newSelected.delete(id);
    } else {
      newSelected.add(id);
    }
    return { selectedDocumentIds: newSelected };
  }),

  selectAllDocuments: () => set((state) => ({
    selectedDocumentIds: new Set(state.documents.filter(d => d.status === 'ready').map(d => d.id))
  })),

  clearSelection: () => set({ selectedDocumentIds: new Set() }),

  setUploading: (v) => set({ isUploading: v }),

  getSelectedContent: () => {
    const state = get();
    const selectedDocs = state.documents.filter(
      d => state.selectedDocumentIds.has(d.id) && d.status === 'ready' && d.content
    );
    if (selectedDocs.length === 0) return '';
    
    return selectedDocs.map(d => 
      `--- Document: ${d.filename} ---\n${d.content}\n`
    ).join('\n');
  }
}));
