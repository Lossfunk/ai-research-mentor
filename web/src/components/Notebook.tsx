import React, { useEffect, useMemo, useRef, useState } from 'react';
import { useEditor, EditorContent } from '@tiptap/react';
import StarterKit from '@tiptap/starter-kit';
import Placeholder from '@tiptap/extension-placeholder';
import Image from '@tiptap/extension-image';
import { 
  ChevronLeft, ChevronRight, Bold, Italic, List, ListOrdered, Quote, Heading1, Heading2, Image as ImageIcon, Download, Save 
} from 'lucide-react';
import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";
import { saveAs } from 'file-saver';
import { tiptapJsonToHtml, tiptapJsonToMarkdown } from '@/lib/export';
import { useNotebookStore } from '@/store/useNotebookStore';

function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

const ToolbarButton = ({ 
  onClick, 
  isActive, 
  children 
}: { 
  onClick: () => void; 
  isActive?: boolean; 
  children: React.ReactNode; 
}) => (
  <button
    onClick={onClick}
    className={cn(
      "p-1.5 rounded-full transition-all hover:scale-105 active:scale-95",
      isActive 
        ? "bg-stone-100 text-stone-900 shadow-sm" 
        : "text-stone-400 hover:text-stone-700 hover:bg-stone-50"
    )}
  >
    {children}
  </button>
);

export const Notebook = () => {
  const scrollRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [pageHeight, setPageHeight] = useState(900);
  const [currentPage, setCurrentPage] = useState(1);
  const [maxPage, setMaxPage] = useState(1);
  const [mode, setMode] = useState<'paged' | 'continuous'>('paged');
  const [showExportMenu, setShowExportMenu] = useState(false);

  const { content: storedContent, setContent, saveNote, lastUpdatedAt } = useNotebookStore();
  const saveTimerRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    const computePageHeight = () => {
      // Leave room for header chrome and breathing space
      const base = Math.max(window.innerHeight - 180, 720);
      setPageHeight(base);
    };
    computePageHeight();
    window.addEventListener('resize', computePageHeight);
    return () => window.removeEventListener('resize', computePageHeight);
  }, []);
  
  useEffect(() => {
    // Re-evaluate paging when mode or height changes
    handleScroll();
  }, [mode, pageHeight]);

  const editor = useEditor({
    immediatelyRender: false,
    extensions: [
      StarterKit,
      Image.configure({
        HTMLAttributes: {
          class: 'my-4 rounded shadow-sm max-w-full mx-auto',
        },
      }),
      Placeholder.configure({
        placeholder: 'Start writing your research paper... (Option+Enter for AI)',
      }),
    ],
    content: storedContent ?? `
      <h1>Research Proposal</h1>
      <p>Start by outlining your hypothesis here.</p>
    `,
    editorProps: {
      attributes: {
        class: 'prose prose-stone prose-lg max-w-none focus:outline-none',
      },
      handlePaste: (_view, event) => {
        const items = event.clipboardData?.items;
        if (!items) return false;
        const imageItem = Array.from(items).find((i) => i.type.startsWith('image/'));
        if (imageItem) {
          const file = imageItem.getAsFile();
          if (file) {
            insertImageFile(file);
            return true;
          }
        }
        return false;
      },
      handleDrop: (_view, event) => {
        if (!event.dataTransfer?.files?.length) return false;
        const file = event.dataTransfer.files[0];
        if (file.type.startsWith('image/')) {
          event.preventDefault();
          insertImageFile(file);
          return true;
        }
        return false;
      },
    },
    onUpdate: ({ editor }) => {
      if (saveTimerRef.current) clearTimeout(saveTimerRef.current);
      const json = editor.getJSON();
      saveTimerRef.current = setTimeout(() => {
        setContent(json);
      }, 400);
    },
  });

  useEffect(() => {
    if (editor) {
      setContent(editor.getJSON());
    }
  }, [editor]);

  useEffect(() => {
    if (editor && storedContent) {
      editor.commands.setContent(storedContent);
    }
  }, [editor, storedContent]);

  const insertImageFile = async (file: File) => {
    if (!editor) return;
    try {
      const formData = new FormData();
      formData.append('file', file);
      const res = await fetch('/api/upload-image', {
        method: 'POST',
        body: formData,
      });
      if (res.ok) {
        const data = await res.json();
        editor.chain().focus().setImage({ src: data.url, alt: file.name }).run();
        return;
      }
    } catch (err) {
      console.warn('Upload failed, falling back to data URL', err);
    }

    const reader = new FileReader();
    reader.onload = () => {
      const src = reader.result as string;
      editor.chain().focus().setImage({ src, alt: file.name }).run();
    };
    reader.readAsDataURL(file);
  };

  const handleImagePick = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file && file.type.startsWith('image/')) {
      insertImageFile(file);
    }
    e.target.value = '';
  };

  const exportAsDocx = () => {
    if (!editor) return;
    const html = tiptapJsonToHtml(editor.getJSON());
    const content = `<!DOCTYPE html><html><head><meta charset="utf-8" /></head><body>${html}</body></html>`;
    const blob = new Blob([content], { type: 'application/msword' });
    saveAs(blob, 'notes.doc');
  };

  const exportAsMarkdown = () => {
    if (!editor) return;
    const md = tiptapJsonToMarkdown(editor.getJSON());
    const blob = new Blob([md], { type: 'text/markdown' });
    saveAs(blob, 'notes.md');
  };

  const handleScroll = () => {
    const el = scrollRef.current;
    if (!el) return;
    if (mode === 'paged') {
      const page = Math.max(1, Math.floor(el.scrollTop / pageHeight) + 1);
      const total = Math.max(1, Math.ceil(el.scrollHeight / pageHeight));
      setCurrentPage(page);
      setMaxPage(total);
    } else {
      setCurrentPage(1);
      setMaxPage(1);
    }
  };

  const scrollByPage = (delta: number) => {
    if (mode !== 'paged') return;
    const el = scrollRef.current;
    if (!el) return;
    const target = Math.max(0, el.scrollTop + delta * pageHeight);
    el.scrollTo({ top: target, behavior: 'smooth' });
  };

  const pageBackground = useMemo(() => {
    if (mode !== 'paged') return undefined;
    return {
      ['--page-height' as string]: `${pageHeight}px`,
      backgroundImage:
        'linear-gradient(to bottom, transparent calc(var(--page-height) - 36px), rgba(0,0,0,0.05), transparent calc(var(--page-height) - 12px))',
      backgroundSize: '100% var(--page-height)',
    };
  }, [mode, pageHeight]);

  return (
    <div className="relative w-full h-full">
      <div
        ref={scrollRef}
        onScroll={handleScroll}
        className="relative h-full overflow-y-auto rounded-2xl"
        style={pageBackground}
      >
        {/* Layered page stack for book-like feel */}
        <div className="absolute inset-3 rounded-[18px] bg-white/70 shadow-[0_20px_50px_-30px_rgba(0,0,0,0.35)] -z-10" />
        <div className="absolute inset-1 rounded-[20px] bg-white/90 shadow-[0_30px_80px_-40px_rgba(0,0,0,0.35)] -z-20" />

        <div
          className="relative z-10 w-full min-h-[calc(100vh-6rem)] bg-white shadow-[0_2px_40px_-12px_rgba(0,0,0,0.08)] border border-stone-200/60 rounded-2xl overflow-hidden flex flex-col"
          style={{ minHeight: `${pageHeight}px` }}
        >
          {/* Top Accent Line */}
          <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-orange-300 via-red-300 to-indigo-300 opacity-80" />

          {/* Floating Sticky Pill Toolbar */}
          {editor && (
            <div 
              data-tour-id="notebook-toolbar"
              className="sticky top-6 z-20 mx-auto w-fit mb-4 transition-all duration-300 animate-fade-in"
            >
              <div className="flex items-center gap-1 px-3 py-1.5 bg-white/90 backdrop-blur-md border border-stone-200 rounded-full shadow-[0_4px_20px_-4px_rgba(0,0,0,0.08)]">
                
                <ToolbarButton 
                  isActive={editor.isActive('heading', { level: 1 })} 
                  onClick={() => editor.chain().focus().toggleHeading({ level: 1 }).run()}
                >
                  <Heading1 size={16} />
                </ToolbarButton>

                <ToolbarButton 
                  isActive={editor.isActive('heading', { level: 2 })} 
                  onClick={() => editor.chain().focus().toggleHeading({ level: 2 }).run()}
                >
                  <Heading2 size={16} />
                </ToolbarButton>

                <div className="w-[1px] h-4 bg-stone-200 mx-1" />

                <ToolbarButton 
                  isActive={editor.isActive('bold')} 
                  onClick={() => editor.chain().focus().toggleBold().run()}
                >
                  <Bold size={16} />
                </ToolbarButton>

                <ToolbarButton 
                  isActive={editor.isActive('italic')} 
                  onClick={() => editor.chain().focus().toggleItalic().run()}
                >
                  <Italic size={16} />
                </ToolbarButton>

                <div className="w-[1px] h-4 bg-stone-200 mx-1" />

                <ToolbarButton 
                  isActive={editor.isActive('bulletList')} 
                  onClick={() => editor.chain().focus().toggleBulletList().run()}
                >
                  <List size={16} />
                </ToolbarButton>

                <ToolbarButton 
                  isActive={editor.isActive('orderedList')} 
                  onClick={() => editor.chain().focus().toggleOrderedList().run()}
                >
                  <ListOrdered size={16} />
                </ToolbarButton>

                <ToolbarButton 
                  isActive={editor.isActive('blockquote')} 
                  onClick={() => editor.chain().focus().toggleBlockquote().run()}
                >
                  <Quote size={16} />
                </ToolbarButton>

                <div className="w-[1px] h-4 bg-stone-200 mx-1" />

                <ToolbarButton 
                  onClick={() => fileInputRef.current?.click()}
                >
                  <ImageIcon size={16} />
                </ToolbarButton>

                <input 
                  type="file" 
                  accept="image/*" 
                  className="hidden" 
                  ref={fileInputRef}
                  onChange={handleImagePick}
                />

                <div className="w-[1px] h-4 bg-stone-200 mx-1" />

                <div className="relative">
                  <ToolbarButton onClick={() => setShowExportMenu(!showExportMenu)}>
                    <Download size={16} />
                  </ToolbarButton>
                  {showExportMenu && (
                    <div className="absolute right-0 mt-2 w-48 rounded-lg border border-stone-200 bg-white shadow-lg z-50 overflow-hidden">
                      <button 
                        onClick={() => { setShowExportMenu(false); exportAsDocx(); }}
                        className="w-full px-3 py-2 text-left text-xs font-mono hover:bg-stone-50"
                      >
                        Export as DOCX
                      </button>
                      <button 
                        onClick={() => { setShowExportMenu(false); exportAsMarkdown(); }}
                        className="w-full px-3 py-2 text-left text-xs font-mono hover:bg-stone-50"
                      >
                        Export as Markdown
                      </button>
                    </div>
                  )}
                </div>

                <button 
                  onClick={() => saveNote()}
                  className="ml-2 px-3 py-1.5 rounded-full bg-amber-100 text-amber-900 border border-amber-200 text-[11px] font-mono hover:bg-amber-200/70 transition-all flex items-center gap-1"
                  title="Save to Notes"
                >
                  <Save size={12} />
                  Save
                </button>

                {lastUpdatedAt && (
                  <span className="ml-2 text-[10px] text-stone-400 font-mono">
                    Saved {new Date(lastUpdatedAt).toLocaleTimeString()}
                  </span>
                )}

              </div>
            </div>
          )}

          {editor ? (
            <EditorContent 
              editor={editor} 
              className="flex-1 px-12 pt-4 pb-36 prose prose-stone prose-lg max-w-none focus:outline-none"
            />
          ) : (
            <div className="flex-1 px-12 pt-12 pb-24 text-sm text-stone-400">
              Loading editor...
            </div>
          )}
        </div>
      </div>

      {/* Page navigation & controls (viewport fixed) */}
      <div className="fixed bottom-6 left-1/2 -translate-x-1/2 flex items-center gap-2 z-50 px-3">
        <button
          onClick={() => {
            const nextMode = mode === 'paged' ? 'continuous' : 'paged';
            setMode(nextMode);
            if (nextMode === 'paged') {
              scrollRef.current?.scrollTo({ top: 0, behavior: 'smooth' });
            }
          }}
          className="px-3 py-1.5 rounded-full bg-white shadow-[0_4px_20px_rgba(0,0,0,0.12)] border border-stone-200 hover:translate-y-[-1px] transition-all text-[11px] font-mono text-stone-700"
        >
          {mode === 'paged' ? 'Paged view' : 'Continuous view'}
        </button>

        {mode === 'paged' && (
          <>
            <button
              onClick={() => scrollByPage(-1)}
              className="p-2 rounded-full bg-white shadow-[0_4px_20px_rgba(0,0,0,0.12)] border border-stone-200 hover:translate-y-[-1px] transition-all text-stone-600 hover:text-stone-900 disabled:opacity-40"
              title="Previous page"
              disabled={currentPage <= 1}
            >
              <ChevronLeft size={16} />
            </button>
            <button
              onClick={() => scrollByPage(1)}
              className="p-2 rounded-full bg-white shadow-[0_4px_20px_rgba(0,0,0,0.12)] border border-stone-200 hover:translate-y-[-1px] transition-all text-stone-600 hover:text-stone-900 disabled:opacity-40"
              title="Next page"
              disabled={currentPage >= maxPage}
            >
              <ChevronRight size={16} />
            </button>
            <span className="px-3 py-1 rounded-full bg-white/90 border border-stone-200 text-[11px] font-mono text-stone-600 shadow-sm">
              Page {currentPage}/{maxPage}
            </span>
          </>
        )}
      </div>
    </div>
  );
};
