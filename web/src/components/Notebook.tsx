import { useEditor, EditorContent } from '@tiptap/react';
import StarterKit from '@tiptap/starter-kit';
import Placeholder from '@tiptap/extension-placeholder';

export const Notebook = () => {
  const editor = useEditor({
    immediatelyRender: false,
    extensions: [
      StarterKit,
      Placeholder.configure({
        placeholder: 'Start writing your research paper... (Option+Enter for AI)',
      }),
    ],
    content: `
      <h1>Research Proposal</h1>
      <p>Start by outlining your hypothesis here.</p>
    `,
    editorProps: {
      attributes: {
        class: 'prose prose-stone prose-lg max-w-none focus:outline-none',
      },
    },
  });

  if (!editor) {
    return null;
  }

  return (
    <div className="mx-auto max-w-4xl py-16 px-12 min-h-[calc(100vh-6rem)] mt-6 bg-white shadow-[0_2px_40px_-12px_rgba(0,0,0,0.08)] border border-stone-200/60 rounded-t-xl relative">
        {/* Top Accent Line */}
        <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-orange-300 via-red-300 to-indigo-300 rounded-t-xl opacity-80" />
      <EditorContent editor={editor} />
    </div>
  );
};
