import { useEditor, EditorContent } from '@tiptap/react';
import StarterKit from '@tiptap/starter-kit';
import Placeholder from '@tiptap/extension-placeholder';

export const Notebook = () => {
  const editor = useEditor({
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
    <div className="mx-auto max-w-4xl py-12 px-8 min-h-screen bg-white shadow-sm border-x border-stone-200/50">
      <EditorContent editor={editor} />
    </div>
  );
};
