import React from 'react';
import ReactMarkdown from 'react-markdown';
import rehypeHighlight from 'rehype-highlight';
import 'highlight.js/styles/atom-one-dark.css'; // Import code highlight style

interface MarkdownRendererProps {
  content: string;
}

// Turn inline citation tokens [P1] into clickable links when a footer provides URLs:
// pattern in footer: "[P1] Title — https://example.com"
function linkifyCitations(markdown: string): string {
  const footerRegex = /\[(A|P|G|W)(\d+)\]\s+([^\n—]+?)\s+—\s+(https?:\/\/\S+)/g;
  const linkMap: Record<string, string> = {};

  let match;
  while ((match = footerRegex.exec(markdown)) !== null) {
    const id = `${match[1]}${match[2]}`;
    linkMap[id] = match[4];
  }

  if (Object.keys(linkMap).length === 0) return markdown;

  // Replace citations in the main text with markdown links; leave footer as-is.
  return markdown.replace(/\[(A|P|G|W)(\d+)\]/g, (full, prefix, num) => {
    const key = `${prefix}${num}`;
    const url = linkMap[key];
    return url ? `[${key}](${url})` : full;
  });
}

export const MarkdownRenderer: React.FC<MarkdownRendererProps> = ({ content }) => {
  const linked = linkifyCitations(content);
  return (
    <div className="prose prose-stone prose-sm max-w-none 
      prose-headings:font-semibold prose-headings:tracking-tight prose-headings:text-stone-900
      prose-p:text-stone-700 prose-p:leading-relaxed
      prose-a:text-indigo-600 prose-a:no-underline hover:prose-a:underline
      prose-strong:text-stone-900 prose-strong:font-semibold
      prose-ul:text-stone-700 prose-ol:text-stone-700
      prose-li:marker:text-stone-400
      prose-code:text-stone-800 prose-code:bg-stone-100 prose-code:px-1.5 prose-code:py-0.5 prose-code:rounded-md prose-code:font-mono prose-code:text-[0.9em] prose-code:before:content-none prose-code:after:content-none
      prose-pre:bg-stone-900 prose-pre:text-stone-50 prose-pre:rounded-xl prose-pre:shadow-sm
      prose-blockquote:border-l-4 prose-blockquote:border-stone-200 prose-blockquote:pl-4 prose-blockquote:italic prose-blockquote:text-stone-600
      prose-img:rounded-lg prose-img:shadow-sm
      break-words">
      <ReactMarkdown rehypePlugins={[rehypeHighlight]}>
        {linked}
      </ReactMarkdown>
    </div>
  );
};
