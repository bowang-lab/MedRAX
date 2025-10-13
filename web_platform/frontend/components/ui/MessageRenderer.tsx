'use client';

import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeSanitize from 'rehype-sanitize';
import { Maximize2 } from 'lucide-react';

interface MessageRendererProps {
    content: string;
    apiBase: string;
    onImageClick?: (src: string, alt: string) => void;
}

export default function MessageRenderer({ content, apiBase, onImageClick }: MessageRendererProps) {
    // Ensure content is always a string
    const safeContent = typeof content === 'string' ? content : String(content || '');

    return (
        <div className="prose prose-invert prose-sm max-w-none">
            <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                rehypePlugins={[rehypeSanitize]}
                components={{
                    // Custom image rendering
                    img: ({ node, src, alt, ...props }) => {
                        let imagePath = src || '';

                        // If path doesn't start with http, prepend API base
                        if (imagePath && !imagePath.startsWith('http')) {
                            imagePath = imagePath.startsWith('/') ? imagePath.slice(1) : imagePath;
                            imagePath = `${apiBase}/${imagePath}`;
                        }

                        return (
                            <div className="my-3 relative group">
                                <img
                                    src={imagePath}
                                    alt={alt || 'Image'}
                                    className="max-w-full rounded-lg border border-zinc-700 cursor-pointer hover:border-blue-500 transition-colors shadow-md"
                                    onClick={() => onImageClick?.(imagePath, alt || 'Image')}
                                    onError={(e) => {
                                        const target = e.target as HTMLImageElement;
                                        target.src = 'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" width="200" height="100"%3E%3Crect fill="%23333" width="200" height="100"/%3E%3Ctext fill="%23888" x="50%25" y="50%25" text-anchor="middle" dy=".3em"%3EImage not found%3C/text%3E%3C/svg%3E';
                                        target.className = 'max-w-full rounded border border-red-500/50';
                                    }}
                                    {...props}
                                />
                                <button
                                    onClick={() => onImageClick?.(imagePath, alt || 'Image')}
                                    className="absolute top-2 right-2 p-1.5 bg-black/60 hover:bg-black/80 rounded opacity-0 group-hover:opacity-100 transition-opacity"
                                    title="View fullscreen"
                                >
                                    <Maximize2 className="h-3 w-3 text-white" />
                                </button>
                                {alt && (
                                    <div className="text-xs text-zinc-500 mt-1.5 text-center italic">{alt}</div>
                                )}
                            </div>
                        );
                    },
                    // Custom heading styles
                    h1: ({ node, ...props }) => <h1 className="text-xl font-bold text-blue-400 mt-4 mb-2" {...props} />,
                    h2: ({ node, ...props }) => <h2 className="text-lg font-bold text-blue-400 mt-3 mb-2" {...props} />,
                    h3: ({ node, ...props }) => <h3 className="text-base font-semibold text-emerald-400 mt-3 mb-1.5" {...props} />,
                    h4: ({ node, ...props }) => <h4 className="text-sm font-semibold text-emerald-400 mt-2 mb-1" {...props} />,

                    // Custom paragraph styles
                    p: ({ node, ...props }) => <p className="my-2 leading-relaxed text-zinc-200" {...props} />,

                    // Custom list styles
                    ul: ({ node, ...props }) => <ul className="my-2 ml-4 space-y-1 list-disc list-inside text-zinc-200" {...props} />,
                    ol: ({ node, ...props }) => <ol className="my-2 ml-4 space-y-1 list-decimal list-inside text-zinc-200" {...props} />,
                    li: ({ node, ...props }) => <li className="text-zinc-200" {...props} />,

                    // Custom code block styles
                    code: ({ node, inline, ...props }: any) => {
                        if (inline) {
                            return <code className="px-1.5 py-0.5 bg-zinc-700/50 text-blue-300 rounded text-xs font-mono" {...props} />;
                        }
                        return <code className="block p-3 bg-zinc-900 text-emerald-300 rounded-lg text-xs font-mono overflow-x-auto my-2" {...props} />;
                    },

                    // Custom table styles
                    table: ({ node, ...props }) => (
                        <div className="overflow-x-auto my-3">
                            <table className="min-w-full border border-zinc-700 rounded-lg" {...props} />
                        </div>
                    ),
                    thead: ({ node, ...props }) => <thead className="bg-zinc-800" {...props} />,
                    tbody: ({ node, ...props }) => <tbody className="divide-y divide-zinc-700" {...props} />,
                    tr: ({ node, ...props }) => <tr className="border-b border-zinc-700" {...props} />,
                    th: ({ node, ...props }) => <th className="px-3 py-2 text-left text-xs font-semibold text-blue-400 uppercase tracking-wider" {...props} />,
                    td: ({ node, ...props }) => <td className="px-3 py-2 text-sm text-zinc-300" {...props} />,

                    // Custom blockquote styles
                    blockquote: ({ node, ...props }) => (
                        <blockquote className="border-l-4 border-blue-500 pl-4 py-2 my-3 bg-zinc-800/50 rounded-r italic text-zinc-300" {...props} />
                    ),

                    // Custom link styles
                    a: ({ node, ...props }) => (
                        <a className="text-blue-400 hover:text-blue-300 underline transition-colors" target="_blank" rel="noopener noreferrer" {...props} />
                    ),

                    // Custom horizontal rule
                    hr: ({ node, ...props }) => <hr className="my-4 border-zinc-700" {...props} />,

                    // Strong/bold - highlight medical terms
                    strong: ({ node, ...props }) => <strong className="font-bold text-emerald-300" {...props} />,

                    // Emphasis/italic
                    em: ({ node, ...props }) => <em className="italic text-zinc-300" {...props} />,
                }}
            >
                {safeContent}
            </ReactMarkdown>
        </div>
    );
}

