'use client';

import { useState } from 'react';
import { Maximize2 } from 'lucide-react';

interface MessageRendererProps {
    content: string;
    apiBase: string;
    onImageClick?: (src: string, alt: string) => void;
}

export default function MessageRenderer({ content, apiBase, onImageClick }: MessageRendererProps) {
    // Parse markdown-style images: ![alt](path)
    const renderContent = () => {
        const parts: React.ReactNode[] = [];
        let lastIndex = 0;

        // Match ![...](...)
        const imageRegex = /!\[(.*?)\]\((.*?)\)/g;
        let match;

        while ((match = imageRegex.exec(content)) !== null) {
            // Add text before the image
            if (match.index > lastIndex) {
                parts.push(
                    <span key={`text-${lastIndex}`}>
                        {content.substring(lastIndex, match.index)}
                    </span>
                );
            }

            const altText = match[1];
            let imagePath = match[2];

            // If path doesn't start with http, prepend API base
            if (imagePath && !imagePath.startsWith('http')) {
                // Remove leading slash if present
                imagePath = imagePath.startsWith('/') ? imagePath.slice(1) : imagePath;
                imagePath = `${apiBase}/${imagePath}`;
            }

            // Add the image if path is not empty
            if (imagePath) {
                parts.push(
                    <div key={`img-${match.index}`} className="my-2 relative group">
                        <img
                            src={imagePath}
                            alt={altText || 'Image'}
                            className="max-w-full rounded border border-zinc-700 cursor-pointer hover:border-blue-500 transition-colors"
                            onClick={() => onImageClick?.(imagePath, altText)}
                            onError={(e) => {
                                // Show broken image placeholder
                                const target = e.target as HTMLImageElement;
                                target.src = 'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" width="200" height="100"%3E%3Crect fill="%23333" width="200" height="100"/%3E%3Ctext fill="%23888" x="50%25" y="50%25" text-anchor="middle" dy=".3em"%3EImage not found%3C/text%3E%3C/svg%3E';
                                target.className = 'max-w-full rounded border border-red-500/50';
                            }}
                        />
                        {/* Expand button overlay */}
                        <button
                            onClick={() => onImageClick?.(imagePath, altText)}
                            className="absolute top-2 right-2 p-1.5 bg-black/60 hover:bg-black/80 rounded opacity-0 group-hover:opacity-100 transition-opacity"
                            title="View fullscreen"
                        >
                            <Maximize2 className="h-3 w-3 text-white" />
                        </button>
                        {altText && (
                            <div className="text-xs text-zinc-500 mt-1">{altText}</div>
                        )}
                    </div>
                );
            }

            lastIndex = match.index + match[0].length;
        }

        // Add remaining text
        if (lastIndex < content.length) {
            parts.push(
                <span key={`text-${lastIndex}`}>
                    {content.substring(lastIndex)}
                </span>
            );
        }

        return parts.length > 0 ? parts : content;
    };

    return (
        <div className="whitespace-pre-wrap break-words">
            {renderContent()}
        </div>
    );
}

