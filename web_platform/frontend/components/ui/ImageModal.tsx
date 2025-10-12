'use client';

import { X, Download, ZoomIn, ZoomOut, RotateCw } from 'lucide-react';
import { useState } from 'react';

interface ImageModalProps {
    src: string;
    alt: string;
    onClose: () => void;
}

export default function ImageModal({ src, alt, onClose }: ImageModalProps) {
    const [zoom, setZoom] = useState(100);
    const [rotation, setRotation] = useState(0);

    const handleDownload = () => {
        const link = document.createElement('a');
        link.href = src;
        link.download = alt || 'image.png';
        link.click();
    };

    return (
        <div
            className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm"
            onClick={onClose}
        >
            <div
                className="relative max-w-7xl max-h-[90vh] overflow-auto bg-zinc-900 rounded-lg border border-zinc-700"
                onClick={(e) => e.stopPropagation()}
            >
                {/* Header */}
                <div className="sticky top-0 z-10 flex items-center justify-between p-4 bg-zinc-900/95 border-b border-zinc-700">
                    <div className="flex items-center gap-2">
                        <span className="text-sm font-medium text-zinc-300">{alt}</span>
                    </div>

                    <div className="flex items-center gap-2">
                        {/* Zoom Controls */}
                        <button
                            onClick={() => setZoom(Math.max(50, zoom - 10))}
                            className="p-2 hover:bg-zinc-700 rounded text-zinc-400"
                            title="Zoom out"
                        >
                            <ZoomOut className="h-4 w-4" />
                        </button>
                        <span className="text-xs text-zinc-500 w-12 text-center">{zoom}%</span>
                        <button
                            onClick={() => setZoom(Math.min(200, zoom + 10))}
                            className="p-2 hover:bg-zinc-700 rounded text-zinc-400"
                            title="Zoom in"
                        >
                            <ZoomIn className="h-4 w-4" />
                        </button>

                        {/* Rotate */}
                        <button
                            onClick={() => setRotation((rotation + 90) % 360)}
                            className="p-2 hover:bg-zinc-700 rounded text-zinc-400"
                            title="Rotate"
                        >
                            <RotateCw className="h-4 w-4" />
                        </button>

                        {/* Download */}
                        <button
                            onClick={handleDownload}
                            className="p-2 hover:bg-zinc-700 rounded text-zinc-400"
                            title="Download"
                        >
                            <Download className="h-4 w-4" />
                        </button>

                        {/* Close */}
                        <button
                            onClick={onClose}
                            className="p-2 hover:bg-zinc-700 rounded text-zinc-400"
                            title="Close"
                        >
                            <X className="h-4 w-4" />
                        </button>
                    </div>
                </div>

                {/* Image */}
                <div className="p-4 flex items-center justify-center min-h-[400px]">
                    <img
                        src={src}
                        alt={alt}
                        style={{
                            transform: `scale(${zoom / 100}) rotate(${rotation}deg)`,
                            transition: 'transform 0.2s ease',
                        }}
                        className="max-w-full max-h-[calc(90vh-100px)] object-contain"
                    />
                </div>
            </div>
        </div>
    );
}

