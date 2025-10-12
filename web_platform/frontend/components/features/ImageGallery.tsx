'use client';

import { useState } from 'react';
import { Image as ImageIcon, X, Eye, Upload } from 'lucide-react';

interface ImageGalleryProps {
    images: string[];
    currentIndex: number;
    apiBase: string;
    onSelectImage: (index: number) => void;
    onDeleteImage?: (index: number) => void;
    onUploadClick: () => void;
}

export default function ImageGallery({
    images,
    currentIndex,
    apiBase,
    onSelectImage,
    onDeleteImage,
    onUploadClick
}: ImageGalleryProps) {
    const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);

    if (images.length === 0) {
        return (
            <div className="text-center text-zinc-500 text-sm mt-8">
                <ImageIcon className="h-12 w-12 mx-auto mb-3 text-zinc-600" />
                <p>No images uploaded yet</p>
                <button
                    onClick={onUploadClick}
                    className="mt-4 px-4 py-2 bg-gradient-to-r from-blue-600 to-emerald-600 hover:from-blue-500 hover:to-emerald-500 rounded-lg text-sm font-semibold flex items-center gap-2 mx-auto transition-all duration-200 hover:scale-105"
                >
                    <Upload className="h-4 w-4" />
                    Upload Images
                </button>
            </div>
        );
    }

    return (
        <div className="space-y-3">
            {/* Image counter */}
            <div className="flex items-center justify-between px-1">
                <span className="text-xs text-zinc-500 font-medium">
                    {images.length} image{images.length !== 1 ? 's' : ''}
                </span>
                {currentIndex !== null && (
                    <span className="text-xs text-blue-400 font-medium">
                        Viewing {currentIndex + 1} of {images.length}
                    </span>
                )}
            </div>

            {/* Image thumbnails */}
            <div className="space-y-2">
                {images.map((img, idx) => (
                    <div
                        key={idx}
                        onClick={() => onSelectImage(idx)}
                        onMouseEnter={() => setHoveredIndex(idx)}
                        onMouseLeave={() => setHoveredIndex(null)}
                        className={`relative cursor-pointer rounded-xl overflow-hidden border-2 transition-all duration-200 hover:scale-[1.02] ${idx === currentIndex
                                ? 'border-blue-500 shadow-lg shadow-blue-500/20'
                                : 'border-zinc-800/50 hover:border-zinc-700'
                            }`}
                    >
                        <img
                            src={`${apiBase}/${img}`}
                            alt={`Upload ${idx + 1}`}
                            className="w-full h-32 object-cover"
                        />

                        {/* Active indicator */}
                        {idx === currentIndex && (
                            <div className="absolute top-2 right-2 px-2 py-1 bg-blue-500 text-white text-xs rounded-full font-semibold flex items-center gap-1">
                                <Eye className="h-3 w-3" />
                                Active
                            </div>
                        )}

                        {/* Delete button on hover */}
                        {hoveredIndex === idx && onDeleteImage && idx !== currentIndex && (
                            <button
                                onClick={(e) => {
                                    e.stopPropagation();
                                    onDeleteImage(idx);
                                }}
                                className="absolute top-2 left-2 p-1.5 bg-red-500/80 hover:bg-red-500 rounded-full transition-colors"
                                title="Remove image"
                            >
                                <X className="h-3 w-3 text-white" />
                            </button>
                        )}

                        {/* Image number */}
                        <div className="absolute bottom-2 left-2 px-2 py-0.5 bg-black/60 text-white text-xs rounded">
                            #{idx + 1}
                        </div>
                    </div>
                ))}
            </div>

            {/* Add more button */}
            <button
                onClick={onUploadClick}
                className="w-full py-2 border-2 border-dashed border-zinc-700 hover:border-blue-500/50 rounded-lg text-sm text-zinc-400 hover:text-zinc-300 transition-all duration-200 hover:bg-zinc-800/30 flex items-center justify-center gap-2"
            >
                <Upload className="h-4 w-4" />
                Add More Images
            </button>
        </div>
    );
}

