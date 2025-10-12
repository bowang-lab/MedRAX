'use client';

import { Upload } from 'lucide-react';

interface ImageUploadZoneProps {
    dragActive: boolean;
    onDragEnter: (e: React.DragEvent) => void;
    onDragLeave: (e: React.DragEvent) => void;
    onDragOver: (e: React.DragEvent) => void;
    onDrop: (e: React.DragEvent) => void;
    onClick: () => void;
}

export default function ImageUploadZone({
    dragActive,
    onDragEnter,
    onDragLeave,
    onDragOver,
    onDrop,
    onClick
}: ImageUploadZoneProps) {
    return (
        <div
            className={`border-2 border-dashed rounded-xl p-12 text-center cursor-pointer transition-colors ${dragActive
                    ? 'border-blue-500 bg-blue-500/10'
                    : 'border-zinc-700 hover:border-zinc-600 bg-zinc-900/30'
                }`}
            onDragEnter={onDragEnter}
            onDragLeave={onDragLeave}
            onDragOver={onDragOver}
            onDrop={onDrop}
            onClick={onClick}
        >
            <Upload className="h-12 w-12 mx-auto mb-4 text-zinc-600" />
            <h3 className="text-lg font-semibold mb-2">Drop X-ray images here</h3>
            <p className="text-sm text-zinc-500">
                or click to browse<br />
                Supports DICOM, JPG, PNG
            </p>
        </div>
    );
}



