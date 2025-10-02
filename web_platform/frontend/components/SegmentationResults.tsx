'use client';

import { useState } from 'react';

interface SegmentationResultsProps {
    result: any;
    idx: number;
    currentImage: string | null;
    apiBase: string;
}

export default function SegmentationResults({ result, idx, currentImage, apiBase }: SegmentationResultsProps) {
    const [showRaw, setShowRaw] = useState(false);

    if (!result.result) {
        return null;
    }

    // Get segmentation image path from either result or metadata
    const segImagePath = result.result?.segmentation_image_path || result.metadata?.segmentation_image_path;

    return (
        <div className="space-y-4">
            {/* Side-by-side comparison */}
            {segImagePath && currentImage && (
                <div className="grid grid-cols-2 gap-4 mb-4">
                    <div>
                        <h4 className="text-xs font-semibold mb-2 text-zinc-400">Original</h4>
                        <img
                            src={`${apiBase}/${currentImage}`}
                            alt="Original X-ray"
                            className="w-full rounded-lg border border-zinc-700"
                        />
                    </div>
                    <div>
                        <h4 className="text-xs font-semibold mb-2 text-zinc-400">Segmented</h4>
                        <img
                            src={`${apiBase}/${segImagePath}`}
                            alt="Segmented X-ray"
                            className="w-full rounded-lg border border-zinc-700"
                        />
                    </div>
                </div>
            )}

            {/* Organ Metrics */}
            {result.result.metrics && (
                <div>
                    <h4 className="text-sm font-semibold mb-3 text-zinc-400">Detected Organs:</h4>
                    <div className="grid grid-cols-2 gap-3">
                        {Object.entries(result.result.metrics).map(([organ, metrics]: [string, any]) => (
                            <div key={organ} className="bg-zinc-800/50 rounded-lg p-3 border border-zinc-700">
                                <div className="font-medium text-sm mb-1">{organ}</div>
                                <div className="text-xs text-zinc-400 space-y-1">
                                    <div>Area: <span className="text-white font-mono">{metrics.area_cm2?.toFixed(2)} cm²</span></div>
                                    <div>Confidence: <span className="text-white font-mono">{(metrics.confidence_score * 100).toFixed(1)}%</span></div>
                                </div>
                            </div>
                        ))}
                    </div>
                    <button
                        onClick={() => setShowRaw(!showRaw)}
                        className="mt-4 text-xs text-zinc-500 hover:text-zinc-300"
                    >
                        {showRaw ? 'Hide' : 'Show'} detailed metrics / raw data
                    </button>
                    {showRaw && (
                        <pre className="bg-zinc-950 p-4 rounded-lg text-xs overflow-x-auto mt-2">
                            {JSON.stringify(result, null, 2)}
                        </pre>
                    )}
                </div>
            )}
        </div>
    );
}


