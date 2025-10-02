'use client';

import { useState } from 'react';
import { ChevronRight } from 'lucide-react';

interface ClassificationResultsProps {
    result: any;
    idx: number;
    apiBase: string;
}

export default function ClassificationResults({ result, idx, apiBase }: ClassificationResultsProps) {
    const [showRaw, setShowRaw] = useState(false);

    console.log('ClassificationResults received:', result);

    if (!result) {
        return <div className="text-sm text-zinc-500">No classification data available (result is null/undefined)</div>;
    }

    if (!result.result) {
        return (
            <div className="space-y-2">
                <div className="text-sm text-zinc-500">No classification result data</div>
                <pre className="text-xs bg-zinc-950 p-2 rounded overflow-x-auto">
                    {JSON.stringify(result, null, 2)}
                </pre>
            </div>
        );
    }

    if (typeof result.result !== 'object' || result.result === null) {
        return (
            <div className="space-y-2">
                <div className="text-sm text-zinc-500">
                    Invalid classification data format (got {typeof result.result})
                </div>
                <pre className="text-xs bg-zinc-950 p-2 rounded overflow-x-auto">
                    {JSON.stringify(result, null, 2)}
                </pre>
            </div>
        );
    }

    // Check if Grad-CAM visualization is available
    const gradcamPath = result.metadata?.gradcam_image_path;

    return (
        <div className="space-y-4">
            {/* Grad-CAM Visualization */}
            {gradcamPath && (
                <div className="mb-4">
                    <h4 className="text-sm font-semibold mb-2 text-zinc-400">Attention Heatmaps (Grad-CAM)</h4>
                    <p className="text-xs text-zinc-500 mb-3">
                        Shows where the AI model focuses to detect each pathology. Warmer colors (red/yellow) indicate stronger attention.
                    </p>
                    <img
                        src={`${apiBase}/${gradcamPath}`}
                        alt="Grad-CAM Heatmaps"
                        className="w-full rounded-lg border border-zinc-700"
                    />
                </div>
            )}

            <div className="space-y-2">
                {Object.entries(result.result)
                    .filter(([_, prob]) => typeof prob === 'number')
                    .sort(([, a], [, b]) => (b as number) - (a as number))
                    .slice(0, 8)
                    .map(([pathology, prob]) => {
                        const probability = (prob as number) * 100;
                        const color = probability > 60 ? 'bg-red-500' : probability > 40 ? 'bg-yellow-500' : 'bg-blue-500';
                        return (
                            <div key={pathology} className="flex items-center gap-3">
                                <span className="w-48 text-sm text-zinc-300">{pathology.replace(/_/g, ' ')}</span>
                                <div className="flex-1 bg-zinc-800 rounded-full h-6 overflow-hidden">
                                    <div
                                        className={`${color} h-6 rounded-full transition-all duration-500 flex items-center justify-end pr-2`}
                                        style={{ width: `${probability}%` }}
                                    >
                                        {probability > 15 && (
                                            <span className="text-xs font-medium text-white">{probability.toFixed(1)}%</span>
                                        )}
                                    </div>
                                </div>
                                {probability <= 15 && (
                                    <span className="text-xs font-mono text-zinc-400 w-16">{probability.toFixed(1)}%</span>
                                )}
                            </div>
                        );
                    })}
            </div>
            <button
                onClick={() => setShowRaw(!showRaw)}
                className="mt-4 text-xs text-zinc-500 hover:text-zinc-300"
            >
                {showRaw ? 'Hide' : 'Show'} all pathologies / raw data
            </button>
            {showRaw && (
                <pre className="bg-zinc-950 p-4 rounded-lg text-xs overflow-x-auto mt-2">
                    {JSON.stringify(result, null, 2)}
                </pre>
            )}
        </div>
    );
}



