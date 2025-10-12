'use client';

import { useState } from 'react';
import { Target, ChevronDown, ChevronUp } from 'lucide-react';

interface GroundingResultsProps {
    result: any;
    idx: number;
    apiBase: string;
}

export default function GroundingResults({ result, idx, apiBase }: GroundingResultsProps) {
    const [showRaw, setShowRaw] = useState(false);
    const [showBoxes, setShowBoxes] = useState(true);

    if (!result.result) {
        return null;
    }

    const vizPath = result.result.visualization_path;
    const predictions = result.result.predictions || [];
    const hasFindings = predictions.length > 0;

    return (
        <div className="space-y-4">
            {/* Header */}
            <div className="flex items-center gap-2 pb-2 border-b border-zinc-700">
                <Target className="h-5 w-5 text-purple-400" />
                <h3 className="text-lg font-semibold">Phrase Grounding</h3>
                {!hasFindings && (
                    <span className="text-xs text-zinc-500 ml-2">(No findings detected)</span>
                )}
            </div>

            {/* Visualization Image */}
            {vizPath && (
                <div className="mb-4">
                    <h4 className="text-sm font-semibold mb-2 text-zinc-400">Annotated Image</h4>
                    <p className="text-xs text-zinc-500 mb-3">
                        Red bounding boxes show the detected location of the medical finding.
                    </p>
                    <img
                        src={`${apiBase}/${vizPath}`}
                        alt="Grounding Visualization"
                        className="w-full rounded-lg border border-zinc-700"
                    />
                </div>
            )}

            {/* Bounding Box Details */}
            {hasFindings && (
                <div className="border border-zinc-700 rounded-lg overflow-hidden">
                    <button
                        onClick={() => setShowBoxes(!showBoxes)}
                        className="w-full px-4 py-3 bg-zinc-800/50 hover:bg-zinc-800 transition-colors flex items-center justify-between"
                    >
                        <div className="flex items-center gap-2">
                            <div className="w-2 h-2 bg-purple-500 rounded-full"></div>
                            <span className="font-semibold text-sm uppercase tracking-wide">
                                Detected Regions ({predictions.length})
                            </span>
                        </div>
                        {showBoxes ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
                    </button>
                    {showBoxes && (
                        <div className="p-4 bg-zinc-900/50 space-y-3">
                            {predictions.map((pred: any, i: number) => (
                                <div key={i} className="text-sm">
                                    <div className="text-zinc-300 font-medium mb-1">
                                        {pred.phrase}
                                    </div>
                                    <div className="text-xs text-zinc-500">
                                        {pred.bounding_boxes?.image_coordinates?.length || 0} bounding box(es)
                                    </div>
                                    {pred.bounding_boxes?.image_coordinates?.map((box: number[], boxIdx: number) => (
                                        <div key={boxIdx} className="font-mono text-xs text-zinc-600 mt-1">
                                            Box {boxIdx + 1}: [{box.map(v => v.toFixed(2)).join(', ')}]
                                        </div>
                                    ))}
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            )}

            {/* Metadata */}
            {result.metadata && (
                <div className="text-xs text-zinc-500 space-y-1 pt-2 border-t border-zinc-800">
                    {result.metadata.original_size && (
                        <div>Image Size: <span className="text-zinc-400">{result.metadata.original_size[0]} × {result.metadata.original_size[1]}</span></div>
                    )}
                    {result.metadata.analysis_status && (
                        <div>Status: <span className="text-zinc-400">{result.metadata.analysis_status}</span></div>
                    )}
                </div>
            )}

            {/* Raw JSON Toggle */}
            <button
                onClick={() => setShowRaw(!showRaw)}
                className="text-xs text-zinc-500 hover:text-zinc-300"
            >
                {showRaw ? 'Hide' : 'Show'} raw data
            </button>
            {showRaw && (
                <pre className="bg-zinc-950 p-4 rounded-lg text-xs overflow-x-auto">
                    {JSON.stringify(result, null, 2)}
                </pre>
            )}
        </div>
    );
}

