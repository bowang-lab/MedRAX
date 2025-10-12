'use client';

import { useState } from 'react';
import { MessageSquare } from 'lucide-react';

interface VQAResultsProps {
    result: any;
    idx: number;
}

export default function VQAResults({ result, idx }: VQAResultsProps) {
    const [showRaw, setShowRaw] = useState(false);

    if (!result.result) {
        return null;
    }

    const response = result.result.response || '';

    return (
        <div className="space-y-4">
            {/* Header */}
            <div className="flex items-center gap-2 pb-2 border-b border-zinc-700">
                <MessageSquare className="h-5 w-5 text-green-400" />
                <h3 className="text-lg font-semibold">Visual Question Answering</h3>
            </div>

            {/* Response */}
            <div className="border border-zinc-700 rounded-lg p-4 bg-zinc-900/50">
                <p className="text-sm text-zinc-300 leading-relaxed whitespace-pre-wrap">
                    {response}
                </p>
            </div>

            {/* Metadata */}
            {result.metadata && (
                <div className="text-xs text-zinc-500 space-y-1 pt-2 border-t border-zinc-800">
                    {result.metadata.prompt && (
                        <div>Prompt: <span className="text-zinc-400">{result.metadata.prompt}</span></div>
                    )}
                    {result.metadata.max_new_tokens && (
                        <div>Max Tokens: <span className="text-zinc-400">{result.metadata.max_new_tokens}</span></div>
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

