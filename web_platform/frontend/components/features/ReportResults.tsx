'use client';

import { useState } from 'react';
import { FileText, ChevronDown, ChevronUp } from 'lucide-react';

interface ReportResultsProps {
    result: any;
    idx: number;
}

export default function ReportResults({ result, idx }: ReportResultsProps) {
    const [showRaw, setShowRaw] = useState(false);
    const [expandedSection, setExpandedSection] = useState<'findings' | 'impression' | null>('findings');

    if (!result.result) {
        return null;
    }

    // Try to parse the result string if it contains "CHEST X-RAY REPORT"
    let findings = '';
    let impression = '';

    if (typeof result.result === 'string') {
        const reportText = result.result;

        // Extract FINDINGS section
        const findingsMatch = reportText.match(/FINDINGS:\s*([\s\S]*?)(?=\n\nIMPRESSION:|$)/i);
        if (findingsMatch) {
            findings = findingsMatch[1].trim();
        }

        // Extract IMPRESSION section
        const impressionMatch = reportText.match(/IMPRESSION:\s*([\s\S]*?)$/i);
        if (impressionMatch) {
            impression = impressionMatch[1].trim();
        }

        // If no sections found, use the whole text as findings
        if (!findings && !impression) {
            findings = reportText;
        }
    } else if (typeof result.result === 'object') {
        findings = result.result.findings || result.result.Findings || '';
        impression = result.result.impression || result.result.Impression || '';
    }

    return (
        <div className="space-y-4">
            {/* Report Header */}
            <div className="flex items-center justify-between pb-2 border-b border-zinc-700">
                <div className="flex items-center gap-2">
                    <FileText className="h-5 w-5 text-blue-400" />
                    <h3 className="text-lg font-semibold">Radiology Report</h3>
                </div>
                {result.metadata && (
                    <div className="flex gap-4 text-xs text-zinc-500">
                        {result.metadata.model && (
                            <div>Model: <span className="text-zinc-400">{result.metadata.model}</span></div>
                        )}
                        {result.metadata.total_time_seconds && (
                            <div>Time: <span className="text-zinc-400">{result.metadata.total_time_seconds.toFixed(2)}s</span></div>
                        )}
                    </div>
                )}
            </div>

            {/* Findings Section */}
            {findings && (
                <div className="border border-zinc-700 rounded-lg overflow-hidden">
                    <button
                        onClick={() => setExpandedSection(expandedSection === 'findings' ? null : 'findings')}
                        className="w-full px-4 py-3 bg-zinc-800/50 hover:bg-zinc-800 transition-colors flex items-center justify-between"
                    >
                        <div className="flex items-center gap-2">
                            <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                            <span className="font-semibold text-sm uppercase tracking-wide">Findings</span>
                        </div>
                        {expandedSection === 'findings' ?
                            <ChevronUp className="h-4 w-4" /> :
                            <ChevronDown className="h-4 w-4" />
                        }
                    </button>
                    {expandedSection === 'findings' && (
                        <div className="p-4 bg-zinc-900/50">
                            <p className="text-sm text-zinc-300 leading-relaxed whitespace-pre-wrap">
                                {findings}
                            </p>
                        </div>
                    )}
                </div>
            )}

            {/* Impression Section */}
            {impression && (
                <div className="border border-zinc-700 rounded-lg overflow-hidden">
                    <button
                        onClick={() => setExpandedSection(expandedSection === 'impression' ? null : 'impression')}
                        className="w-full px-4 py-3 bg-zinc-800/50 hover:bg-zinc-800 transition-colors flex items-center justify-between"
                    >
                        <div className="flex items-center gap-2">
                            <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                            <span className="font-semibold text-sm uppercase tracking-wide">Impression</span>
                        </div>
                        {expandedSection === 'impression' ?
                            <ChevronUp className="h-4 w-4" /> :
                            <ChevronDown className="h-4 w-4" />
                        }
                    </button>
                    {expandedSection === 'impression' && (
                        <div className="p-4 bg-zinc-900/50">
                            <p className="text-sm text-zinc-300 leading-relaxed whitespace-pre-wrap">
                                {impression}
                            </p>
                        </div>
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

