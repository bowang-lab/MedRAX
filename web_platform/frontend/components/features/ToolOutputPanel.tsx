'use client';

import { useState } from 'react';
import { Settings, ChevronDown, ChevronUp } from 'lucide-react';
import ClassificationResults from './ClassificationResults';
import SegmentationResults from './SegmentationResults';
import ReportResults from './ReportResults';
import GroundingResults from './GroundingResults';
import VQAResults from './VQAResults';

interface ToolOutputPanelProps {
    analysisResults: any[];
    currentImage: string | null;
    collapsed: boolean;
    apiBase: string;
    onToggleCollapse: () => void;
}

export default function ToolOutputPanel({
    analysisResults,
    currentImage,
    collapsed,
    apiBase,
    onToggleCollapse
}: ToolOutputPanelProps) {
    const [collapsedTools, setCollapsedTools] = useState<Set<number>>(new Set());

    const toggleToolCollapse = (index: number) => {
        const newCollapsed = new Set(collapsedTools);
        if (newCollapsed.has(index)) {
            newCollapsed.delete(index);
        } else {
            newCollapsed.add(index);
        }
        setCollapsedTools(newCollapsed);
    };

    // Filter out image_visualizer tool
    const filteredResults = analysisResults.filter(([toolName, _]: [string, any]) => {
        const isUtility = toolName.includes('visualizer') || toolName.includes('image_visualizer');
        return !isUtility;
    });

    return (
        <div className="h-full flex flex-col overflow-hidden">
            {/* Header - Removed collapse button as it's handled by parent */}
            <div className="p-4 border-b border-zinc-800/50">
                <div className="flex items-center justify-between">
                    <h2 className="text-sm font-semibold text-zinc-300 flex items-center gap-2">
                        <Settings className="h-4 w-4 text-emerald-400" />
                        Analysis Results
                    </h2>
                </div>
                {filteredResults.length > 0 && (
                    <div className="mt-2 text-xs text-zinc-500">
                        {filteredResults.length} tool{filteredResults.length !== 1 ? 's' : ''} executed
                    </div>
                )}
            </div>

            {/* Tool Results */}
            <div className="flex-1 overflow-y-auto p-4 space-y-4">
                {filteredResults.length === 0 ? (
                    <div className="text-center text-zinc-500 text-sm mt-8">
                        <Settings className="h-12 w-12 mx-auto mb-3 text-zinc-600" />
                        <p>No results yet.</p>
                        <p className="mt-1 text-xs">Run analysis or chat with AI to see results here.</p>
                    </div>
                ) : (
                    filteredResults.map(([toolName, result]: [string, any], idx) => {
                        const isClassification = toolName.includes('classification') || toolName.includes('classifier');
                        const isSegmentation = toolName.includes('segmentation');
                        const isReport = toolName.includes('report');
                        const isGrounding = toolName.includes('grounding');
                        const isExpert = toolName.includes('expert') || toolName.includes('vqa');

                        const isCollapsed = collapsedTools.has(idx);

                        return (
                            <div
                                key={idx}
                                className="bg-gradient-to-br from-zinc-900/80 to-zinc-900/50 backdrop-blur-sm border border-zinc-800/50 rounded-xl overflow-hidden shadow-lg hover:shadow-xl transition-all duration-300 hover:border-zinc-700/50"
                            >
                                <div
                                    onClick={() => toggleToolCollapse(idx)}
                                    className="flex items-center justify-between p-4 cursor-pointer hover:bg-zinc-800/30 transition-colors"
                                >
                                    <h3 className="text-sm font-semibold flex items-center gap-2">
                                        <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></div>
                                        {toolName.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                                    </h3>
                                    <button className="p-1 hover:bg-zinc-700 rounded-lg transition-colors">
                                        {isCollapsed ? <ChevronDown className="h-4 w-4" /> : <ChevronUp className="h-4 w-4" />}
                                    </button>
                                </div>

                                {!isCollapsed && (
                                    <div className="px-4 pb-4">
                                        {isClassification && (
                                            <ClassificationResults
                                                result={result}
                                                idx={idx}
                                                apiBase={apiBase}
                                            />
                                        )}

                                        {isSegmentation && (
                                            <SegmentationResults
                                                result={result}
                                                idx={idx}
                                                currentImage={currentImage}
                                                apiBase={apiBase}
                                            />
                                        )}

                                        {isReport && (
                                            <ReportResults
                                                result={result}
                                                idx={idx}
                                            />
                                        )}

                                        {isGrounding && (
                                            <GroundingResults
                                                result={result}
                                                idx={idx}
                                                apiBase={apiBase}
                                            />
                                        )}

                                        {isExpert && (
                                            <VQAResults
                                                result={result}
                                                idx={idx}
                                            />
                                        )}

                                        {!isClassification && !isSegmentation && !isReport && !isExpert && !isGrounding && (
                                            <div className="text-sm text-zinc-300">
                                                {typeof result === 'string' ? result : JSON.stringify(result, null, 2)}
                                            </div>
                                        )}
                                    </div>
                                )}
                            </div>
                        );
                    })
                )}
            </div>
        </div>
    );
}

