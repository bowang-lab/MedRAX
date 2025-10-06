import { CheckCircle, Loader2, Circle, XCircle, ChevronRight } from 'lucide-react';

interface PipelineStep {
    name: string;
    display_name: string;
    status: 'pending' | 'running' | 'completed' | 'error';
    duration?: number;
    error?: string;
}

interface PipelineVisualizationProps {
    steps: PipelineStep[];
    isActive: boolean;
}

const statusIcons = {
    pending: <Circle className="h-4 w-4 text-zinc-500" />,
    running: <Loader2 className="h-4 w-4 text-blue-400 animate-spin" />,
    completed: <CheckCircle className="h-4 w-4 text-green-400" />,
    error: <XCircle className="h-4 w-4 text-red-400" />,
};

const statusColors = {
    pending: 'bg-zinc-800 text-zinc-500',
    running: 'bg-blue-500/20 text-blue-300 border-blue-500/50',
    completed: 'bg-green-500/20 text-green-300 border-green-500/50',
    error: 'bg-red-500/20 text-red-300 border-red-500/50',
};

export default function PipelineVisualization({ steps, isActive }: PipelineVisualizationProps) {
    if (steps.length === 0 && !isActive) {
        return null;
    }

    return (
        <div className="bg-zinc-900 border-t border-zinc-800">
            <div className="p-4">
                <div className="flex items-center gap-2 mb-3">
                    <div className={`h-2 w-2 rounded-full ${isActive ? 'bg-blue-400 animate-pulse' : 'bg-zinc-600'}`} />
                    <h3 className="text-sm font-semibold text-zinc-400">
                        {isActive ? 'Analysis Pipeline' : 'Pipeline Ready'}
                    </h3>
                </div>

                {/* Pipeline Flow */}
                <div className="flex items-center gap-2 overflow-x-auto pb-2">
                    {steps.length === 0 ? (
                        <div className="text-xs text-zinc-500 italic">
                            No pipeline steps to display
                        </div>
                    ) : (
                        steps.map((step, idx) => (
                            <div key={step.name} className="flex items-center gap-2">
                                {/* Step Card */}
                                <div
                                    className={`
                                        flex items-center gap-2 px-3 py-2 rounded-lg border
                                        transition-all duration-200 min-w-max
                                        ${statusColors[step.status]}
                                    `}
                                >
                                    <div className="flex-shrink-0">
                                        {statusIcons[step.status]}
                                    </div>

                                    <div className="min-w-0">
                                        <div className="text-xs font-medium truncate">
                                            {step.display_name}
                                        </div>

                                        {step.duration && (
                                            <div className="text-xs text-zinc-500 mt-0.5">
                                                {step.duration.toFixed(1)}s
                                            </div>
                                        )}

                                        {step.error && (
                                            <div className="text-xs text-red-400 mt-0.5 truncate max-w-32">
                                                {step.error}
                                            </div>
                                        )}
                                    </div>
                                </div>

                                {/* Arrow */}
                                {idx < steps.length - 1 && (
                                    <ChevronRight className="h-4 w-4 text-zinc-600 flex-shrink-0" />
                                )}
                            </div>
                        ))
                    )}
                </div>

                {/* Progress Bar */}
                {isActive && steps.length > 0 && (
                    <div className="mt-3">
                        <div className="h-1 bg-zinc-800 rounded-full overflow-hidden">
                            <div
                                className="h-full bg-blue-500 transition-all duration-300 rounded-full"
                                style={{
                                    width: `${(steps.filter(s => s.status === 'completed').length / steps.length) * 100}%`
                                }}
                            />
                        </div>
                        <div className="text-xs text-zinc-500 mt-1 text-center">
                            {steps.filter(s => s.status === 'completed').length} / {steps.length} steps completed
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}

