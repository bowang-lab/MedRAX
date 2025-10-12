'use client';

interface AnalysisProgressProps {
    isAnalyzing: boolean;
}

export default function AnalysisProgress({ isAnalyzing }: AnalysisProgressProps) {
    if (!isAnalyzing) return null;

    return (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50">
            <div className="bg-zinc-900 border border-zinc-700 rounded-lg p-8 max-w-md">
                <div className="flex flex-col items-center gap-4">
                    {/* Spinner */}
                    <div className="relative">
                        <div className="w-16 h-16 border-4 border-zinc-700 border-t-blue-500 rounded-full animate-spin"></div>
                        <div className="absolute inset-0 flex items-center justify-center">
                            <div className="w-8 h-8 bg-blue-500/20 rounded-full animate-pulse"></div>
                        </div>
                    </div>

                    {/* Text */}
                    <div className="text-center">
                        <h3 className="text-lg font-semibold text-white mb-2">
                            Analyzing X-Ray...
                        </h3>
                        <p className="text-sm text-zinc-400">
                            Running AI tools • This may take 1-2 minutes
                        </p>
                    </div>

                    {/* Progress steps */}
                    <div className="w-full space-y-2 mt-4">
                        <div className="flex items-center gap-2 text-xs text-zinc-500">
                            <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse"></div>
                            <span>Classifying pathologies...</span>
                        </div>
                        <div className="flex items-center gap-2 text-xs text-zinc-500">
                            <div className="w-2 h-2 bg-zinc-600 rounded-full"></div>
                            <span>Segmenting anatomy...</span>
                        </div>
                        <div className="flex items-center gap-2 text-xs text-zinc-500">
                            <div className="w-2 h-2 bg-zinc-600 rounded-full"></div>
                            <span>Generating analysis...</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}


