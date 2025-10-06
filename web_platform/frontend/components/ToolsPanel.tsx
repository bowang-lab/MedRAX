import { useState, useEffect } from 'react';
import { Settings, CheckCircle, Circle, XCircle, Loader2, Download, Trash2, Info, AlertCircle, RefreshCw, HardDrive } from 'lucide-react';
import axios from 'axios';

const API_BASE = 'http://localhost:8000';

interface Tool {
    name: string;
    display_name: string;
    description: string;
    category: string;
    status: 'available' | 'loaded' | 'unavailable' | 'error';
    requires_model: boolean;
    model_size_gb: number;
    mac_compatible: boolean;
    dependencies: string[];
    error_message?: string;
    is_loaded: boolean;
    is_cached: boolean;
}

interface ToolsPanelProps {
    sessionId: string | null;
}

const categoryColors = {
    utility: 'bg-blue-500/10 text-blue-400 border-blue-500/30',
    analysis: 'bg-green-500/10 text-green-400 border-green-500/30',
    expert: 'bg-purple-500/10 text-purple-400 border-purple-500/30',
    generation: 'bg-yellow-500/10 text-yellow-400 border-yellow-500/30',
    knowledge: 'bg-pink-500/10 text-pink-400 border-pink-500/30',
};

const categoryIcons = {
    utility: '🔧',
    analysis: '🔬',
    expert: '👨‍⚕️',
    generation: '🎨',
    knowledge: '📚',
};

export default function ToolsPanel({ sessionId }: ToolsPanelProps) {
    const [tools, setTools] = useState<Record<string, Tool>>({});
    const [loadingTools, setLoadingTools] = useState<Set<string>>(new Set());
    const [selectedTools, setSelectedTools] = useState<Set<string>>(new Set());
    const [expandedTools, setExpandedTools] = useState<Set<string>>(new Set());
    const [recommendations, setRecommendations] = useState<any>(null);
    const [isRefreshing, setIsRefreshing] = useState(false);

    useEffect(() => {
        fetchTools();
    }, []);

    const fetchTools = async (showRefreshState = false) => {
        if (showRefreshState) setIsRefreshing(true);
        try {
            const response = await axios.get(`${API_BASE}/api/tools`);
            setTools(response.data.tools);
            setRecommendations(response.data.recommendations);
        } catch (error) {
            console.error('Failed to fetch tools:', error);
        } finally {
            if (showRefreshState) setIsRefreshing(false);
        }
    };

    const handleManualRefresh = () => {
        fetchTools(true);
    };

    const handleLoadTool = async (toolId: string) => {
        setLoadingTools(prev => new Set(prev).add(toolId));
        try {
            await axios.post(`${API_BASE}/api/tools/${toolId}/load`);
            await fetchTools();
        } catch (error: any) {
            console.error(`Failed to load tool ${toolId}:`, error);
            alert(`Failed to load tool: ${error.response?.data?.detail || error.message}`);
        } finally {
            setLoadingTools(prev => {
                const newSet = new Set(prev);
                newSet.delete(toolId);
                return newSet;
            });
        }
    };

    const handleUnloadTool = async (toolId: string) => {
        if (!confirm(`Unload ${tools[toolId].display_name}? This will free memory.`)) return;

        setLoadingTools(prev => new Set(prev).add(toolId));
        try {
            await axios.post(`${API_BASE}/api/tools/${toolId}/unload`);
            await fetchTools();
        } catch (error: any) {
            console.error(`Failed to unload tool ${toolId}:`, error);
            alert(`Failed to unload tool: ${error.response?.data?.detail || error.message}`);
        } finally {
            setLoadingTools(prev => {
                const newSet = new Set(prev);
                newSet.delete(toolId);
                return newSet;
            });
        }
    };

    const toggleToolSelection = (toolId: string) => {
        setSelectedTools(prev => {
            const newSet = new Set(prev);
            if (newSet.has(toolId)) {
                newSet.delete(toolId);
            } else {
                newSet.add(toolId);
            }
            return newSet;
        });
    };

    const toggleToolExpanded = (toolId: string) => {
        setExpandedTools(prev => {
            const newSet = new Set(prev);
            if (newSet.has(toolId)) {
                newSet.delete(toolId);
            } else {
                newSet.add(toolId);
            }
            return newSet;
        });
    };

    const runSelectedTools = async () => {
        if (!sessionId || selectedTools.size === 0) return;
        // TODO: Implement running selected tools
        alert(`Would run ${selectedTools.size} tools (implementation pending)`);
    };

    const getStatusIcon = (tool: Tool, isLoading: boolean) => {
        if (isLoading) return <Loader2 className="h-4 w-4 animate-spin text-blue-400" />;

        switch (tool.status) {
            case 'loaded':
                return <CheckCircle className="h-4 w-4 text-green-400" />;
            case 'available':
                return <Circle className="h-4 w-4 text-yellow-400" />;
            case 'unavailable':
                return <XCircle className="h-4 w-4 text-red-400" />;
            case 'error':
                return <AlertCircle className="h-4 w-4 text-orange-400" />;
            default:
                return <Circle className="h-4 w-4 text-zinc-500" />;
        }
    };

    const groupedTools = Object.entries(tools).reduce((acc, [id, tool]) => {
        if (!acc[tool.category]) acc[tool.category] = [];
        acc[tool.category].push({ id, ...tool });
        return acc;
    }, {} as Record<string, Array<Tool & { id: string }>>);

    return (
        <div className="h-full flex flex-col bg-zinc-900/50">
            {/* Header */}
            <div className="p-4 border-b border-zinc-800">
                <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2">
                        <Settings className="h-5 w-5 text-zinc-400" />
                        <h2 className="text-sm font-semibold text-zinc-300">MedRAX Tools</h2>
                    </div>
                    <button
                        onClick={handleManualRefresh}
                        disabled={isRefreshing}
                        className="p-1.5 hover:bg-zinc-700 rounded text-zinc-400 disabled:opacity-50"
                        title="Refresh tools status"
                    >
                        <RefreshCw className={`h-4 w-4 ${isRefreshing ? 'animate-spin' : ''}`} />
                    </button>
                </div>
                <div className="flex gap-2 text-xs text-zinc-500">
                    <span className="px-2 py-1 bg-green-500/10 text-green-400 rounded">
                        {Object.values(tools).filter(t => t.is_loaded).length} Loaded
                    </span>
                    <span className="px-2 py-1 bg-yellow-500/10 text-yellow-400 rounded">
                        {Object.values(tools).filter(t => t.status === 'available').length} Available
                    </span>
                    <span className="px-2 py-1 bg-red-500/10 text-red-400 rounded">
                        {Object.values(tools).filter(t => t.status === 'unavailable').length} Unavailable
                    </span>
                </div>
            </div>

            {/* Tools List */}
            <div className="flex-1 overflow-y-auto p-4 space-y-4">
                {Object.entries(groupedTools).map(([category, categoryTools]) => (
                    <div key={category}>
                        <div className="flex items-center gap-2 mb-2">
                            <span className="text-lg">{categoryIcons[category as keyof typeof categoryIcons]}</span>
                            <h3 className="text-xs font-semibold text-zinc-400 uppercase tracking-wide">
                                {category}
                            </h3>
                        </div>

                        <div className="space-y-2">
                            {categoryTools.map((tool) => {
                                const isLoading = loadingTools.has(tool.id);
                                const isExpanded = expandedTools.has(tool.id);
                                const isSelected = selectedTools.has(tool.id);

                                return (
                                    <div
                                        key={tool.id}
                                        className={`
                                            bg-zinc-800/50 border rounded-lg overflow-hidden transition-all
                                            ${isSelected ? 'border-blue-500' : 'border-zinc-700 hover:border-zinc-600'}
                                        `}
                                    >
                                        {/* Tool Header */}
                                        <div className="p-3">
                                            <div className="flex items-start gap-3">
                                                <div className="mt-0.5">
                                                    {getStatusIcon(tool, isLoading)}
                                                </div>

                                                <div className="flex-1 min-w-0">
                                                    <div className="flex items-center gap-2 mb-1">
                                                        <h4 className="text-sm font-medium text-white truncate">
                                                            {tool.display_name}
                                                        </h4>
                                                        {tool.requires_model && (
                                                            <span className="text-xs text-zinc-500">
                                                                {tool.model_size_gb.toFixed(1)}GB
                                                            </span>
                                                        )}
                                                        {tool.is_cached && (
                                                            <span
                                                                className="flex items-center gap-1 px-1.5 py-0.5 bg-green-500/10 text-green-400 text-xs rounded border border-green-500/30"
                                                                title="Model already downloaded"
                                                            >
                                                                <HardDrive className="h-3 w-3" />
                                                                Cached
                                                            </span>
                                                        )}
                                                    </div>

                                                    <p className="text-xs text-zinc-400 line-clamp-2">
                                                        {tool.description}
                                                    </p>
                                                </div>

                                                {/* Action Button */}
                                                <div className="flex gap-1">
                                                    {tool.status === 'loaded' && (
                                                        <button
                                                            onClick={() => handleUnloadTool(tool.id)}
                                                            disabled={isLoading}
                                                            className="p-1.5 hover:bg-red-500/10 rounded text-red-400 disabled:opacity-50"
                                                            title="Unload tool"
                                                        >
                                                            <Trash2 className="h-3.5 w-3.5" />
                                                        </button>
                                                    )}

                                                    {tool.status === 'available' && (
                                                        <button
                                                            onClick={() => handleLoadTool(tool.id)}
                                                            disabled={isLoading}
                                                            className="p-1.5 hover:bg-green-500/10 rounded text-green-400 disabled:opacity-50"
                                                            title="Load tool"
                                                        >
                                                            <Download className="h-3.5 w-3.5" />
                                                        </button>
                                                    )}

                                                    <button
                                                        onClick={() => toggleToolExpanded(tool.id)}
                                                        className="p-1.5 hover:bg-zinc-700 rounded text-zinc-400"
                                                        title="More info"
                                                    >
                                                        <Info className="h-3.5 w-3.5" />
                                                    </button>
                                                </div>
                                            </div>

                                            {/* Error Message */}
                                            {tool.error_message && (
                                                <div className="mt-2 p-2 bg-red-500/10 border border-red-500/30 rounded text-xs text-red-300">
                                                    {tool.error_message}
                                                </div>
                                            )}
                                        </div>

                                        {/* Expanded Details */}
                                        {isExpanded && (
                                            <div className="px-3 pb-3 border-t border-zinc-700 bg-zinc-900/30">
                                                <div className="mt-2 space-y-2 text-xs">
                                                    <div>
                                                        <span className="text-zinc-500">Dependencies:</span>
                                                        <div className="flex flex-wrap gap-1 mt-1">
                                                            {tool.dependencies.map(dep => (
                                                                <span key={dep} className="px-2 py-0.5 bg-zinc-800 rounded text-zinc-400">
                                                                    {dep}
                                                                </span>
                                                            ))}
                                                        </div>
                                                    </div>

                                                    <div className="flex gap-2">
                                                        <span className={`px-2 py-0.5 rounded ${categoryColors[tool.category as keyof typeof categoryColors]}`}>
                                                            {tool.category}
                                                        </span>
                                                        {!tool.mac_compatible && (
                                                            <span className="px-2 py-0.5 bg-orange-500/10 text-orange-400 border border-orange-500/30 rounded">
                                                                ⚠️ Mac incompatible
                                                            </span>
                                                        )}
                                                    </div>
                                                </div>
                                            </div>
                                        )}
                                    </div>
                                );
                            })}
                        </div>
                    </div>
                ))}
            </div>

            {/* Footer Actions */}
            <div className="p-4 border-t border-zinc-800 bg-zinc-900">
                <button
                    onClick={runSelectedTools}
                    disabled={selectedTools.size === 0 || !sessionId}
                    className="w-full px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-zinc-700 disabled:text-zinc-500 rounded-lg text-sm font-medium transition-colors"
                >
                    {selectedTools.size > 0
                        ? `Run ${selectedTools.size} Selected Tool${selectedTools.size > 1 ? 's' : ''}`
                        : 'Select Tools to Run'}
                </button>
            </div>
        </div>
    );
}

